package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestPhase8ResponseReplayBuildsExactMultiTurnRequest(t *testing.T) {
	response := []byte(`{"id":"msg_1","model":"claude-3-haiku-20240307","role":"assistant","stop_reason":"tool_use","usage":{},"content":[{"type":"text","text":"I will look that up.","citations":[{"type":"web_search_result_location","url":"https://example.com","title":"Example","encrypted_index":"index","cited_text":"quoted"}],"future_text":{"kept":true}},{"type":"tool_use","id":"client","name":"local_weather","input":null,"provider_metadata":{"discard":true}},{"type":"server_tool_use","id":"search","name":"web_search","input":{"query":"Anthropic"},"dynamic":true,"caller":{"type":"code_execution_20250825","tool_id":"code","future":"caller"},"future_server":1,"provider_metadata":{"discard":true}},{"type":"web_search_tool_result","tool_use_id":"search","caller":{"type":"direct"},"content":[{"type":"web_search_result","url":"https://example.com","title":"Example","encrypted_content":"cipher","page_age":"today","future_result":true}]},{"type":"mcp_tool_use","id":"mcp","name":"lookup","server_name":"catalog","input":{"term":"sdk"},"dynamic":true,"future_mcp":"kept"},{"type":"mcp_tool_result","tool_use_id":"mcp","content":[{"type":"text","text":"found"}],"dynamic":true,"future_mcp_result":2}]}`)
	result, err := parseGenerateResponse(response)
	if err != nil {
		t.Fatal(err)
	}

	contents := make([]AssistantContent, 0, len(result.Content))
	for _, content := range result.Content {
		assistantContent, ok := content.(AssistantContent)
		if !ok {
			t.Fatalf("response content %T is not assistant content", content)
		}
		contents = append(contents, assistantContent)
	}
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{
		MCPServers: []MCPServer{{Name: "catalog", URL: "https://mcp.example", ToolConfiguration: &ToolConfiguration{AllowedTools: []string{"lookup"}}}},
	}).(*anthropicLanguageModel)
	request, _, _, _, err := model.buildRequestWithOptions(GenerateOptions{
		MaxTokens: 64,
		Messages:  []Message{UserMessage{Content: []UserContent{TextContent{Text: "Research this"}}}, AssistantMessage{Content: contents}},
		Tools:     []Tool{{ID: "anthropic.web_search_20250305", MaxUses: intPtr(2)}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":[{"text":"Research this","type":"text"}]},{"role":"assistant","content":[{"citations":[{"type":"web_search_result_location","cited_text":"quoted","url":"https://example.com","title":"Example","encrypted_index":"index"}],"future_text":{"kept":true},"text":"I will look that up.","type":"text"},{"caller":{"future":"caller","tool_id":"code","type":"code_execution_20250825"},"dynamic":true,"future_server":1,"id":"search","input":{"query":"Anthropic"},"name":"web_search","type":"server_tool_use"},{"caller":{"type":"direct"},"content":[{"encrypted_content":"cipher","future_result":true,"page_age":"today","title":"Example","type":"web_search_result","url":"https://example.com"}],"tool_use_id":"search","type":"web_search_tool_result"},{"dynamic":true,"future_mcp":"kept","id":"mcp","input":{"term":"sdk"},"name":"lookup","server_name":"catalog","type":"mcp_tool_use"},{"content":[{"text":"found","type":"text"}],"dynamic":true,"future_mcp_result":2,"tool_use_id":"mcp","type":"mcp_tool_result"},{"id":"client","input":{},"name":"local_weather","type":"tool_use"}]}],"max_tokens":64,"tools":[{"name":"web_search","type":"web_search_20250305","max_uses":2},{"type":"mcp_toolset","mcp_server_name":"catalog","default_config":{"enabled":false},"configs":{"lookup":{"enabled":true}}}],"mcp_servers":[{"name":"catalog","url":"https://mcp.example"}],"output_config":{"effort":"high"}}`
	if string(got) != want {
		t.Fatalf("request = %s\nwant = %s", got, want)
	}
	if strings.Contains(string(got), "provider_metadata") {
		t.Fatalf("outbound request contains provider_metadata: %s", got)
	}
}

func TestPhase8ResponseReplayIncludesSubsequentClientToolResult(t *testing.T) {
	response := []byte(`{"id":"msg_1","model":"claude-3-haiku-20240307","role":"assistant","usage":{},"content":[{"type":"server_tool_use","id":"search","name":"web_search","input":{"query":"Anthropic"}},{"type":"web_search_tool_result","tool_use_id":"search","content":[{"type":"web_search_result","url":"https://example.com","title":"Example","encrypted_content":"cipher"}]},{"type":"tool_use","id":"client","name":"local_weather","input":null}]}`)
	result, err := parseGenerateResponse(response)
	if err != nil {
		t.Fatal(err)
	}
	contents := make([]AssistantContent, 0, len(result.Content))
	for _, content := range result.Content {
		assistantContent, ok := content.(AssistantContent)
		if !ok {
			t.Fatalf("response content %T is not assistant content", content)
		}
		contents = append(contents, assistantContent)
	}
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	request, _, _, _, err := model.buildRequestWithOptions(GenerateOptions{
		MaxTokens: 64,
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "Research this"}}},
			AssistantMessage{Content: contents},
			UserMessage{Content: []UserContent{ToolResultContent{ToolCallID: "client", Result: []ToolResultPart{ToolResultText{Text: "72F"}}}}},
		},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":[{"text":"Research this","type":"text"}]},{"role":"assistant","content":[{"id":"search","input":{"query":"Anthropic"},"name":"web_search","type":"server_tool_use"},{"content":[{"encrypted_content":"cipher","title":"Example","type":"web_search_result","url":"https://example.com"}],"tool_use_id":"search","type":"web_search_tool_result"},{"id":"client","input":{},"name":"local_weather","type":"tool_use"}]},{"role":"user","content":[{"content":[{"text":"72F","type":"text"}],"tool_use_id":"client","type":"tool_result"}]}],"max_tokens":64,"output_config":{"effort":"high"}}`
	if string(got) != want {
		t.Fatalf("request = %s\nwant = %s", got, want)
	}
}

func TestPhase8CapturedStreamPreservesLifecycleMetadataAndUsage(t *testing.T) {
	sse := event("message_start", `{"message":{"id":"msg-stream","model":"claude-3-haiku-20240307","usage":{"input_tokens":10,"cache_read_input_tokens":2}}}`) +
		event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) +
		event("content_block_delta", `{"index":0,"delta":{"type":"text_delta","text":"Answer","citations":[{"type":"web_search_result_location","url":"https://example.com","encrypted_index":"idx"}]}}`) +
		event("content_block_stop", `{"index":0}`) +
		event("content_block_start", `{"index":1,"content_block":{"type":"thinking"}}`) +
		event("content_block_delta", `{"index":1,"delta":{"type":"thinking_delta","thinking":"reason"}}`) +
		event("content_block_delta", `{"index":1,"delta":{"type":"signature_delta","signature":"sig"}}`) +
		event("content_block_stop", `{"index":1}`) +
		event("content_block_start", `{"index":2,"content_block":{"type":"tool_use","id":"client","name":"weather"}}`) +
		event("content_block_delta", `{"index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}}`) +
		event("content_block_stop", `{"index":2}`) +
		event("content_block_start", `{"index":3,"content_block":{"type":"source","id":"source","url":"https://source.example","title":"Source","media_type":"text/html","filename":"source.html"}}`) +
		event("content_block_stop", `{"index":3}`) +
		event("content_block_start", `{"index":4,"content_block":{"type":"web_search_tool_result","tool_use_id":"search","content":[{"type":"web_search_result","url":"https://example.com","encrypted_content":"cipher"}]}}`) +
		event("content_block_stop", `{"index":4}`) +
		event("content_block_start", `{"index":5,"content_block":{"type":"code_execution_tool_result","tool_use_id":"code","content":[{"type":"code_execution_result","stdout":"ok","return_code":0}]}}`) +
		event("content_block_stop", `{"index":5}`) +
		event("message_delta", `{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7,"output_tokens_details":{"thinking_tokens":3}}}`) +
		event("message_stop", `{}`)
	model := CreateAnthropic(ProviderSettings{Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(sse))}, nil
	}}}).Model("claude-3-haiku-20240307")
	stream, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var parts []StreamPart
	for part := range stream.Parts {
		if failure, ok := part.(StreamError); ok {
			t.Fatalf("StreamError = %v", failure.Err)
		}
		parts = append(parts, part)
	}
	var starts, ends, terminals int
	var citation StreamTextDelta
	var call StreamToolCall
	var finish StreamFinish
	var source StreamSource
	var toolResults []StreamToolResult
	var order []string
	for _, part := range parts {
		order = append(order, fmt.Sprintf("%T", part))
		switch value := part.(type) {
		case StreamTextStart, StreamReasoningStart, StreamToolInputStart:
			starts++
		case StreamTextEnd, StreamReasoningEnd, StreamToolInputEnd:
			ends++
		case StreamTextDelta:
			citation = value
		case StreamToolCall:
			call = value
		case StreamFinish:
			terminals++
			finish = value
		case StreamSource:
			source = value
		case StreamToolResult:
			toolResults = append(toolResults, value)
		}
	}
	if starts != 3 || ends != 3 || terminals != 1 {
		t.Fatalf("lifecycle starts=%d ends=%d terminals=%d parts=%#v", starts, ends, terminals, parts)
	}
	if stream.Response.ID != "msg-stream" || stream.Response.ModelID != "claude-3-haiku-20240307" {
		t.Fatalf("response metadata = %#v", stream.Response)
	}
	if len(citation.Citations) != 1 || citation.Citations[0].URL != "https://example.com" || citation.Citations[0].EncryptedIndex != "idx" {
		t.Fatalf("citation = %#v", citation)
	}
	if call.ToolCallID != "client" || call.ToolName != "weather" || string(call.Input) != `{"city":"SF"}` {
		t.Fatalf("tool call = %#v", call)
	}
	if finish.Usage.InputTokens.Total != 12 || finish.Usage.InputTokens.CacheRead != 2 || finish.Usage.OutputTokens.Total != 7 || finish.Usage.OutputTokens.Reasoning != 3 || finish.Usage.OutputTokens.Text != 4 || finish.Usage.TotalTokens != 19 || !strings.Contains(string(finish.Usage.Raw), `"thinking_tokens":3`) {
		t.Fatalf("usage = %#v", finish.Usage)
	}
	if source.SourceType != "source" || source.ID != "source" || source.URL != "https://source.example" || source.Title != "Source" || source.MediaType != "text/html" || source.Filename != "source.html" {
		t.Fatalf("source = %#v", source)
	}
	if len(toolResults) != 2 {
		t.Fatalf("tool results = %#v", toolResults)
	}
	search := toolResults[0]
	if search.Type != "web_search_tool_result" || search.ToolCallID != "search" || !search.ProviderExecuted || len(search.Result) != 1 {
		t.Fatalf("web search result = %#v", search)
	}
	webResult, ok := search.Result[0].(WebSearchResult)
	if !ok || webResult.Type != "web_search_result" || webResult.URL != "https://example.com" || webResult.EncryptedContent != "cipher" {
		t.Fatalf("web search payload = %#v", search.Result)
	}
	code := toolResults[1]
	if code.Type != "code_execution_tool_result" || code.ToolCallID != "code" || !code.ProviderExecuted || len(code.Result) != 1 {
		t.Fatalf("code execution result = %#v", code)
	}
	codeResult, ok := code.Result[0].(CodeExecutionResult)
	if !ok || codeResult.Type != "code_execution_result" || codeResult.Stdout != "ok" || codeResult.ReturnCode != 0 {
		t.Fatalf("code execution payload = %#v", code.Result)
	}
	wantOrder := []string{"anthropic.StreamStart", "anthropic.StreamResponseMetadata", "anthropic.StreamTextStart", "anthropic.StreamTextDelta", "anthropic.StreamTextEnd", "anthropic.StreamReasoningStart", "anthropic.StreamReasoningDelta", "anthropic.StreamReasoningDelta", "anthropic.StreamReasoningEnd", "anthropic.StreamToolInputStart", "anthropic.StreamToolInputDelta", "anthropic.StreamToolInputEnd", "anthropic.StreamToolCall", "anthropic.StreamSource", "anthropic.StreamToolResult", "anthropic.StreamToolResult", "anthropic.StreamFinish"}
	if strings.Join(order, ",") != strings.Join(wantOrder, ",") {
		t.Fatalf("part order = %#v\nwant = %#v", order, wantOrder)
	}
}

func TestPhase8InvalidProviderToolFailsBeforeRequestAndToolInputNormalizes(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	_, _, _, _, err := model.buildRequestWithOptions(GenerateOptions{MaxTokens: 1, Tools: []Tool{{ID: "anthropic.future_tool"}}}, false)
	var construction *RequestConstructionError
	if !errors.As(err, &construction) || construction.ToolIndex != 0 || construction.ToolID != "anthropic.future_tool" {
		t.Fatalf("error = %#v", err)
	}

	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{ToolCallContent{ToolCallID: "call", ToolName: "local", Input: nil}}}})
	got, err := json.Marshal(messages[0].Content[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"id":"call","input":{},"name":"local","type":"tool_use"}` {
		t.Fatalf("normalized tool call = %s", got)
	}
}

func TestPhase8UnknownProviderToolPublicCallsFailBeforeTransport(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "generate"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			calls := 0
			model := CreateAnthropic(ProviderSettings{Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
				calls++
				return nil, errors.New("unexpected transport call")
			}}}).Model("claude-3-haiku-20240307")
			var err error
			if stream {
				_, err = model.DoStream(context.Background(), StreamOptions{Tools: []Tool{{ID: "anthropic.unknown"}}})
			} else {
				_, err = model.DoGenerate(context.Background(), GenerateOptions{Tools: []Tool{{ID: "anthropic.unknown"}}})
			}
			var construction *RequestConstructionError
			if !errors.As(err, &construction) || construction.ToolIndex != 0 || construction.ToolID != "anthropic.unknown" {
				t.Fatalf("error = %#v", err)
			}
			if calls != 0 {
				t.Fatalf("transport calls = %d, want 0", calls)
			}
		})
	}
}

func intPtr(value int) *int { return &value }
