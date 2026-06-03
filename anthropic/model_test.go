package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type roundTripFetcher struct {
	fn func(*http.Request) (*http.Response, error)
}

func (f roundTripFetcher) Do(req *http.Request) (*http.Response, error) { return f.fn(req) }

func TestPromptConvertTextOnly(t *testing.T) {
	system, messages := convertPrompt([]Message{
		SystemMessage{Content: "system"},
		UserMessage{Content: []UserContent{TextContent{Text: "hello"}}},
	})
	if len(system) != 1 || system[0].Text != "system" {
		t.Fatalf("system = %#v", system)
	}
	if len(messages) != 1 || messages[0].Role != "user" || messages[0].Content[0].Text != "hello" {
		t.Fatalf("messages = %#v", messages)
	}
}

func TestPromptConvertWithImages(t *testing.T) {
	_, messages := convertPrompt([]Message{UserMessage{Content: []UserContent{ImageContent{Source: ImageSource{MediaType: "image/png", Data: "abc"}}}}})
	block := messages[0].Content[0]
	if block.Type != "image" || block.Source.Type != "base64" || block.Source.MediaType != "image/png" || block.Source.Data != "abc" {
		t.Fatalf("image block = %#v", block)
	}
}

func TestPromptConvertWithDocuments(t *testing.T) {
	_, messages := convertPrompt([]Message{UserMessage{Content: []UserContent{DocumentContent{Source: DocumentSource{MediaType: "application/pdf", Data: "abc"}}}}})
	block := messages[0].Content[0]
	if block.Type != "document" || block.Source.Type != "base64" || block.Source.MediaType != "application/pdf" || block.Source.Data != "abc" {
		t.Fatalf("document block = %#v", block)
	}
}

func TestPromptConvertWithAssistantContent(t *testing.T) {
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{
		TextContent{Text: "answer"},
		ThinkingContent{Thinking: "think", Signature: "sig"},
		RedactedThinkingContent{Data: "redacted"},
		CompactionContent{Text: "compact"},
	}}})
	blocks := messages[0].Content
	if blocks[0].Type != "text" || blocks[0].Text != "answer" {
		t.Fatalf("text block = %#v", blocks[0])
	}
	if blocks[1].Type != "thinking" || blocks[1].Thinking != "think" || blocks[1].Signature != "sig" {
		t.Fatalf("thinking block = %#v", blocks[1])
	}
	if blocks[2].Type != "redacted_thinking" || blocks[2].Data != "redacted" {
		t.Fatalf("redacted block = %#v", blocks[2])
	}
	if blocks[3].Type != "compaction" || blocks[3].Text != "compact" {
		t.Fatalf("compaction block = %#v", blocks[3])
	}
}

func TestBuildRequest(t *testing.T) {
	temp := 0.5
	topK := 3
	topP := 0.8
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	req, warnings := model.buildRequest(GenerateOptions{
		Messages:      []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}},
		MaxTokens:     100,
		Temperature:   &temp,
		TopK:          &topK,
		TopP:          &topP,
		StopSequences: []string{"stop"},
	}, false)
	if req.Model != "claude-3-haiku-20240307" || req.MaxTokens != 100 || req.Stream {
		t.Fatalf("request = %#v", req)
	}
	if *req.Temperature != 0.5 || *req.TopK != 3 || *req.TopP != 0.8 || req.StopSequences[0] != "stop" {
		t.Fatalf("request params = %#v", req)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestBuildRequestWithModelOptions(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-opus-4-6", ModelOptions{
		Metadata:     &Metadata{UserID: "user"},
		Speed:        "fast",
		InferenceGeo: "us",
		Container:    &Container{ID: "container"},
		MCPServers:   []MCPServer{{Name: "mcp"}},
	}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{}, false)
	if req.Metadata.UserID != "user" || req.Speed != "fast" || req.InferenceGeo != "us" || req.Container.ID != "container" || req.MCPServers[0].Name != "mcp" {
		t.Fatalf("request = %#v", req)
	}
}

func TestBuildRequestWithToolsAndToolChoice(t *testing.T) {
	deferLoading := true
	eager := true
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{
		Tools:       []Tool{{Name: "weather", Description: "Get weather", InputSchema: map[string]any{"type": "object"}}},
		ToolOptions: ToolOptions{DeferLoading: &deferLoading, EagerInputStreaming: &eager, AllowedCallers: []ToolCallCaller{"code"}},
		ToolChoice:  &ToolChoice{Type: "required", Name: "weather", DisableParallelToolUse: true},
	}, false)
	if len(req.Tools) != 1 || req.Tools[0].Name != "weather" || req.Tools[0].DeferLoading == nil || req.Tools[0].AllowedCallers[0] != "code" {
		t.Fatalf("tools = %#v", req.Tools)
	}
	if req.ToolChoice.Type != "required" || req.ToolChoice.Name != "weather" || !req.DisableParallelToolUse {
		t.Fatalf("request = %#v", req)
	}
}

func TestBuildRequestWithProviderToolsAndModelOptions(t *testing.T) {
	budget := 100
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{SendReasoning: true, DisableParallelToolUse: true, Effort: "xhigh", TaskBudget: &budget}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{Tools: []Tool{{ID: "anthropic.code_execution_20250522", Name: "code_execution_20250522", Type: "anthropic.code_execution_20250522", ProviderExecuted: true}}}, false)
	if len(req.Tools) != 1 || req.Tools[0].ID != "anthropic.code_execution_20250522" || !req.SendReasoning || !req.DisableParallelToolUse || req.Effort != "xhigh" || *req.TaskBudget != 100 {
		t.Fatalf("request = %#v", req)
	}
}

func TestBetaHeadersProviderTools(t *testing.T) {
	p := CreateAnthropic(ProviderSettings{}).(*anthropicProvider)
	headers := p.headersForOptions(ModelOptions{RequestTools: []Tool{{ID: "anthropic.code_execution_20250522"}, {ID: "anthropic.web_fetch_20260209"}}})
	values := headers.Values("anthropic-beta")
	if len(values) != 2 || values[0] != "code-execution-2025-05-22" || values[1] != "code-execution-web-tools-2026-02-09" {
		t.Fatalf("headers = %#v", values)
	}
}

func TestPromptConvertToolResultsAndAssistantToolCalls(t *testing.T) {
	_, userMessages := convertPrompt([]Message{UserMessage{Content: []UserContent{ToolResultContent{ToolCallID: "call", IsError: true, Result: []ToolResultPart{ToolResultText{Text: "ok"}, ToolResultReference{ID: "ref"}}}}}})
	result := userMessages[0].Content[0]
	if result.Type != "tool_result" || result.ToolUseID != "call" || !result.IsError {
		t.Fatalf("tool result = %#v", result)
	}

	_, assistantMessages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{
		ToolCallContent{ToolCallID: "call", ToolName: "weather", Input: json.RawMessage(`{"city":"SF"}`)},
		ServerToolUseContent{ID: "server", Name: "web_search", Input: json.RawMessage(`{"query":"x"}`)},
		MCPToolUseContent{ID: "mcp", Name: "tool", ServerName: "server", Input: json.RawMessage(`{"a":1}`)},
	}}})
	blocks := assistantMessages[0].Content
	if blocks[0].Type != "tool_use" || blocks[0].Name != "weather" || blocks[1].Type != "server_tool_use" || blocks[2].Type != "mcp_tool_use" || blocks[2].ServerName != "server" {
		t.Fatalf("blocks = %#v", blocks)
	}
}

func TestTemperatureClamping(t *testing.T) {
	high := 2.0
	low := -1.0
	got, warnings := normalizeTemperature(&high)
	if *got != 1.0 || len(warnings) != 1 {
		t.Fatalf("high = %v %#v", *got, warnings)
	}
	got, warnings = normalizeTemperature(&low)
	if *got != 0 || len(warnings) != 1 {
		t.Fatalf("low = %v %#v", *got, warnings)
	}
}

func TestUnsupportedParameterWarnings(t *testing.T) {
	value := 1.0
	seed := 1
	warnings := unsupportedWarnings(GenerateOptions{FrequencyPenalty: &value, PresencePenalty: &value, Seed: &seed, TopP: &value, Temperature: &value})
	if len(warnings) != 4 {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestUsageConversionWithCacheAndIterations(t *testing.T) {
	usage := UsageFromResponseUsage(ResponseUsage{InputTokens: 10, OutputTokens: 5, CacheCreationInputTokens: 3, CacheReadInputTokens: 2, Iterations: []UsageIteration{{InputTokens: 1, OutputTokens: 1}}})
	if usage.InputTokens.Total != 10 || usage.InputTokens.CacheWrite != 3 || usage.InputTokens.CacheRead != 2 || usage.OutputTokens.Total != 5 || usage.TotalTokens != 15 || len(usage.Iterations) != 1 {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestDoGenerateTextOnly(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		if req.URL.Path != "/messages" {
			t.Fatalf("path = %q", req.URL.Path)
		}
		if req.Header.Get("x-api-key") != "key" || req.Header.Get("anthropic-version") == "" {
			t.Fatalf("headers = %#v", req.Header)
		}
		body, _ := io.ReadAll(req.Body)
		var decoded apiMessagesRequest
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Messages[0].Content[0].Text != "hello" {
			t.Fatalf("body = %s", body)
		}
		return jsonResponse(200, `{"id":"msg_1","model":"claude-3-haiku-20240307","role":"assistant","content":[{"type":"text","text":"answer"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content[0].(TextContent).Text != "answer" || result.FinishReason != FinishReasonStop || result.Usage.InputTokens.Total != 2 || result.Usage.OutputTokens.Total != 3 {
		t.Fatalf("result = %#v", result)
	}
}

func TestDoGenerateAPIError(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		resp := jsonResponse(400, `{"error":{"type":"invalid_request_error","message":"bad"}}`)
		resp.Header.Set("x-request-id", "req_123")
		return resp, nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || apiErr.Status != 400 || apiErr.Message != "bad" || apiErr.Type != "invalid_request_error" || apiErr.RequestID != "req_123" || apiErr.Retryable {
		t.Fatalf("err = %#v", err)
	}
}

func TestDoGenerateAPIErrorBodyLimit(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		return jsonResponse(500, strings.Repeat("x", 64)), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 0}, MaxErrorResponseBytes: 8}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || !apiErr.Truncated || len(apiErr.Body) != 8 || !apiErr.Retryable {
		t.Fatalf("err = %#v", err)
	}
}

func TestDoGenerateSuccessBodyLimit(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		return jsonResponse(200, `{"content":[{"type":"text","text":"too large"}]}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, MaxResponseBodyBytes: 8}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || !apiErr.Truncated || apiErr.Status != 200 || apiErr.Retryable {
		t.Fatalf("err = %#v", err)
	}
}

func TestDoStreamTextOnly(t *testing.T) {
	sse := "event: message_start\ndata: {\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-haiku-20240307\"}}\n\n" +
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	var parts []StreamPart
	for part := range result.Stream {
		parts = append(parts, part)
	}
	if len(parts) != 5 || parts[2].(StreamTextDelta).Text != "hi" || result.Response.ID != "msg_1" {
		t.Fatalf("parts = %#v response = %#v", parts, result.Response)
	}
}

func TestModelConcurrentGenerateReuse(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(req.Body)
		var decoded apiMessagesRequest
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Messages[0].Content[0].Text == "" {
			t.Fatalf("body = %s", body)
		}
		return jsonResponse(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := model.DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, Tools: []Tool{{Name: "tool", InputSchema: map[string]any{"type": "object"}}}, MaxTokens: 10})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}

func TestDoStreamLargeDataLine(t *testing.T) {
	large := strings.Repeat("a", 128*1024)
	sse := "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"" + large + "\"}}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	part := <-result.Stream
	delta, ok := part.(StreamTextDelta)
	if !ok || len(delta.Text) != len(large) {
		t.Fatalf("part = %#v", part)
	}
	for part := range result.Stream {
		if err, ok := part.(StreamError); ok {
			t.Fatalf("stream error: %v", err.Err)
		}
	}
}

func TestDoGenerateRetriesRetryableStatus(t *testing.T) {
	var calls int32
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		if atomic.AddInt32(&calls, 1) == 1 {
			return jsonResponse(429, `{"error":{"type":"rate_limit_error","message":"limited"}}`), nil
		}
		return jsonResponse(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 1, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}}).Model("claude-3-haiku-20240307")
	result, err := model.DoGenerate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content[0].(TextContent).Text != "ok" || atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("result = %#v calls = %d", result, calls)
	}
}

func TestDoGenerateRetriesCanBeDisabled(t *testing.T) {
	var calls int32
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		atomic.AddInt32(&calls, 1)
		return jsonResponse(500, `{"error":{"type":"server_error","message":"bad"}}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 0}}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	if err == nil || atomic.LoadInt32(&calls) != 1 {
		t.Fatalf("err = %v calls = %d", err, calls)
	}
}

func TestDoGenerateNilResponseFetcher(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return nil, nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 0}}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || apiErr.Message == "" {
		t.Fatalf("err = %#v", err)
	}
}

func TestResponseContentParsing(t *testing.T) {
	body := []byte(`{"content":[{"type":"thinking","thinking":"think","signature":"sig"},{"type":"redacted_thinking","data":"hidden"},{"type":"compaction","text":"compact"},{"type":"source","id":"src","url":"https://example.com","title":"title","media_type":"text/html","filename":"a.html"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":2}}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != FinishReasonLength || result.Content[0].(ReasoningContent).Type != "reasoning" || result.Content[1].(ReasoningContent).Type != "redacted_reasoning" || result.Content[2].(CompactionContent).Text != "compact" || result.Content[3].(SourceContent).Filename != "a.html" {
		t.Fatalf("result = %#v", result)
	}
}

func TestResponseParseToolCallsAndResults(t *testing.T) {
	body := []byte(`{"content":[{"type":"tool_use","id":"call","name":"weather","input":{"city":"SF"},"dynamic":true,"provider_metadata":{"a":1}},{"type":"server_tool_use","id":"server","name":"web_search","input":{"query":"x"}},{"type":"mcp_tool_result","tool_use_id":"mcp","content":[{"type":"text","text":"done"}],"is_error":true,"provider_executed":true}],"stop_reason":"tool_use"}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	call := result.Content[0].(ToolCallContent)
	server := result.Content[1].(ToolCallContent)
	toolResult := result.Content[2].(ToolResultContent)
	if result.FinishReason != FinishReasonToolCalls || call.Type != "tool-call" || !call.Dynamic || call.ProviderMetadata["a"].(float64) != 1 || server.Type != "server-tool-call" || !server.ProviderExecuted || !toolResult.IsError || !toolResult.ProviderExecuted || toolResult.Result[0].(ToolResultText).Text != "done" {
		t.Fatalf("result = %#v", result)
	}
}

func TestStreamToolEvents(t *testing.T) {
	sse := "event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"input_json\",\"name\":\"weather\",\"dynamic\":true}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\"\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":0}\n\n" +
		"event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call\",\"name\":\"weather\",\"input\":{\"city\":\"SF\"},\"provider_metadata\":{\"x\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"index\":2,\"content_block\":{\"type\":\"tool_result\",\"tool_use_id\":\"call\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}],\"provider_executed\":true}}\n\n" +
		"event: content_block_start\ndata: {\"index\":3,\"content_block\":{\"type\":\"source\",\"id\":\"src\",\"url\":\"https://example.com\",\"title\":\"Example\"}}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var parts []StreamPart
	for part := range result.Stream {
		parts = append(parts, part)
	}
	if parts[0].(StreamToolInputStart).ToolName != "weather" || !parts[0].(StreamToolInputStart).Dynamic || parts[1].(StreamToolInputDelta).Delta.(InputJSONDelta).PartialJSON == "" || parts[2].(StreamToolInputEnd).ID != "0" || parts[3].(StreamToolCall).ToolName != "weather" || parts[4].(StreamToolResult).Result[0].(ToolResultText).Text != "ok" || parts[5].(StreamSource).URL != "https://example.com" {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestSupportURLs(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307")
	urls := model.SupportURLs()
	if len(urls["anthropic"]) == 0 || !urls["anthropic"][0].MatchString("https://docs.anthropic.com/en/docs") {
		t.Fatalf("urls = %#v", urls)
	}
	if model.Provider() != "anthropic" {
		t.Fatalf("provider = %q", model.Provider())
	}
}

func jsonResponse(status int, body string) *http.Response {
	resp := textResponse(status, body)
	resp.Header.Set("Content-Type", "application/json")
	return resp
}

func textResponse(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}
