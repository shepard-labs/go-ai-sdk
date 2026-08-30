package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPhase2ResponseRequestReplay(t *testing.T) {
	body := []byte(`{"id":"msg","model":"claude","role":"assistant","stop_reason":"end_turn","usage":{},"content":[{"type":"server_tool_use","id":"srv","name":"web_search","input":{"query":"go"},"dynamic":true,"caller":{"type":"code_execution_20250825","tool_id":"code"},"future":"kept","provider_metadata":{"bad":true}},{"type":"web_search_tool_result","tool_use_id":"srv","caller":{"type":"direct"},"content":[{"type":"web_search_result","url":"https://example.com","title":"Example","encrypted_content":"cipher","page_age":"today"}],"future_result":7},{"type":"text","text":"answer","citations":[{"type":"web_search_result_location","encrypted_index":"index","cited_text":"quote"}]}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{result.Content[0].(ServerToolUseContent), result.Content[1].(ToolResultContent), result.Content[2].(TextContent)}}})
	got, err := json.Marshal(messages[0])
	if err != nil {
		t.Fatal(err)
	}
	wantParts := []string{`"type":"server_tool_use"`, `"dynamic":true`, `"caller":{"tool_id":"code","type":"code_execution_20250825"}`, `"future":"kept"`, `"type":"web_search_tool_result"`, `"future_result":7`, `"encrypted_content":"cipher"`, `"encrypted_index":"index"`}
	for _, part := range wantParts {
		if !strings.Contains(string(got), part) {
			t.Fatalf("replay %s missing %s", got, part)
		}
	}
	if strings.Contains(string(got), "provider_metadata") {
		t.Fatalf("replay contains provider_metadata: %s", got)
	}
}

func TestPhase2UnknownSpecializedResultPayloadSurvives(t *testing.T) {
	body := []byte(`{"id":"msg","model":"claude","role":"assistant","stop_reason":"end_turn","usage":{},"content":[{"type":"text_editor_code_execution_tool_result","tool_use_id":"edit","content":{"type":"text_editor_code_execution_view_result","content":"file","file_type":"text","num_lines":1,"future_nested":true}}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{result.Content[0].(ToolResultContent)}}})
	got, _ := json.Marshal(messages[0].Content[0])
	for _, part := range []string{`"type":"text_editor_code_execution_tool_result"`, `"type":"text_editor_code_execution_view_result"`, `"future_nested":true`} {
		if !strings.Contains(string(got), part) {
			t.Fatalf("result %s missing %s", got, part)
		}
	}
}

func TestPhase2ToolInputNormalization(t *testing.T) {
	for _, input := range []json.RawMessage{nil, {}, []byte(`bad`), []byte(`null`), []byte(`[]`), []byte(`1`), []byte(`"x"`)} {
		_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{ToolCallContent{ToolCallID: "id", ToolName: "tool", Input: input}}}})
		got, _ := json.Marshal(messages[0].Content[0].Input)
		if string(got) != `{}` {
			t.Fatalf("input %q = %s", input, got)
		}
	}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{ToolCallContent{ToolCallID: "id", ToolName: "tool", Input: json.RawMessage(`{"b":2,"a":1}`)}}}})
	got, _ := json.Marshal(messages[0].Content[0].Input)
	if string(got) != `{"a":1,"b":2}` {
		t.Fatalf("valid input = %s", got)
	}
}

// Fixture pinned from vercel/ai PR #16312 merge commit
// 19c5ee243b29734ff9e0b727bcd5ab6d689285cf,
// packages/anthropic/src/convert-to-anthropic-prompt.test.ts:1771-1894:
// [text, client tool_use, server_tool_use, provider result] ->
// [text, server_tool_use, provider result, client tool_use].
func TestPhase2MixedToolOrderingIsStableAndLocal(t *testing.T) {
	contents := []AssistantContent{
		TextContent{Text: "A"},
		ToolCallContent{ToolCallID: "c1", ToolName: "client1", Input: json.RawMessage(`{}`)},
		ServerToolUseContent{ID: "p1", Name: "web_search", Input: json.RawMessage(`{}`)},
		ToolResultContent{Type: "web_search_tool_result", ToolCallID: "p1", ProviderExecuted: true},
		TextContent{Text: "B"},
		ToolCallContent{ToolCallID: "c2", ToolName: "client2", Input: json.RawMessage(`{}`)},
		ServerToolUseContent{ID: "p2", Name: "web_fetch", Input: json.RawMessage(`{}`)},
	}
	blocks := convertAssistantContent(contents)
	want := []string{"text", "server_tool_use", "web_search_tool_result", "tool_use", "text", "server_tool_use", "tool_use"}
	for i := range want {
		if blocks[i].Type != want[i] {
			t.Fatalf("order[%d] = %s, all=%#v", i, blocks[i].Type, blocks)
		}
	}
}

func TestPhase2TypedFieldsWinOverRaw(t *testing.T) {
	call := ToolCallContent{ToolCallID: "typed-id", ToolName: "typed-name", Input: json.RawMessage(`{"typed":true}`), Raw: map[string]json.RawMessage{
		"type": json.RawMessage(`"server_tool_use"`), "id": json.RawMessage(`"raw-id"`), "name": json.RawMessage(`"raw-name"`), "input": json.RawMessage(`{"raw":true}`), "provider_metadata": json.RawMessage(`{"bad":true}`), "future": json.RawMessage(`42`),
	}}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{call}}})
	got, _ := json.Marshal(messages[0].Content[0])
	if string(got) != `{"future":42,"id":"typed-id","input":{"typed":true},"name":"typed-name","type":"tool_use"}` {
		t.Fatalf("block = %s", got)
	}
}

func TestPhase2TypedNestedResultWinsAndClearsRaw(t *testing.T) {
	result := ToolResultContent{Type: "web_search_tool_result", ToolCallID: "id", Result: []ToolResultPart{WebSearchResult{Type: "web_search_result", URL: "typed", Title: ""}}, RawContent: json.RawMessage(`[{"type":"web_search_result","url":"raw","title":"raw title","encrypted_content":"raw cipher","page_age":"old","future":{"kept":true}}]`)}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{result}}})
	got, _ := json.Marshal(messages[0].Content[0])
	want := `{"content":[{"future":{"kept":true},"type":"web_search_result","url":"typed"}],"tool_use_id":"id","type":"web_search_tool_result"}`
	if string(got) != want {
		t.Fatalf("block = %s, want %s", got, want)
	}
}

func TestPhase2RawSupportedKeysCannotReappearAfterClearing(t *testing.T) {
	block := apiContentBlock{Type: "web_search_result", Raw: map[string]json.RawMessage{"encrypted_content": json.RawMessage(`"cipher"`), "page_age": json.RawMessage(`"old"`), "source": json.RawMessage(`{"future":true}`), "future": json.RawMessage(`1`)}}
	got, _ := json.Marshal(block)
	if string(got) != `{"future":1,"source":{"future":true},"type":"web_search_result"}` {
		t.Fatalf("block = %s", got)
	}

	document := apiContentBlock{Type: "document", Raw: map[string]json.RawMessage{"source": json.RawMessage(`{"type":"url","url":"raw"}`), "citations": json.RawMessage(`[1]`)}}
	got, _ = json.Marshal(document)
	if string(got) != `{"type":"document"}` {
		t.Fatalf("document = %s", got)
	}
}

func TestPhase2OrdinaryToolResultRemainsClientOwnedDuringOrdering(t *testing.T) {
	blocks := convertAssistantContent([]AssistantContent{
		ToolCallContent{ToolCallID: "call", ToolName: "client", Input: json.RawMessage(`{}`)},
		ToolResultContent{ToolCallID: "call", Result: []ToolResultPart{ToolResultText{Text: "client result"}}},
		ServerToolUseContent{ID: "server", Name: "web_search", Input: json.RawMessage(`{}`)},
		ToolResultContent{Type: "web_search_tool_result", ToolCallID: "server"},
	})
	want := []string{"server_tool_use", "web_search_tool_result", "tool_use", "tool_result"}
	for i := range want {
		if blocks[i].Type != want[i] {
			t.Fatalf("order[%d] = %s, all=%#v", i, blocks[i].Type, blocks)
		}
	}
}

func TestPhase2UserProviderResultPreservesNativeShape(t *testing.T) {
	content := ToolResultContent{Type: "mcp_tool_result", ToolCallID: "mcp", IsError: true, Dynamic: true, Caller: &ToolCallCallerInfo{Type: "direct"}, Result: []ToolResultPart{ToolResultText{Text: "typed"}}, Raw: map[string]json.RawMessage{"future": json.RawMessage(`7`)}}
	_, messages := convertPrompt([]Message{UserMessage{Content: []UserContent{content}}})
	got, _ := json.Marshal(messages[0].Content[0])
	for _, part := range []string{`"type":"mcp_tool_result"`, `"tool_use_id":"mcp"`, `"is_error":true`, `"dynamic":true`, `"caller":{"type":"direct"}`, `"future":7`, `"text":"typed"`} {
		if !strings.Contains(string(got), part) {
			t.Fatalf("user result %s missing %s", got, part)
		}
	}
}

func TestPhase2WebFetchResultPreservesNestedDocumentSource(t *testing.T) {
	body := []byte(`{"id":"msg","model":"claude","role":"assistant","stop_reason":"end_turn","usage":{},"content":[{"type":"web_fetch_tool_result","tool_use_id":"fetch","content":[{"type":"web_fetch_result","url":"https://page.example","retrieved_at":"today","content":{"type":"document","source":{"type":"url","media_type":"text/html","data":"encoded","url":"https://source.example/doc"}}}]}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	fetch := result.Content[0].(ToolResultContent).Result[0].(WebFetchResult)
	if got, want := fetch.Content.Source, (DocumentSource{Type: "url", MediaType: "text/html", Data: "encoded", URL: "https://source.example/doc"}); got != want {
		t.Fatalf("nested source = %#v, want %#v", got, want)
	}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{result.Content[0].(ToolResultContent)}}})
	got, _ := json.Marshal(messages[0].Content[0])
	for _, part := range []string{`"media_type":"text/html"`, `"data":"encoded"`, `"url":"https://source.example/doc"`} {
		if !strings.Contains(string(got), part) {
			t.Fatalf("replay %s missing %s", got, part)
		}
	}
}

func TestPhase2UnknownResultArrayEntryRetainsOrderAndTypedOverrides(t *testing.T) {
	body := []byte(`{"id":"msg","model":"claude","role":"assistant","stop_reason":"end_turn","usage":{},"content":[{"type":"web_search_tool_result","tool_use_id":"search","content":[{"type":"web_search_result","url":"first-raw","title":"First"},{"type":"future_result","value":7},{"type":"web_search_result","url":"last-raw","title":"Last"}]}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	toolResult := result.Content[0].(ToolResultContent)
	if len(toolResult.Result) != 3 {
		t.Fatalf("parts = %#v", toolResult.Result)
	}
	if unknown, ok := toolResult.Result[1].(ToolResultUnknown); !ok || unknown.Type != "future_result" {
		t.Fatalf("middle part = %#v", toolResult.Result[1])
	}
	toolResult.Result[0] = WebSearchResult{Type: "web_search_result", URL: "first-typed", Title: "First"}
	toolResult.Result[2] = WebSearchResult{Type: "web_search_result", URL: "last-typed", Title: "Last"}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{toolResult}}})
	got, _ := json.Marshal(messages[0].Content[0])
	want := `{"content":[{"title":"First","type":"web_search_result","url":"first-typed"},{"type":"future_result","value":7},{"title":"Last","type":"web_search_result","url":"last-typed"}],"tool_use_id":"search","type":"web_search_tool_result"}`
	if string(got) != want {
		t.Fatalf("block = %s, want %s", got, want)
	}
}

func TestPhase2CallerPreservesUnknownFieldsAndTypedFieldsWin(t *testing.T) {
	body := []byte(`{"id":"msg","model":"claude","role":"assistant","stop_reason":"end_turn","usage":{},"content":[{"type":"server_tool_use","id":"srv","name":"tool","input":{},"caller":{"type":"raw-type","tool_id":"raw-tool","future":{"kept":true}}}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	call := result.Content[0].(ServerToolUseContent)
	call.Caller.Type = "typed-type"
	call.Caller.ToolID = "typed-tool"
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{call}}})
	got, _ := json.Marshal(messages[0].Content[0])
	want := `{"caller":{"future":{"kept":true},"tool_id":"typed-tool","type":"typed-type"},"id":"srv","input":{},"name":"tool","type":"server_tool_use"}`
	if string(got) != want {
		t.Fatalf("block = %s, want %s", got, want)
	}
}

func TestPhase2AssistantTextPreservesUnknownFieldsAndTypedFieldsWin(t *testing.T) {
	body := []byte(`{"id":"msg","model":"claude","role":"assistant","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"raw text","citations":[{"type":"raw"}],"future":{"kept":true}}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	text := result.Content[0].(TextContent)
	text.Text = "typed text"
	text.Citations = []Citation{{Type: "typed"}}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{text}}})
	got, _ := json.Marshal(messages[0].Content[0])
	want := `{"citations":[{"type":"typed"}],"future":{"kept":true},"text":"typed text","type":"text"}`
	if string(got) != want {
		t.Fatalf("block = %s, want %s", got, want)
	}
}

func TestPhase2MCPAndProviderBlocksRoundTripFromResponse(t *testing.T) {
	body := []byte(`{"id":"msg","model":"claude","role":"assistant","stop_reason":"end_turn","usage":{},"content":[{"type":"tool_use","id":"client","name":"client_tool","input":{"value":1},"dynamic":true,"caller":{"type":"direct","future":"caller"}},{"type":"mcp_tool_use","id":"mcp","name":"lookup","server_name":"tools","input":{"query":"go"},"dynamic":true,"future_call":true},{"type":"mcp_tool_result","tool_use_id":"mcp","content":[{"type":"text","text":"found"}],"dynamic":true,"caller":{"type":"direct"},"future_result":2}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{
		result.Content[0].(ToolCallContent),
		result.Content[1].(MCPToolUseContent),
		result.Content[2].(ToolResultContent),
	}}})
	got, err := json.Marshal(messages[0])
	if err != nil {
		t.Fatal(err)
	}
	want := `{"role":"assistant","content":[{"dynamic":true,"future_call":true,"id":"mcp","input":{"query":"go"},"name":"lookup","server_name":"tools","type":"mcp_tool_use"},{"caller":{"type":"direct"},"content":[{"text":"found","type":"text"}],"dynamic":true,"future_result":2,"tool_use_id":"mcp","type":"mcp_tool_result"},{"caller":{"future":"caller","type":"direct"},"dynamic":true,"id":"client","input":{"value":1},"name":"client_tool","type":"tool_use"}]}`
	if string(got) != want {
		t.Fatalf("replay = %s, want %s", got, want)
	}
}
