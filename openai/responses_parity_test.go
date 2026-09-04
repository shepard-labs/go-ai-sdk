package openai

import (
	"context"
	"net/http"
	"reflect"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestResponsesReasoningSummaryDefaultsDetailed verifies reasoning.summary
// defaults to "detailed" when reasoning is enabled without an explicit
// summary, and that explicit summaries (typed or via provider options)
// are preserved.
func TestResponsesReasoningSummaryDefaultsDetailed(t *testing.T) {
	effort := "medium"
	explicit := "auto"
	cases := []struct {
		name            string
		reasoning       *ReasoningConfig
		providerOptions ProviderOptions
		wantKey         bool
		wantSummary     string
	}{
		{"effort-only-defaults-detailed", &ReasoningConfig{Effort: &effort}, nil, true, "detailed"},
		{"explicit-summary-preserved", &ReasoningConfig{Effort: &effort, Summary: &explicit}, nil, true, "auto"},
		{"summary-only-enables-reasoning", &ReasoningConfig{Summary: &explicit}, nil, true, "auto"},
		{"provider-option-summary-preserved", &ReasoningConfig{Effort: &effort}, ProviderOptions{"openai": map[string]any{"reasoningSummary": "auto"}}, true, "auto"},
		{"no-reasoning-no-key", nil, nil, false, ""},
		{"empty-reasoning-no-key", &ReasoningConfig{}, nil, false, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := &openaiResponsesModel{modelID: "gpt-4o"}
			body, _, err := m.buildResponsesRequest(ResponsesGenerateOptions{
				Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
				Reasoning:       c.reasoning,
				ProviderOptions: c.providerOptions,
			})
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			r, has := body["reasoning"].(map[string]any)
			if has != c.wantKey {
				t.Fatalf("reasoning key present=%v, want %v (body=%v)", has, c.wantKey, body["reasoning"])
			}
			if !c.wantKey {
				return
			}
			if r["summary"] != c.wantSummary {
				t.Errorf("summary=%v, want %q", r["summary"], c.wantSummary)
			}
		})
	}
}

// TestResponsesEmptyOutputReturnsError verifies an absent, empty, or
// non-array output is rejected with InvalidResponseDataError, while a
// populated output parses cleanly.
func TestResponsesEmptyOutputReturnsError(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr bool
	}{
		{"absent-output", `{"id":"r","status":"completed"}`, true},
		{"empty-output", `{"id":"r","status":"completed","output":[]}`, true},
		{"null-output", `{"id":"r","status":"completed","output":null}`, true},
		{"non-array-output", `{"id":"r","status":"completed","output":"oops"}`, true},
		{"message-output", `{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}`, false},
		{"function-call-output", `{"id":"r","status":"completed","output":[{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}]}`, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := &openaiResponsesModel{modelID: "gpt-4o"}
			_, err := m.parseResponsesResponse([]byte(c.body), nil)
			if c.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				if _, ok := err.(InvalidResponseDataError); !ok {
					t.Errorf("expected InvalidResponseDataError, got %T: %v", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
		})
	}
}

// TestResponsesToolSearchIdentityRoundTrip covers tool_search identity end
// to end: parse preserves call_id and server execution, the request path
// emits native tool_search_output only for declared provider tool
// searches, and an ordinary function named "tool_search" replays as
// function_call.
func TestResponsesToolSearchIdentityRoundTrip(t *testing.T) {
	t.Run("parse", func(t *testing.T) {
		cases := []struct {
			name           string
			body           string
			wantCallID     string
			wantProviderEx bool
		}{
			{"server-executed", `{"id":"r","status":"completed","output":[{"type":"tool_search_call","call_id":"ts-1","execution":"server","arguments":"{}"}]}`, "ts-1", true},
			{"client-executed", `{"id":"r","status":"completed","output":[{"type":"tool_search_call","call_id":"ts-2","execution":"client","arguments":"{}"}]}`, "ts-2", false},
			{"missing-execution-is-client", `{"id":"r","status":"completed","output":[{"type":"tool_search_call","call_id":"ts-3","arguments":"{}"}]}`, "ts-3", false},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				m := &openaiResponsesModel{modelID: "gpt-4o"}
				res, err := m.parseResponsesResponse([]byte(c.body), nil)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				tc, ok := res.Content[0].(ToolCallContent)
				if !ok {
					t.Fatalf("content type=%T", res.Content[0])
				}
				if tc.ToolCallID != c.wantCallID {
					t.Errorf("ToolCallID=%q, want %q", tc.ToolCallID, c.wantCallID)
				}
				if tc.ToolName != "tool_search" {
					t.Errorf("ToolName=%q, want tool_search", tc.ToolName)
				}
				if tc.ProviderExecuted != c.wantProviderEx {
					t.Errorf("ProviderExecuted=%v, want %v", tc.ProviderExecuted, c.wantProviderEx)
				}
			})
		}
	})

	t.Run("request", func(t *testing.T) {
		searchResult := ToolResultContent{
			ToolResultContent: openaicompatible.ToolResultContent{
				ToolCallID: "ts-9",
				Output:     openaicompatible.ToolResultOutput{Type: "json", Value: []any{map[string]any{"name": "f"}}},
			},
			ToolName: "tool_search",
		}
		cases := []struct {
			name          string
			tools         []Tool
			wantType      string
			wantExecution string
		}{
			{
				"declared-provider-search",
				[]Tool{{Type: "provider", ID: "openai.toolSearch", Args: ToolSearchArgs{}}},
				"tool_search_output", "client",
			},
			{"ordinary-function-named-tool-search", nil, "function_call_output", ""},
			{
				"function-tool-named-tool-search",
				[]Tool{{Type: "function", Name: "tool_search", InputSchema: map[string]any{"type": "object"}}},
				"function_call_output", "",
			},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				m := &openaiResponsesModel{modelID: "gpt-4o"}
				body, _, err := m.buildResponsesRequest(ResponsesGenerateOptions{
					Messages: []Message{
						UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
						ToolMessage{Content: []ToolContent{searchResult}},
					},
					Tools: c.tools,
				})
				if err != nil {
					t.Fatalf("build: %v", err)
				}
				input, _ := body["input"].([]any)
				if len(input) != 2 {
					t.Fatalf("input len=%d: %v", len(input), input)
				}
				item, _ := input[1].(map[string]any)
				if item["type"] != c.wantType {
					t.Errorf("type=%v, want %v (item=%v)", item["type"], c.wantType, item)
				}
				if c.wantExecution != "" && item["execution"] != c.wantExecution {
					t.Errorf("execution=%v, want %v", item["execution"], c.wantExecution)
				}
			})
		}
	})
}

// TestResponsesStructuredOutputTextFormat verifies StructuredOutput (and
// the ResponseFormat json path) serialize as
// text.format={type:json_schema, strict:true, name, schema, description?}.
func TestResponsesStructuredOutputTextFormat(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}
	cases := []struct {
		name            string
		structured      *StructuredOutput
		responseFormat  *ResponseFormat
		wantType        string
		wantName        string
		wantStrict      bool
		wantDescription bool
	}{
		{"full", &StructuredOutput{Name: "City", Description: "A city", Schema: schema}, nil, "json_schema", "City", true, true},
		{"default-name", &StructuredOutput{Schema: schema}, nil, "json_schema", "response", true, false},
		{"no-description", &StructuredOutput{Name: "N", Schema: schema}, nil, "json_schema", "N", true, false},
		{"response-format-json", nil, &ResponseFormat{Type: "json", Name: "RF", Description: "d", Schema: schema}, "json_schema", "RF", true, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := &openaiResponsesModel{modelID: "gpt-4o"}
			body, _, err := m.buildResponsesRequest(ResponsesGenerateOptions{
				Messages:         []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
				StructuredOutput: c.structured,
				ResponseFormat:   c.responseFormat,
			})
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			text, ok := body["text"].(map[string]any)
			if !ok {
				t.Fatalf("text missing: %v", body["text"])
			}
			format, ok := text["format"].(map[string]any)
			if !ok {
				t.Fatalf("format missing: %v", text["format"])
			}
			if format["type"] != c.wantType {
				t.Errorf("type=%v, want %v", format["type"], c.wantType)
			}
			if format["strict"] != c.wantStrict {
				t.Errorf("strict=%v, want %v", format["strict"], c.wantStrict)
			}
			if format["name"] != c.wantName {
				t.Errorf("name=%v, want %v", format["name"], c.wantName)
			}
			if !reflect.DeepEqual(format["schema"], schema) {
				t.Errorf("schema=%v", format["schema"])
			}
			_, hasDesc := format["description"]
			if hasDesc != c.wantDescription {
				t.Errorf("description present=%v, want %v", hasDesc, c.wantDescription)
			}
		})
	}
}

// TestResponsesAllowedToolsNormalization verifies allowedTools accepts
// []string and {tool_names: [...]} in struct/map/slice forms and
// serializes wire allowed_tools.
func TestResponsesAllowedToolsNormalization(t *testing.T) {
	mcpURL := "https://mcp.example.com"
	cases := []struct {
		name       string
		args       any
		want       any
		wantAbsent bool
	}{
		{"string-slice", MCPArgs{ServerLabel: "s", ServerURL: &mcpURL, AllowedTools: []string{"a", "b"}}, []string{"a", "b"}, false},
		{"any-slice", map[string]any{"serverLabel": "s", "serverUrl": mcpURL, "allowedTools": []any{"a", "b"}}, []string{"a", "b"}, false},
		{"map-tool-names", map[string]any{"serverLabel": "s", "serverUrl": mcpURL, "allowedTools": map[string]any{"tool_names": []string{"a"}}}, map[string]any{"tool_names": []string{"a"}}, false},
		{"map-toolNames", map[string]any{"serverLabel": "s", "serverUrl": mcpURL, "allowedTools": map[string]any{"toolNames": []any{"a", "b"}}}, map[string]any{"tool_names": []string{"a", "b"}}, false},
		{
			"struct-form",
			MCPArgs{ServerLabel: "s", ServerURL: &mcpURL, AllowedTools: struct {
				ToolNames []string `json:"toolNames"`
			}{ToolNames: []string{"a"}}},
			map[string]any{"tool_names": []string{"a"}}, false,
		},
		{"nil-omitted", MCPArgs{ServerLabel: "s", ServerURL: &mcpURL}, nil, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := &openaiResponsesModel{modelID: "gpt-4o"}
			body, _, err := m.buildResponsesRequest(ResponsesGenerateOptions{
				Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
				Tools:    []Tool{{Type: "provider", ID: "openai.mcp", Args: c.args}},
			})
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			tools, _ := body["tools"].([]map[string]any)
			if len(tools) != 1 {
				t.Fatalf("tools=%v", body["tools"])
			}
			got, has := tools[0]["allowed_tools"]
			if c.wantAbsent {
				if has {
					t.Errorf("allowed_tools should be absent, got %v", got)
				}
				return
			}
			if !has {
				t.Fatalf("allowed_tools missing: %v", tools[0])
			}
			if !reflect.DeepEqual(normalizeParityWire(got), normalizeParityWire(c.want)) {
				t.Errorf("allowed_tools=%v (%T), want %v (%T)", got, got, c.want, c.want)
			}
		})
	}
}

// normalizeParityWire renders wire values into comparable form ([]any of
// strings / map form) for the allowed_tools assertions.
func normalizeParityWire(v any) any {
	if names, ok := stringSliceValue(v); ok {
		out := make([]any, 0, len(names))
		for _, s := range names {
			out = append(out, s)
		}
		return out
	}
	if m, ok := v.(map[string]any); ok {
		out := map[string]any{}
		for k, val := range m {
			out[k] = normalizeParityWire(val)
		}
		return out
	}
	return v
}

// TestResponsesStoreFalseReplaysShellCallNatively verifies that with
// store=false a provider-executed shell call replays as a native
// shell_call item (paired with shell_call_output), not a generic
// function_call.
func TestResponsesStoreFalseReplaysShellCallNatively(t *testing.T) {
	respBody := `{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	storeFalse := false
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Store: &storeFalse,
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
			AssistantMessage{Content: []AssistantContent{ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: "sc-call",
					ToolName:   "shell",
					Input:      []byte(`{"commands":["ls"],"timeout_ms":1000}`),
					ProviderMetadata: ProviderMetadata{
						"openai": map[string]any{"itemId": "sc-1"},
					},
				},
				ProviderExecuted: true,
			}}},
			ToolMessage{Content: []ToolContent{ToolResultContent{
				ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "sc-call", Output: ToolResultOutput{Type: "text", Value: "ok"}},
				ToolName:          "shell",
			}}},
		},
		Tools: []Tool{Shell(ShellArgs{})},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, res.Request.Body)
	input, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("input: %v", body["input"])
	}
	var sawShellCall, sawShellOutput, sawFunctionCall bool
	for _, item := range input {
		m, _ := item.(map[string]any)
		switch m["type"] {
		case "shell_call":
			sawShellCall = true
			if m["call_id"] != "sc-call" {
				t.Errorf("shell_call call_id = %v", m["call_id"])
			}
			if m["id"] != "sc-1" {
				t.Errorf("shell_call id = %v (itemId should round-trip)", m["id"])
			}
			if m["status"] != "completed" {
				t.Errorf("shell_call status = %v", m["status"])
			}
			action, _ := m["action"].(map[string]any)
			commands, _ := action["commands"].([]any)
			if len(commands) != 1 || commands[0] != "ls" {
				t.Errorf("shell_call action = %v", m["action"])
			}
		case "shell_call_output":
			sawShellOutput = true
		case "function_call":
			sawFunctionCall = true
		}
	}
	if !sawShellCall {
		t.Errorf("expected native shell_call item, input: %v", input)
	}
	if !sawShellOutput {
		t.Errorf("expected shell_call_output item, input: %v", input)
	}
	if sawFunctionCall {
		t.Errorf("provider-executed shell call must not replay as function_call, input: %v", input)
	}
}

// TestResponsesCompactionTriggerGenerate verifies a CompactionTrigger
// option appends a terminal compaction_trigger input item on generate.
func TestResponsesCompactionTriggerGenerate(t *testing.T) {
	respBody := `{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"compactionTrigger": CompactionTrigger{Instructions: "summarize"}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	assertTerminalCompactionTrigger(t, res.Request.Body)
}

// TestResponsesCompactionTriggerStream verifies a CompactionTrigger
// option appends a terminal compaction_trigger input item on stream.
func TestResponsesCompactionTriggerStream(t *testing.T) {
	sseBody := "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"id\":\"r1\"}}\n\n"
	f := &recordingFetcher{responses: []*http.Response{response(200, sseBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoStream(context.Background(), ResponsesStreamOptions{
		ResponsesGenerateOptions: ResponsesGenerateOptions{
			Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
			ProviderOptions: ProviderOptions{"openai": map[string]any{"compactionTrigger": CompactionTrigger{Instructions: "summarize"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	for range res.Parts {
	}
	assertTerminalCompactionTrigger(t, res.Request.Body)
}

func assertTerminalCompactionTrigger(t *testing.T, reqBody []byte) {
	t.Helper()
	body := decodeRequestBody(t, reqBody)
	input, ok := body["input"].([]any)
	if !ok || len(input) == 0 {
		t.Fatalf("input: %v", body["input"])
	}
	last, _ := input[len(input)-1].(map[string]any)
	if last["type"] != "compaction_trigger" {
		t.Fatalf("last input item should be compaction_trigger, got: %v", input[len(input)-1])
	}
	// Upstream never sends instructions: the item is always bare.
	if _, has := last["instructions"]; has {
		t.Errorf("compaction_trigger must be bare, got: %v", last)
	}
}

// TestResponsesParseParallelWrapper is a table test for the Azure
// parallel-dispatch wrapper expansion in generate responses.
func TestResponsesParseParallelWrapper(t *testing.T) {
	call := func(id, name string) string {
		return `{"type":"function_call","call_id":"` + id + `","name":"` + name + `","arguments":"{}"}`
	}
	for _, tc := range []struct {
		name      string
		output    string
		wantCalls []string
	}{
		{"unwrapped single call", `[` + call("c1", "f1") + `]`, []string{"f1"}},
		{"wrapped calls key", `[{"type":"parallel_tool_calls","calls":[` + call("c1", "f1") + `,` + call("c2", "f2") + `]}]`, []string{"f1", "f2"}},
		{"wrapped tool_calls key", `[{"type":"parallel","tool_calls":[` + call("c1", "f1") + `]}]`, []string{"f1"}},
		{"nested wrappers", `[{"type":"parallel_tool_calls","calls":[{"type":"parallel_tool_calls","calls":[` + call("c1", "f1") + `]}]}]`, []string{"f1"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			respBody := `{"id":"r","status":"completed","output":` + tc.output + `}`
			f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
			p := newOpenAIForTest(f, "https://example.test/v1")
			res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
				Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
			})
			if err != nil {
				t.Fatalf("DoGenerate: %v", err)
			}
			var names []string
			for _, c := range res.Content {
				if tc, ok := c.(ToolCallContent); ok {
					names = append(names, tc.ToolName)
				}
			}
			if len(names) != len(tc.wantCalls) {
				t.Fatalf("tool calls = %v, want %v", names, tc.wantCalls)
			}
			for i := range names {
				if names[i] != tc.wantCalls[i] {
					t.Errorf("tool calls = %v, want %v", names, tc.wantCalls)
					break
				}
			}
		})
	}
}

// TestResponsesToolResultBreakpointPassthrough is a table test verifying
// breakpoint markers on scalar tool results are forwarded to the wire
// function_call_output item in input_text array form carrying
// prompt_cache_breakpoint, and never added for content-type results.
func TestResponsesToolResultBreakpointPassthrough(t *testing.T) {
	for _, tc := range []struct {
		name           string
		output         ToolResultOutput
		breakpoint     any
		wantBreakpoint bool
	}{
		{"text with breakpoint", ToolResultOutput{Type: "text", Value: "ok"}, map[string]any{"mode": "explicit"}, true},
		{"json with breakpoint", ToolResultOutput{Type: "json", Value: map[string]any{"a": 1}}, map[string]any{"mode": "explicit"}, true},
		{"content skips breakpoint", ToolResultOutput{Type: "content", Value: []any{"x"}}, map[string]any{"mode": "explicit"}, false},
		{"scalar without breakpoint", ToolResultOutput{Type: "text", Value: "ok"}, nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			respBody := `{"id":"r","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}`
			f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
			p := newOpenAIForTest(f, "https://example.test/v1")
			result := openaicompatible.ToolResultContent{ToolCallID: "c1", Output: tc.output}
			if tc.breakpoint != nil {
				result.ProviderOptions = openaicompatible.ProviderMetadata{"openai": map[string]any{"breakpoint": tc.breakpoint}}
			}
			res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
				Messages: []Message{
					UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
					ToolMessage{Content: []ToolContent{ToolResultContent{
						ToolResultContent: result,
						ToolName:          "get_weather",
					}}},
				},
			})
			if err != nil {
				t.Fatalf("DoGenerate: %v", err)
			}
			body := decodeRequestBody(t, res.Request.Body)
			input, ok := body["input"].([]any)
			if !ok || len(input) != 2 {
				t.Fatalf("input: %v", body["input"])
			}
			item, _ := input[1].(map[string]any)
			if item["type"] != "function_call_output" {
				t.Fatalf("item type = %v", item["type"])
			}
			if _, has := item["breakpoint"]; has {
				t.Errorf("top-level breakpoint must not be set (item: %v)", item)
			}
			parts, isArray := item["output"].([]any)
			if tc.wantBreakpoint {
				if !isArray || len(parts) != 1 {
					t.Fatalf("output should be a single input_text part, got: %v", item["output"])
				}
				part, _ := parts[0].(map[string]any)
				if part["type"] != "input_text" {
					t.Errorf("part type = %v, want input_text", part["type"])
				}
				bp, _ := part["prompt_cache_breakpoint"].(map[string]any)
				if bp["mode"] != "explicit" {
					t.Errorf("prompt_cache_breakpoint = %v, want {mode:explicit}", part["prompt_cache_breakpoint"])
				}
			} else if isArray {
				t.Errorf("output should stay a scalar string, got: %v", item["output"])
			}
		})
	}
}
