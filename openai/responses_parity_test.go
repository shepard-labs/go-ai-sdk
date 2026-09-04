package openai

import (
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
