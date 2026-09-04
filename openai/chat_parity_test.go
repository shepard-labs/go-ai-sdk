package openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestChatEmptyToolCallIDReturnsError verifies an empty tool-call id in a
// non-streaming chat response is rejected with InvalidResponseDataError
// (no id is synthesized).
func TestChatEmptyToolCallIDReturnsError(t *testing.T) {
	cases := []struct {
		name    string
		call    map[string]any
		wantErr bool
		wantID  string
	}{
		{
			"valid-id",
			map[string]any{"id": "call_1", "function": map[string]any{"name": "f", "arguments": "{}"}},
			false, "call_1",
		},
		{
			"missing-id",
			map[string]any{"function": map[string]any{"name": "f", "arguments": "{}"}},
			true, "",
		},
		{
			"empty-id",
			map[string]any{"id": "", "function": map[string]any{"name": "f", "arguments": "{}"}},
			true, "",
		},
		{
			"whitespace-id",
			map[string]any{"id": "   ", "function": map[string]any{"name": "f", "arguments": "{}"}},
			false, "   ",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertChatToolCallFromResponse(c.call)
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
				t.Fatalf("convert: %v", err)
			}
			if got.ToolCallID != c.wantID {
				t.Errorf("ToolCallID=%q, want %q (must not be synthesized)", got.ToolCallID, c.wantID)
			}
		})
	}
}

// TestStrictModeEmptySchemaSerializesEmptyObject verifies strict mode
// serializes {} (never null/omitted) for function tool parameters, and
// that chat tool-call arguments default empty input to "{}".
func TestStrictModeEmptySchemaSerializesEmptyObject(t *testing.T) {
	t.Run("responses-parameters", func(t *testing.T) {
		schema := map[string]any{"type": "object"}
		cases := []struct {
			name       string
			tool       Tool
			wantAbsent bool
			want       any
		}{
			{"nil-schema-default-strict", Tool{Type: "function", Name: "f"}, false, map[string]any{}},
			{"nil-schema-explicit-strict", Tool{Type: "function", Name: "f", Strict: boolP(true)}, false, map[string]any{}},
			{"nil-schema-non-strict-omitted", Tool{Type: "function", Name: "f", Strict: boolP(false)}, true, nil},
			{"schema-preserved", Tool{Type: "function", Name: "f", InputSchema: schema}, false, schema},
			{"schema-preserved-non-strict", Tool{Type: "function", Name: "f", Strict: boolP(false), InputSchema: schema}, false, schema},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				payload, _, err := buildResponsesFunctionTool(c.tool)
				if err != nil {
					t.Fatalf("build: %v", err)
				}
				got, has := payload["parameters"]
				if c.wantAbsent {
					if has {
						t.Errorf("parameters should be omitted, got %v", got)
					}
					return
				}
				if !has {
					t.Fatal("parameters missing")
				}
				if !reflect.DeepEqual(got, c.want) {
					t.Errorf("parameters=%v (%T), want %v", got, got, c.want)
				}
			})
		}
	})

	t.Run("chat-arguments", func(t *testing.T) {
		cases := []struct {
			name   string
			input  string
			want   string
			compat bool
		}{
			{"empty-becomes-object", "", "{}", false},
			{"verbatim-object", `{"x":1}`, `{"x":1}`, false},
			{"compat-empty-becomes-object", "", "{}", true},
			{"compat-verbatim", `{"x":1}`, `{"x":1}`, true},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				m := newTestChatModel()
				var content []AssistantContent
				if c.compat {
					content = []AssistantContent{openaicompatible.ToolCallContent{
						ToolCallID: "c1", ToolName: "f",
						Input: json.RawMessage(c.input),
					}}
				} else {
					content = []AssistantContent{ToolCallContent{
						ToolCallContentEmbed: ToolCallContentEmbed{
							ToolCallID: "c1", ToolName: "f",
							Input: json.RawMessage(c.input),
						},
					}}
				}
				out, err := m.convertAssistantMessage(AssistantMessage{Content: content})
				if err != nil {
					t.Fatalf("convert: %v", err)
				}
				tcs, _ := out["tool_calls"].([]map[string]any)
				if len(tcs) != 1 {
					t.Fatalf("tool_calls=%v", out["tool_calls"])
				}
				fn, _ := tcs[0]["function"].(map[string]any)
				if fn["arguments"] != c.want {
					t.Errorf("arguments=%q, want %q", fn["arguments"], c.want)
				}
			})
		}
	})
}
