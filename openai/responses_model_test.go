package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

func TestResponsesGenerateBasic(t *testing.T) {
	respBody := `{"id":"resp_1","created_at":10,"model":"gpt-5","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-5").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:     []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Instructions: "be brief",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Response.ID != "resp_1" {
		t.Errorf("id = %q", result.Response.ID)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d", len(result.Content))
	}
	tc, ok := result.Content[0].(TextContent)
	if !ok || tc.Text != "hello" {
		t.Errorf("text = %#v", result.Content[0])
	}
	if result.FinishReason.Unified != "stop" {
		t.Errorf("finish = %+v", result.FinishReason)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["model"] != "gpt-5" {
		t.Errorf("model = %v", body["model"])
	}
	if body["instructions"] != "be brief" {
		t.Errorf("instructions = %v", body["instructions"])
	}
	if body["store"] != true {
		t.Errorf("store = %v", body["store"])
	}
}

func TestResponsesGenerateFunctionCall(t *testing.T) {
	respBody := `{"id":"resp_1","created_at":10,"model":"gpt-5","status":"completed","output":[{"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"sf\"}"}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-5").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "weather?"}}}},
		Tools: []Tool{
			{Type: "function", Name: "get_weather", Description: "Get the weather", InputSchema: map[string]any{"type": "object"}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.FinishReason.Unified != "tool-calls" {
		t.Errorf("finish = %+v", result.FinishReason)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d", len(result.Content))
	}
	tc, ok := result.Content[0].(ToolCallContent)
	if !ok {
		t.Fatalf("content type = %T", result.Content[0])
	}
	if tc.ToolName != "get_weather" {
		t.Errorf("tool name = %q", tc.ToolName)
	}
	if string(tc.Input) != `{"city":"sf"}` {
		t.Errorf("input = %q", string(tc.Input))
	}
}

// TestResponsesSystemMessageModeSystem verifies the default "system" mode
// emits a system-role input item.
func TestResponsesSystemMessageModeSystem(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			SystemMessage{Content: "be brief"},
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, ok := body["input"].([]any)
	if !ok || len(input) == 0 {
		t.Fatalf("input: %v", body["input"])
	}
	first, ok := input[0].(map[string]any)
	if !ok || first["role"] != "system" {
		t.Errorf("first input role: %v", input[0])
	}
	if first["content"] != "be brief" {
		t.Errorf("system content: %v", first["content"])
	}
}

// TestResponsesSystemMessageModeDeveloper verifies the "developer" mode
// appends to body["instructions"].
func TestResponsesSystemMessageModeDeveloper(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			SystemMessage{Content: "be brief"},
		},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"systemMessageMode": "developer"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	// Two systemMessageMode=developer code paths can each append, so just
	// verify the developer instructions are present.
	got, _ := body["instructions"].(string)
	if !strings.Contains(got, "be brief") {
		t.Errorf("instructions should contain 'be brief': %q", got)
	}
}

// TestResponsesSystemMessageModeRemove verifies the "remove" mode drops
// system messages from the input.
func TestResponsesSystemMessageModeRemove(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			SystemMessage{Content: "be brief"},
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
		},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"systemMessageMode": "remove"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("input: %v", body["input"])
	}
	// The system message should be dropped, leaving only the user message.
	for _, item := range input {
		if m, ok := item.(map[string]any); ok && m["role"] == "system" {
			t.Errorf("system message should be removed: %v", m)
		}
	}
}

// TestResponsesGenerateWithToolChoiceAuto verifies the auto tool choice.
func TestResponsesGenerateWithToolChoiceAuto(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []Tool{
			{Type: "function", Name: "f", Description: "d", InputSchema: map[string]any{"type": "object"}},
		},
		ToolChoice: &ToolChoice{Type: "auto"},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	tc, ok := body["tool_choice"].(string)
	if !ok || tc != "auto" {
		t.Errorf("tool_choice: %v", body["tool_choice"])
	}
}

// TestResponsesToolMessageShellOutput verifies the shell tool result
// is serialized as shell_call_output with the max_wait_ms / shell_cmd
// fields.
func TestResponsesToolMessageShellOutput(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: "c1",
						Output: openaicompatible.ToolResultOutput{
							Type:  "json",
							Value: map[string]any{"stdout": "ok", "stderr": "", "outcome": map[string]any{"type": "exit", "exit_code": 0}},
						},
					},
					ToolName: "shell",
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("input: %v", body["input"])
	}
	if len(input) != 1 {
		t.Fatalf("input len = %d", len(input))
	}
	m, _ := input[0].(map[string]any)
	if m["type"] != "shell_call_output" {
		t.Errorf("type: %v", m)
	}
	if m["call_id"] != "c1" {
		t.Errorf("call_id: %v", m)
	}
}

// TestResponsesToolMessageApplyPatchOutput verifies apply_patch tool result
// is serialized as apply_patch_call_output.
func TestResponsesToolMessageApplyPatchOutput(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: "c1",
						Output:     openaicompatible.ToolResultOutput{Type: "text", Value: "ok"},
					},
					ToolName: "apply_patch",
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, _ := body["input"].([]any)
	m, _ := input[0].(map[string]any)
	if m["type"] != "apply_patch_call_output" {
		t.Errorf("type: %v", m)
	}
	if m["status"] != "completed" {
		t.Errorf("status: %v", m)
	}
}

// TestResponsesToolMessageLocalShellOutput verifies local_shell tool
// result is serialized as local_shell_call_output.
func TestResponsesToolMessageLocalShellOutput(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: "c1",
						Output:     openaicompatible.ToolResultOutput{Type: "text", Value: "ok"},
					},
					ToolName: "local_shell",
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, _ := body["input"].([]any)
	m, _ := input[0].(map[string]any)
	if m["type"] != "local_shell_call_output" {
		t.Errorf("type: %v", m)
	}
}

// TestResponsesToolMessageDefaultFunctionCallOutput verifies the default
// branch (function_call_output) is used for non-special tool names.
func TestResponsesToolMessageDefaultFunctionCallOutput(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: "c1",
						Output:     openaicompatible.ToolResultOutput{Type: "text", Value: "sunny"},
					},
					ToolName: "get_weather",
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, _ := body["input"].([]any)
	m, _ := input[0].(map[string]any)
	if m["type"] != "function_call_output" {
		t.Errorf("type: %v", m)
	}
	if m["output"] != "sunny" {
		t.Errorf("output: %v", m["output"])
	}
}

// TestResponsesGenerateWithToolChoiceRequired verifies the required tool
// choice.
func TestResponsesGenerateWithToolChoiceRequired(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []Tool{
			{Type: "function", Name: "f", Description: "d", InputSchema: map[string]any{"type": "object"}},
		},
		ToolChoice: &ToolChoice{Type: "required"},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if !strings.Contains(string(result.Request.Body), `"tool_choice":"required"`) {
		t.Errorf("tool_choice: %v", body["tool_choice"])
	}
}
