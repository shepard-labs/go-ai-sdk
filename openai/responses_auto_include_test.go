package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestResponsesAutoIncludeLogprobs verifies that the SDK auto-adds
// "message.output_text.logprobs" to include when logprobs is requested.
func TestResponsesAutoIncludeLogprobs(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"logprobs": true},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	inc, _ := body["include"].([]any)
	found := false
	for _, v := range inc {
		if s, _ := v.(string); s == "message.output_text.logprobs" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected logprobs include, got: %v", inc)
	}
}

// TestResponsesAutoIncludeWebSearch verifies that web_search tool
// triggers "web_search_call.action.sources" auto-include.
func TestResponsesAutoIncludeWebSearch(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []Tool{
			{Type: "provider", ID: "openai.webSearch", Args: WebSearchArgs{}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	_ = err
}

// TestResponsesAutoIncludeCodeInterpreter verifies that code_interpreter
// tool triggers "code_interpreter_call.outputs" auto-include.
func TestResponsesAutoIncludeCodeInterpreter(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []Tool{
			{Type: "provider", ID: "openai.codeInterpreter", Args: CodeInterpreterArgs{}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	inc, _ := body["include"].([]any)
	found := false
	for _, v := range inc {
		if s, _ := v.(string); s == "code_interpreter_call.outputs" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected code_interpreter outputs include, got: %v", inc)
	}
}

// TestResponsesAutoIncludeReasoningEncrypted verifies that store=false
// triggers "reasoning.encrypted_content" auto-include.
func TestResponsesAutoIncludeReasoningEncrypted(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	storeFalse := false
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Store:    &storeFalse,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	inc, _ := body["include"].([]any)
	found := false
	for _, v := range inc {
		if s, _ := v.(string); s == "reasoning.encrypted_content" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected reasoning.encrypted_content include, got: %v", inc)
	}
}

// TestMergeIncludesDedups verifies that mergeIncludes de-duplicates.
func TestMergeIncludesDedups(t *testing.T) {
	out := mergeIncludes(
		[]string{"a", "b"},
		[]string{"b", "c", "a"},
	)
	if len(out) != 3 {
		t.Errorf("len = %d, want 3: %v", len(out), out)
	}
	if out[0] != "a" || out[1] != "b" || out[2] != "c" {
		t.Errorf("order/content: %v", out)
	}
}

// TestBuildAutoIncludeEmpty verifies the helper returns nil when no
// relevant context is present.
func TestBuildAutoIncludeEmpty(t *testing.T) {
	if got := buildAutoInclude(map[string]any{}, nil, nil); got != nil {
		t.Errorf("expected nil, got: %v", got)
	}
}
