package openai

import (
	"context"
	"net/http"
	"testing"
)

// Helper that runs a responses request with provider options and returns
// the request body.
func runResponsesWithOptions(t *testing.T, opts ProviderOptions) map[string]any {
	t.Helper()
	respBody := `{"id":"r","created_at":1,"model":"gpt-5","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-5").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: opts,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	return decodeRequestBody(t, result.Request.Body)
}

// Verifies that the responses API forwards user, prompt_cache_key, and
// safety_identifier to the wire.
func TestResponsesForwardsUserCacheSafety(t *testing.T) {
	body := runResponsesWithOptions(t, ProviderOptions{
		"openai": map[string]any{
			"user":             "user-1",
			"promptCacheKey":   "cache-1",
			"safetyIdentifier": "user-1",
		},
	})
	if body["user"] != "user-1" {
		t.Errorf("user: %v", body["user"])
	}
	if body["prompt_cache_key"] != "cache-1" {
		t.Errorf("prompt_cache_key: %v", body["prompt_cache_key"])
	}
	if body["safety_identifier"] != "user-1" {
		t.Errorf("safety_identifier: %v", body["safety_identifier"])
	}
}

// Verifies that the responses API forwards truncation, max_tool_calls, and
// parallel_tool_calls.
func TestResponsesForwardsTruncationMaxAndParallel(t *testing.T) {
	body := runResponsesWithOptions(t, ProviderOptions{
		"openai": map[string]any{
			"truncation":        "auto",
			"maxToolCalls":      3,
			"parallelToolCalls": true,
		},
	})
	if body["truncation"] != "auto" {
		t.Errorf("truncation: %v", body["truncation"])
	}
	if v, ok := body["max_tool_calls"].(float64); !ok || v != 3 {
		t.Errorf("max_tool_calls: %v", body["max_tool_calls"])
	}
	if body["parallel_tool_calls"] != true {
		t.Errorf("parallel_tool_calls: %v", body["parallel_tool_calls"])
	}
}

// Verifies that the responses API forwards top_logprobs / logprobs.
func TestResponsesForwardsLogprobs(t *testing.T) {
	body := runResponsesWithOptions(t, ProviderOptions{
		"openai": map[string]any{
			"logprobs":    true,
			"topLogprobs": 5,
		},
	})
	if body["logprobs"] != true {
		t.Errorf("logprobs: %v", body["logprobs"])
	}
	if v, ok := body["top_logprobs"].(float64); !ok || v != 5 {
		t.Errorf("top_logprobs: %v", body["top_logprobs"])
	}
}

// Verifies that the responses API strips service_tier "flex" on models
// that don't support flex processing.
func TestResponsesRejectsFlexServiceTierOnNonFlex(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"o1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("o1").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"serviceTier": "flex"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, has := body["service_tier"]; has {
		t.Errorf("service_tier should be stripped on o1: %v", body["service_tier"])
	}
}

// Verifies that the responses API forwards store and metadata.
func TestResponsesForwardsStoreAndMetadata(t *testing.T) {
	body := runResponsesWithOptions(t, ProviderOptions{
		"openai": map[string]any{
			"store":    true,
			"metadata": map[string]any{"trace_id": "x"},
		},
	})
	if body["store"] != true {
		t.Errorf("store: %v", body["store"])
	}
	md, ok := body["metadata"].(map[string]any)
	if !ok || md["trace_id"] != "x" {
		t.Errorf("metadata: %v", body["metadata"])
	}
}

// Verifies that store=false in the Responses API request body is honored
// when supplied.
func TestResponsesForwardsStoreFalse(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-5","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	storeFalse := false
	result, err := p.Responses("gpt-5").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Store:    &storeFalse,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["store"] != false {
		t.Errorf("store: %v", body["store"])
	}
}
