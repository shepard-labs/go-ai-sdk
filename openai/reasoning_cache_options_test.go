package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestResponsesReasoningModeContext verifies reasoningMode and
// reasoningContext provider options land in the reasoning object.
func TestResponsesReasoningModeContext(t *testing.T) {
	body := runResponsesWithOptions(t, ProviderOptions{
		"openai": map[string]any{
			"reasoningEffort":  "medium",
			"reasoningMode":    "standard",
			"reasoningContext": "all_turns",
		},
	})
	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning: %v", body["reasoning"])
	}
	if reasoning["effort"] != "medium" {
		t.Errorf("effort: %v", reasoning["effort"])
	}
	if reasoning["mode"] != "standard" {
		t.Errorf("mode: %v", reasoning["mode"])
	}
	if reasoning["context"] != "all_turns" {
		t.Errorf("context: %v", reasoning["context"])
	}
}

// TestResponsesReasoningEffortMax verifies the "max" effort value passes
// through for gpt-5.6 models.
func TestResponsesReasoningEffortMax(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-5.6","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-5.6").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"reasoningEffort": "max"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning: %v", body["reasoning"])
	}
	if reasoning["effort"] != "max" {
		t.Errorf("effort: %v", reasoning["effort"])
	}
}

// TestResponsesPromptCacheOptions verifies promptCacheOptions forwards
// with explicit mode and ttl.
func TestResponsesPromptCacheOptions(t *testing.T) {
	body := runResponsesWithOptions(t, ProviderOptions{
		"openai": map[string]any{
			"promptCacheOptions": map[string]any{"mode": "explicit", "ttl": "30m"},
		},
	})
	opts, ok := body["prompt_cache_options"].(map[string]any)
	if !ok {
		t.Fatalf("prompt_cache_options: %v", body["prompt_cache_options"])
	}
	if opts["mode"] != "explicit" || opts["ttl"] != "30m" {
		t.Errorf("prompt_cache_options: %v", opts)
	}
}

// TestResponsesPromptCacheOptionsImplicit verifies the implicit mode
// forwards as well.
func TestResponsesPromptCacheOptionsImplicit(t *testing.T) {
	body := runResponsesWithOptions(t, ProviderOptions{
		"openai": map[string]any{
			"promptCacheOptions": map[string]any{"mode": "implicit"},
		},
	})
	opts, ok := body["prompt_cache_options"].(map[string]any)
	if !ok {
		t.Fatalf("prompt_cache_options: %v", body["prompt_cache_options"])
	}
	if opts["mode"] != "implicit" {
		t.Errorf("prompt_cache_options: %v", opts)
	}
}

// TestChatPromptCacheOptions verifies the chat path forwards
// promptCacheOptions to prompt_cache_options.
func TestChatPromptCacheOptions(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-5.6","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-5.6").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{
			"promptCacheOptions": map[string]any{"mode": "explicit", "ttl": "30m"},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	opts, ok := body["prompt_cache_options"].(map[string]any)
	if !ok {
		t.Fatalf("prompt_cache_options: %v", body["prompt_cache_options"])
	}
	if opts["mode"] != "explicit" || opts["ttl"] != "30m" {
		t.Errorf("prompt_cache_options: %v", opts)
	}
}

// TestResponsesUsageCacheWrite verifies input_tokens_details
// cache_write_tokens surfaces as Usage.InputTokens.CacheWrite.
func TestResponsesUsageCacheWrite(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-5.6","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":100,"output_tokens":50,"input_tokens_details":{"cached_tokens":20,"cache_write_tokens":30}}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-5.6").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Usage.InputTokens.CacheWrite == nil || *result.Usage.InputTokens.CacheWrite != 30 {
		t.Errorf("CacheWrite: %+v", result.Usage.InputTokens.CacheWrite)
	}
	if result.Usage.InputTokens.CacheRead == nil || *result.Usage.InputTokens.CacheRead != 20 {
		t.Errorf("CacheRead: %+v", result.Usage.InputTokens.CacheRead)
	}
}
