package openai

import (
	"context"
	"net/http"
	"testing"
)

// Verifies that logprobs/top_logprobs are forwarded to the wire for
// non-reasoning models. Per the chat model logic, a positive int in the
// "logprobs" key sets both logprobs: true and top_logprobs: N.
func TestChatForwardsLogprobs(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"logprobs": 5}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["logprobs"] != true {
		t.Errorf("logprobs: %v", body["logprobs"])
	}
	if v, ok := body["top_logprobs"].(float64); !ok || v != 5 {
		t.Errorf("top_logprobs: %v", body["top_logprobs"])
	}
}

// Verifies that logprobs are stripped (with warning) on reasoning models.
func TestChatStripsLogprobsOnReasoning(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"o3","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("o3").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"logprobs": 5}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, has := body["logprobs"]; has {
		t.Errorf("logprobs should be stripped on o3: %v", body["logprobs"])
	}
}

// Verifies that metadata is forwarded to the wire.
func TestChatForwardsMetadata(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := CreateOpenAICompatibleProviderForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"metadata": map[string]any{"trace_id": "abc"}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	md, ok := body["metadata"].(map[string]any)
	if !ok || md["trace_id"] != "abc" {
		t.Errorf("metadata: %v", body["metadata"])
	}
}

// Verifies that prompt_cache_key is forwarded to the wire.
func TestChatForwardsPromptCacheKey(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := CreateOpenAICompatibleProviderForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"promptCacheKey": "my-cache-key"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["prompt_cache_key"] != "my-cache-key" {
		t.Errorf("prompt_cache_key: %v", body["prompt_cache_key"])
	}
}

// Verifies that reasoning_effort is forwarded to the wire.
func TestChatForwardsReasoningEffort(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"o3","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := CreateOpenAICompatibleProviderForTest(f, "https://example.test/v1")
	result, err := p.Chat("o3").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"reasoningEffort": "high"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort: %v", body["reasoning_effort"])
	}
}

// Verifies that parallel_tool_calls is forwarded to the wire.
func TestChatForwardsParallelToolCalls(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := CreateOpenAICompatibleProviderForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"parallelToolCalls": true}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["parallel_tool_calls"] != true {
		t.Errorf("parallel_tool_calls: %v", body["parallel_tool_calls"])
	}
}

// Verifies that the store field is forwarded.
func TestChatForwardsStore(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"store": true}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["store"] != true {
		t.Errorf("store: %v", body["store"])
	}
}

// Verifies that the safety_identifier field is forwarded.
func TestChatForwardsSafetyIdentifier(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"safetyIdentifier": "user-abc"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["safety_identifier"] != "user-abc" {
		t.Errorf("safety_identifier: %v", body["safety_identifier"])
	}
}

// Verifies that the prediction field is forwarded (chat-only).
func TestChatForwardsPrediction(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	pred := map[string]any{"type": "content", "content": "x"}
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"prediction": pred}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	got, ok := body["prediction"].(map[string]any)
	if !ok || got["type"] != "content" {
		t.Errorf("prediction: %v", body["prediction"])
	}
}

// Verifies that service_tier "flex" is rejected on models that don't
// support flex processing.
func TestChatRejectsFlexServiceTierOnNonFlex(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"o1","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("o1").DoGenerate(context.Background(), GenerateOptions{
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

// Verifies that prompt_cache_retention is forwarded for gpt-5.1.
func TestChatForwardsPromptCacheRetentionOnGPT51(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-5.1","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-5.1").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"promptCacheRetention": "24h"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["prompt_cache_retention"] != "24h" {
		t.Errorf("prompt_cache_retention: %v", body["prompt_cache_retention"])
	}
}

// Verifies that on gpt-5.1 with reasoningEffort: "none", temperature is
// kept (per the SupportsNonReasoningParameters capability), and that
// reasoning_effort: "none" is forwarded to the wire.
func TestChatGpt51ReasoningNoneKeepsTemperature(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-5.1","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	temperature := 0.5
	result, err := p.Chat("gpt-5.1").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Temperature:     &temperature,
		ProviderOptions: ProviderOptions{"openai": map[string]any{"reasoningEffort": "none"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if v, ok := body["temperature"].(float64); !ok || v != 0.5 {
		t.Errorf("temperature should be kept on gpt-5.1 + reason.none: %v", body["temperature"])
	}
	if body["reasoning_effort"] != "none" {
		t.Errorf("reasoning_effort: %v", body["reasoning_effort"])
	}
}

// Verifies that on gpt-5.1 WITHOUT reasoningEffort: "none", temperature
// is stripped (per the SupportsNonReasoningParameters + non-"none" rule).
func TestChatGpt51ReasoningDefaultStripsTemperature(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-5.1","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	temperature := 0.7
	result, err := p.Chat("gpt-5.1").DoGenerate(context.Background(), GenerateOptions{
		Messages:    []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Temperature: &temperature,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, has := body["temperature"]; has {
		t.Errorf("temperature should be stripped on gpt-5.1 with default reasoning: %v", body["temperature"])
	}
}

// Verifies that gpt-5-chat-latest is treated as non-reasoning: it keeps
// temperature and is in system message mode.
func TestChatGpt5ChatLatestKeepsTemperature(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-5-chat-latest","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	temperature := 0.7
	result, err := p.Chat("gpt-5-chat-latest").DoGenerate(context.Background(), GenerateOptions{
		Messages:    []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Temperature: &temperature,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if v, ok := body["temperature"].(float64); !ok || v != 0.7 {
		t.Errorf("temperature should be kept on gpt-5-chat-latest: %v", body["temperature"])
	}
	if IsReasoningModel("gpt-5-chat-latest") {
		t.Errorf("gpt-5-chat-latest should not be a reasoning model")
	}
	if SystemMessageMode("gpt-5-chat-latest") != "system" {
		t.Errorf("gpt-5-chat-latest systemMessageMode = %q", SystemMessageMode("gpt-5-chat-latest"))
	}
}
