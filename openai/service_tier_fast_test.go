package openai

import (
	"context"
	"net/http"
	"testing"
)

// runChatWithServiceTier runs a chat request for model with the given
// serviceTier provider option, returning the request body and warnings.
func runChatWithServiceTier(t *testing.T, model, tier string) (map[string]any, []Warning) {
	t.Helper()
	respBody := `{"id":"r","created":1,"model":"` + model + `","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat(model).DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"serviceTier": tier}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	return decodeRequestBody(t, result.Request.Body), result.Warnings
}

// runResponsesWithServiceTier runs a responses request for model with the
// given serviceTier provider option, returning body and warnings.
func runResponsesWithServiceTier(t *testing.T, model, tier string) (map[string]any, []Warning) {
	t.Helper()
	respBody := `{"id":"r","created_at":1,"model":"` + model + `","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses(model).DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"serviceTier": tier}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	return decodeRequestBody(t, result.Request.Body), result.Warnings
}

// TestChatServiceTierFast verifies service_tier "fast" passes through on
// priority-capable models without warnings.
func TestChatServiceTierFast(t *testing.T) {
	body, warnings := runChatWithServiceTier(t, "gpt-5", "fast")
	if body["service_tier"] != "fast" {
		t.Errorf("service_tier: %v", body["service_tier"])
	}
	if len(warnings) != 0 {
		t.Errorf("warnings: %+v", warnings)
	}
}

// TestChatServiceTierFastUnsupported verifies service_tier "fast" is
// stripped with a warning on models without priority processing.
func TestChatServiceTierFastUnsupported(t *testing.T) {
	body, warnings := runChatWithServiceTier(t, "gpt-5-nano", "fast")
	if _, has := body["service_tier"]; has {
		t.Errorf("service_tier should be stripped on gpt-5-nano: %v", body["service_tier"])
	}
	if len(warnings) == 0 {
		t.Error("expected a service_tier warning on gpt-5-nano")
	}
}

// TestResponsesServiceTierFast verifies service_tier "fast" passes
// through on priority-capable models without warnings.
func TestResponsesServiceTierFast(t *testing.T) {
	body, warnings := runResponsesWithServiceTier(t, "gpt-5.6", "fast")
	if body["service_tier"] != "fast" {
		t.Errorf("service_tier: %v", body["service_tier"])
	}
	if len(warnings) != 0 {
		t.Errorf("warnings: %+v", warnings)
	}
}

// TestResponsesServiceTierFastUnsupported verifies service_tier "fast" is
// stripped with a warning on models without priority processing.
func TestResponsesServiceTierFastUnsupported(t *testing.T) {
	body, warnings := runResponsesWithServiceTier(t, "gpt-4o-mini", "fast")
	if _, has := body["service_tier"]; has {
		t.Errorf("service_tier should be stripped on gpt-4o-mini: %v", body["service_tier"])
	}
	if len(warnings) == 0 {
		t.Error("expected a service_tier warning on gpt-4o-mini")
	}
}
