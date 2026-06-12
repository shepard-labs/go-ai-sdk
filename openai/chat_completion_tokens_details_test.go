package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
)

// TestChatCompletionTokensDetailsMetadata verifies that the
// completion_tokens_details.accepted_prediction_tokens and
// completion_tokens_details.rejected_prediction_tokens are surfaced
// in result.ProviderMetadata["openai"] per the spec.
func TestChatCompletionTokensDetailsMetadata(t *testing.T) {
	respBody := `{
		"id":"chat-1",
		"created":1,
		"model":"gpt-4o",
		"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{
			"prompt_tokens":10,
			"completion_tokens":20,
			"total_tokens":30,
			"completion_tokens_details":{"accepted_prediction_tokens":15,"rejected_prediction_tokens":5}
		}
	}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	om, ok := result.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected providerMetadata[openai] to be set, got: %v", result.ProviderMetadata)
	}
	if v, _ := om["acceptedPredictionTokens"].(int); v != 15 {
		t.Errorf("acceptedPredictionTokens = %v, want 15", om["acceptedPredictionTokens"])
	}
	if v, _ := om["rejectedPredictionTokens"].(int); v != 5 {
		t.Errorf("rejectedPredictionTokens = %v, want 5", om["rejectedPredictionTokens"])
	}
}

// TestChatCompletionTokensDetailsMetadataNoDetails verifies that
// acceptedPredictionTokens and rejectedPredictionTokens are absent
// (but responseId/modelId are still set per the spec).
func TestChatCompletionTokensDetailsMetadataNoDetails(t *testing.T) {
	respBody := `{
		"id":"chat-1",
		"created":1,
		"model":"gpt-4o",
		"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}
	}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	om, has := result.ProviderMetadata["openai"].(map[string]any)
	if !has {
		t.Fatalf("providerMetadata[openai] should be set (responseId/modelId): %v", result.ProviderMetadata)
	}
	if _, has := om["acceptedPredictionTokens"]; has {
		t.Errorf("acceptedPredictionTokens should not be set: %v", om)
	}
	if _, has := om["rejectedPredictionTokens"]; has {
		t.Errorf("rejectedPredictionTokens should not be set: %v", om)
	}
	if om["responseId"] != "chat-1" {
		t.Errorf("responseId should be set, got: %v", om["responseId"])
	}
	if om["modelId"] != "gpt-4o" {
		t.Errorf("modelId should be set, got: %v", om["modelId"])
	}
}

// TestBuildChatProviderMetadataFromUsage tests the helper directly,
// which is the streaming-path equivalent.
func TestBuildChatProviderMetadataFromUsage(t *testing.T) {
	raw := json.RawMessage(`{"prompt_tokens":10,"completion_tokens":20,"completion_tokens_details":{"accepted_prediction_tokens":15,"rejected_prediction_tokens":5}}`)
	pm := buildChatProviderMetadataFromUsage(raw)
	om, ok := pm["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected openai metadata, got: %v", pm)
	}
	if v, _ := om["acceptedPredictionTokens"].(int); v != 15 {
		t.Errorf("accepted = %v", om["acceptedPredictionTokens"])
	}
	if v, _ := om["rejectedPredictionTokens"].(int); v != 5 {
		t.Errorf("rejected = %v", om["rejectedPredictionTokens"])
	}
}

// TestBuildChatProviderMetadataFromUsageNoDetails verifies the helper
// returns nil when completion_tokens_details is absent.
func TestBuildChatProviderMetadataFromUsageNoDetails(t *testing.T) {
	raw := json.RawMessage(`{"prompt_tokens":10,"completion_tokens":20}`)
	if pm := buildChatProviderMetadataFromUsage(raw); pm != nil {
		t.Errorf("expected nil, got: %v", pm)
	}
	if pm := buildChatProviderMetadataFromUsage(nil); pm != nil {
		t.Errorf("expected nil for nil input, got: %v", pm)
	}
}
