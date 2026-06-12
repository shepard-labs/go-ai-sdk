package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestChatProviderMetadataLogprobs verifies that the chat completion
// response surface logprobs, responseId, and modelId into
// providerMetadata["openai"] per the spec table.
func TestChatProviderMetadataLogprobs(t *testing.T) {
	respBody := `{
		"id":"chat-1",
		"created":1,
		"model":"gpt-4o",
		"choices":[{
			"message":{"role":"assistant","content":"hi"},
			"finish_reason":"stop",
			"logprobs":{"content":[{"token":"hi","logprob":-0.1,"bytes":[104,105]}]}
		}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
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
		t.Fatalf("expected providerMetadata[openai], got %v", result.ProviderMetadata)
	}
	if om["responseId"] != "chat-1" {
		t.Errorf("responseId = %v", om["responseId"])
	}
	if om["modelId"] != "gpt-4o" {
		t.Errorf("modelId = %v", om["modelId"])
	}
	lp, ok := om["logprobs"].(map[string]any)
	if !ok {
		t.Fatalf("expected logprobs, got %T", om["logprobs"])
	}
	if _, has := lp["content"]; !has {
		t.Errorf("logprobs should have content key: %v", lp)
	}
}

// TestResponsesProviderMetadataResponseLevel verifies that the
// Responses response surface responseId and serviceTier into
// providerMetadata["openai"].
func TestResponsesProviderMetadataResponseLevel(t *testing.T) {
	respBody := `{
		"id":"resp_1",
		"created_at":1,
		"model":"gpt-5",
		"status":"completed",
		"service_tier":"default",
		"output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
	}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-5").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	om, ok := result.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected providerMetadata[openai], got %v", result.ProviderMetadata)
	}
	if om["responseId"] != "resp_1" {
		t.Errorf("responseId = %v", om["responseId"])
	}
	if om["serviceTier"] != "default" {
		t.Errorf("serviceTier = %v", om["serviceTier"])
	}
}

// TestResponsesTextContentProviderMetadata verifies that text content
// emitted from a Responses message carries itemId, phase, and
// annotations in ProviderOptions["openai"].
func TestResponsesTextContentProviderMetadata(t *testing.T) {
	respBody := `{
		"id":"resp_1",
		"created_at":1,
		"model":"gpt-5",
		"status":"completed",
		"output":[{
			"type":"message",
			"id":"msg-99",
			"phase":"final_answer",
			"annotations":[{"type":"url_citation","url":"https://example.com","title":"ex"}],
			"content":[{
				"type":"output_text",
				"text":"hello"
			}]
		}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
	}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-5").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(result.Content) == 0 {
		t.Fatal("expected content")
	}
	tc, ok := result.Content[0].(TextContent)
	if !ok {
		t.Fatalf("expected TextContent, got %T", result.Content[0])
	}
	om, ok := tc.ProviderOptions["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected ProviderOptions[openai], got %v", tc.ProviderOptions)
	}
	if om["itemId"] != "msg-99" {
		t.Errorf("itemId = %v", om["itemId"])
	}
	if om["phase"] != "final_answer" {
		t.Errorf("phase = %v", om["phase"])
	}
	ann, ok := om["annotations"].([]map[string]any)
	if !ok || len(ann) == 0 {
		t.Fatalf("expected annotations, got %v", om["annotations"])
	}
}
