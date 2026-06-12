package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestCompletionForwardsProviderOptions verifies that completion model
// forwards provider-specific options: echo, logit_bias, suffix, user.
func TestCompletionForwardsProviderOptions(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{
				"echo":      true,
				"logitBias": map[string]any{"50256": -100},
				"suffix":    "END",
				"user":      "u-1",
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["echo"] != true {
		t.Errorf("echo: %v", body["echo"])
	}
	lb, ok := body["logit_bias"].(map[string]any)
	if !ok {
		t.Errorf("logit_bias: %v", body["logit_bias"])
	} else if v, _ := lb["50256"].(float64); v != -100 {
		t.Errorf("logit_bias[50256]: %v", lb["50256"])
	}
	if body["suffix"] != "END" {
		t.Errorf("suffix: %v", body["suffix"])
	}
	if body["user"] != "u-1" {
		t.Errorf("user: %v", body["user"])
	}
}

// TestCompletionDoesNotIncludeProviderOptionsWhenAbsent verifies that
// echo, suffix, user, logit_bias are absent from the body when not set.
func TestCompletionDoesNotIncludeProviderOptionsWhenAbsent(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	for _, key := range []string{"echo", "suffix", "user", "logit_bias"} {
		if _, has := body[key]; has {
			t.Errorf("%q should not be in body: %v", key, body[key])
		}
	}
}
