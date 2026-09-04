package openaicompatible

import (
	"context"
	"net/http"
	"testing"
)

func TestImageGenerationParsesURLResponses(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"b64_json":"img-1"},{"url":"https://cdn.test/a.png"},{"b64_json":"img-both","url":"https://cdn.test/ignored.png"},{"url":"https://cdn.test/b.png"}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "draw", N: 4})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	want := []string{"img-1", "https://cdn.test/a.png", "img-both", "https://cdn.test/b.png"}
	if len(result.Images) != len(want) {
		t.Fatalf("images = %#v, want %#v", result.Images, want)
	}
	for i := range want {
		if result.Images[i] != want[i] {
			t.Fatalf("images = %#v, want %#v", result.Images, want)
		}
	}
}

func TestImageGenerationDefaultsB64JSONResponseFormat(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "draw"})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["response_format"] != "b64_json" {
		t.Fatalf("response_format = %#v, want b64_json default", body["response_format"])
	}
}

func TestImageGenerationPassesThroughRequestedResponseFormat(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"url":"https://cdn.test/a.png"}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt:          "draw",
		ProviderOptions: ProviderOptions{"acme": map[string]any{"response_format": "url"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["response_format"] != "url" {
		t.Fatalf("response_format = %#v, want url", body["response_format"])
	}
	if len(result.Images) != 1 || result.Images[0] != "https://cdn.test/a.png" {
		t.Fatalf("images = %#v", result.Images)
	}
}
