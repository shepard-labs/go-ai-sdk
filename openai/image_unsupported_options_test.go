package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestImageAspectRatioWarns verifies that the image model emits a warning
// when aspectRatio is set in provider options (per spec: not supported by
// the OpenAI image API).
func TestImageAspectRatioWarns(t *testing.T) {
	respBody := `{"created":1,"data":[{"b64_json":""}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"aspectRatio": "16:9"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	found := false
	for _, w := range res.Warnings {
		if w.Feature == "aspectRatio" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected aspectRatio warning, got: %+v", res.Warnings)
	}
}

// TestImageSeedWarns verifies that the image model emits a warning when
// seed is set in provider options.
func TestImageSeedWarns(t *testing.T) {
	respBody := `{"created":1,"data":[{"b64_json":""}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"seed": 42},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	found := false
	for _, w := range res.Warnings {
		if w.Feature == "seed" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected seed warning, got: %+v", res.Warnings)
	}
}

// TestImageNoWarningsWhenNotSet verifies that no aspectRatio/seed warnings
// are emitted when those options are absent.
func TestImageNoWarningsWhenNotSet(t *testing.T) {
	respBody := `{"created":1,"data":[{"b64_json":""}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	for _, w := range res.Warnings {
		if w.Feature == "aspectRatio" || w.Feature == "seed" {
			t.Errorf("unexpected warning: %+v", w)
		}
	}
}
