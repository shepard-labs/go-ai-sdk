package openai

import (
	"bytes"
	"context"
	"net/http"
	"strings"
	"testing"
)

func TestImageEditInputFidelityPropagates(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit me",
		Files:  []ImageFile{{Type: "bytes", Data: []byte("fakepng"), MediaType: "image/png"}},
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"inputFidelity": "high"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if !strings.Contains(f.captured, "input_fidelity") {
		t.Errorf("input_fidelity missing: %q", f.captured)
	}
	if !strings.Contains(f.captured, "high") {
		t.Errorf("input_fidelity value missing: %q", f.captured)
	}
}

func TestImageEditInputFidelityRejectsNonGptImage(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"b64_json":""}]}`)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("dall-e-2").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit me",
		Files:  []ImageFile{{Type: "bytes", Data: []byte("fakepng"), MediaType: "image/png"}},
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"inputFidelity": "high"},
		},
	})
	if err == nil {
		t.Fatal("expected error for dall-e-2 inputFidelity")
	}
	if !strings.Contains(err.Error(), "inputFidelity") {
		t.Errorf("error should mention inputFidelity: %v", err)
	}
}

// TestBuildImageRequestForGPTImageOmitsResponseFormat verifies that
// gpt-image-1 doesn't send the response_format field (it always returns b64_json).
func TestBuildImageRequestForGPTImageOmitsResponseFormat(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	// The body sent is the JSON, captured in f.captured for json path
	// (but this is for image generations, not edits, so the body is JSON).
	// Check that response_format is not in the body.
	if strings.Contains(f.captured, "response_format") {
		t.Errorf("response_format should not be in gpt-image-1 body: %q", f.captured)
	}
}

func TestBuildImageRequestForDallE3IncludesResponseFormat(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if !strings.Contains(f.captured, "response_format") {
		t.Errorf("response_format should be in dall-e-3 body: %q", f.captured)
	}
}

// keep imports alive
var _ = http.MethodGet
var _ = bytes.NewReader
