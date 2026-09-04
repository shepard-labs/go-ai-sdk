package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

func TestImageGenerationsSendsB64JSONByDefault(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[]}`)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "x"})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["response_format"] != "b64_json" {
		t.Errorf("response_format = %v, want b64_json default: %#v", body["response_format"], body)
	}
}

func TestImageGenerationsSendsRequestedResponseFormat(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"url":"https://cdn.test/a.png"}]}`)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt:          "x",
		ProviderOptions: ProviderOptions{"openai": map[string]any{"responseFormat": "url"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["response_format"] != "url" {
		t.Errorf("response_format = %v, want url", body["response_format"])
	}
	if len(result.Images) != 1 || result.Images[0] != "https://cdn.test/a.png" {
		t.Errorf("images = %#v, want the url response", result.Images)
	}
}

func TestImageGenerationsIgnoresRequestedResponseFormatForDefaultFormatModels(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[]}`)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt:          "x",
		ProviderOptions: ProviderOptions{"openai": map[string]any{"responseFormat": "url"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, ok := body["response_format"]; ok {
		t.Errorf("response_format should be omitted for gpt-image-1: %#v", body)
	}
}

func TestImageGenerationsParsesURLResponses(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"url":"https://cdn.test/a.png"},{"b64_json":"aGk="}]}`)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Image("dall-e-2").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "x"})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(result.Images) != 2 || result.Images[0] != "https://cdn.test/a.png" || result.Images[1] != "hi" {
		t.Errorf("images = %#v, want [url hi]", result.Images)
	}
}

func TestImageEditsSendsRequestedResponseFormat(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("dall-e-2").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt:          "edit me",
		Files:           []ImageFile{{Type: "bytes", Data: []byte("fakepng"), MediaType: "image/png"}},
		ProviderOptions: ProviderOptions{"openai": map[string]any{"responseFormat": "url"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if !strings.Contains(f.captured, "response_format") || !strings.Contains(f.captured, "url") {
		t.Errorf("requested response_format missing from edit body: %q", f.captured)
	}
}
