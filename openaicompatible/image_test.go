package openaicompatible

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestImageGenerationRequestResponseWarningsAndMetadata(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"b64_json":"img-1"},{"b64_json":"img-2"}]}`)}}
	instant := time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC)
	oldNow := imageNow
	imageNow = func() time.Time { return instant }
	t.Cleanup(func() { imageNow = oldNow })
	seed := 1
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test/v1", Name: "acme-provider", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt:      "draw",
		N:           2,
		Size:        "1024x1024",
		AspectRatio: "1:1",
		Seed:        &seed,
		ProviderOptions: ProviderOptions{
			"acme-provider": map[string]any{"quality": "standard", "style": "old"},
			"acmeProvider":  map[string]any{"style": "vivid"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if f.requests[0].URL.Path != "/v1/images/generations" {
		t.Fatalf("path = %q", f.requests[0].URL.Path)
	}
	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "image-model" || body["prompt"] != "draw" || body["n"].(float64) != 2 || body["response_format"] != "b64_json" || body["size"] != "1024x1024" || body["quality"] != "standard" || body["style"] != "vivid" {
		t.Fatalf("body = %#v", body)
	}
	if result.Request.FormFields != nil {
		t.Fatalf("form fields = %#v", result.Request.FormFields)
	}
	if len(result.Warnings) != 2 || result.Warnings[0].Details != "This model does not support aspect ratio. Use size instead." || result.Warnings[1].Feature != "seed" {
		t.Fatalf("warnings = %#v", result.Warnings)
	}
	if len(result.Images) != 2 || result.Images[0] != "img-1" || result.Images[1] != "img-2" {
		t.Fatalf("images = %#v", result.Images)
	}
	if !result.Response.Timestamp.Equal(instant) || result.Response.ModelID != "image-model" || result.Response.Headers.Get("x-request-id") != "resp-id" {
		t.Fatalf("response = %#v", result.Response)
	}
}

func TestImageGenerationIncludesEmptyPromptAndZeroNOmitsEmptySize(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["prompt"]; !ok || body["prompt"] != "" {
		t.Fatalf("prompt missing/body = %#v", body)
	}
	if _, ok := body["n"]; !ok || body["n"].(float64) != 0 {
		t.Fatalf("n missing/body = %#v", body)
	}
	if _, ok := body["size"]; ok {
		t.Fatalf("size present/body = %#v", body)
	}
}

func TestImageEditsMultipartFieldsFilesAndMetadata(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"b64_json":"edited"}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme-provider", Fetch: f, GenerateID: func() string { return "req-id" }, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit",
		N:      0,
		Size:   "512x512",
		Files: []ImageFile{
			{Type: "bytes", Data: []byte("one"), MediaType: "image/png"},
			{Type: "base64", Base64: base64.StdEncoding.EncodeToString([]byte("two")), MediaType: "image/jpeg"},
		},
		Mask: &ImageFile{Type: "bytes", Data: []byte("mask"), MediaType: "image/png"},
		ProviderOptions: ProviderOptions{
			"acme-provider": map[string]any{"arr": []int{1, 2}, "obj": map[string]any{"a": float64(1)}, "nil": nil, "num": 1.5},
			"acmeProvider":  map[string]any{"flag": true},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	req := f.requests[0]
	if req.URL.Path != "/images/edits" {
		t.Fatalf("path = %q", req.URL.Path)
	}
	if got := req.Header.Get("x-request-id"); got != "req-id" {
		t.Fatalf("x-request-id = %q", got)
	}
	if !strings.HasPrefix(req.Header.Get("Content-Type"), "multipart/form-data") {
		t.Fatalf("content type = %q", req.Header.Get("Content-Type"))
	}
	if len(result.Request.Body) != 0 {
		t.Fatalf("multipart request body metadata retained %d bytes", len(result.Request.Body))
	}
	fields := result.Request.FormFields
	if fields["model"][0] != "image-model" || fields["prompt"][0] != "edit" || fields["n"][0] != "0" || fields["size"][0] != "512x512" || fields["arr"][0] != "[1,2]" || fields["obj"][0] != `{"a":1}` || fields["flag"][0] != "true" || fields["num"][0] != "1.5" {
		t.Fatalf("form fields = %#v", fields)
	}
	if _, ok := fields["nil"]; ok {
		t.Fatalf("nil provider option present: %#v", fields)
	}
	parts := readMultipartParts(t, req)
	if len(parts["image"]) != 2 || parts["image"][0] != "one" || parts["image"][1] != "two" {
		t.Fatalf("image parts = %#v", parts["image"])
	}
	if _, ok := parts["image[]"]; ok {
		t.Fatalf("image[] part present")
	}
	if len(parts["mask"]) != 1 || parts["mask"][0] != "mask" {
		t.Fatalf("mask parts = %#v", parts["mask"])
	}
	if len(result.Images) != 1 || result.Images[0] != "edited" {
		t.Fatalf("images = %#v", result.Images)
	}
}

func TestImageEditOmitsEmptyPrompt(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{Files: []ImageFile{{Type: "bytes", Data: []byte("one")}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := result.Request.FormFields["prompt"]; ok {
		t.Fatalf("prompt present in form fields = %#v", result.Request.FormFields)
	}
}

func TestImageURLDownloadHeadersRetryAndNon2xx(t *testing.T) {
	t.Run("download before upload with retry and user-agent only", func(t *testing.T) {
		f := &recordingFetcher{responses: []*http.Response{
			response(500, `{"error":{"message":"try again"}}`),
			{StatusCode: 200, Header: http.Header{"Content-Type": []string{"image/png"}}, Body: io.NopCloser(strings.NewReader("url-bytes"))},
			response(200, `{"data":[]}`),
		}}
		p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", APIKey: "secret", Headers: http.Header{"X-Provider": []string{"provider"}}, Fetch: f, Retry: &RetryOptions{MaxRetries: 1}})
		_, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{Files: []ImageFile{{Type: "url", URL: "https://cdn.test/image.png"}}})
		if err != nil {
			t.Fatal(err)
		}
		if len(f.requests) != 3 {
			t.Fatalf("requests = %d", len(f.requests))
		}
		for i := 0; i < 2; i++ {
			if f.requests[i].URL.String() != "https://cdn.test/image.png" || f.requests[i].Header.Get("User-Agent") == "" || f.requests[i].Header.Get("Authorization") != "" || f.requests[i].Header.Get("X-Provider") != "" || f.requests[i].Header.Get("x-request-id") != "" {
				t.Fatalf("download request %d headers/url = %s %#v", i, f.requests[i].URL, f.requests[i].Header)
			}
		}
		parts := readMultipartParts(t, f.requests[2])
		if parts["image"][0] != "url-bytes" {
			t.Fatalf("uploaded image = %#v", parts["image"])
		}
	})
	t.Run("non-2xx limited body", func(t *testing.T) {
		f := &recordingFetcher{responses: []*http.Response{{StatusCode: 400, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("abcdef"))}}}
		p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, MaxErrorResponseBytes: 3})
		_, err := p.Image("image-model").DoGenerate(context.Background(), ImageGenerateOptions{Files: []ImageFile{{Type: "url", URL: "https://cdn.test/image.png"}}})
		apiErr := new(APICallError)
		if !errors.As(err, &apiErr) || string(apiErr.Body) != "abc" || !apiErr.Truncated {
			t.Fatalf("error = %#v", err)
		}
	})
}

func readMultipartParts(t *testing.T, req *http.Request) map[string][]string {
	t.Helper()
	mediaType := req.Header.Get("Content-Type")
	idx := strings.Index(mediaType, "boundary=")
	if idx < 0 {
		t.Fatalf("missing boundary in %q", mediaType)
	}
	reader := multipart.NewReader(req.Body, mediaType[idx+len("boundary="):])
	parts := map[string][]string{}
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Fatal(err)
		}
		parts[part.FormName()] = append(parts[part.FormName()], string(data))
	}
	return parts
}
