package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ---- Test fixtures ----

// geminiImageResponseBody is a fake :generateContent response for the
// Gemini image path. The LanguageModel parses the response and converts
// inline-data parts into ImageContent (per chat.go's parseGenerateResponse).
const geminiImageResponseBody = `{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          { "inlineData": { "mimeType": "image/png", "data": "BASE64-OF-IMAGE" } }
        ]
      },
      "finishReason": "STOP"
    }
  ],
  "modelVersion": "models/gemini-2.5-flash-image",
  "responseId": "resp-123",
  "usageMetadata": {
    "promptTokenCount": 5,
    "candidatesTokenCount": 10,
    "totalTokenCount": 15
  }
}`

// imagenResponseBody is a fake :predict response for Imagen.
const imagenResponseBody = `{
  "predictions": [
    { "bytesBase64Encoded": "AAA" },
    { "bytesBase64Encoded": "BBB" }
  ]
}`

// ---- Gemini image path tests ----

// TestImage_GeminiPath_DelegatesToLanguageModel verifies the Gemini image
// path: DoGenerate posts to the :generateContent endpoint with
// responseModalities: ["IMAGE"] in the body and the right model path.
func TestImage_GeminiPath_DelegatesToLanguageModel(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, geminiImageResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	im := p.ImageModel("gemini-2.5-flash-image")
	if im.Provider() != defaultProviderName+".image" {
		t.Fatalf("Provider() = %q", im.Provider())
	}
	if im.ModelID() != "gemini-2.5-flash-image" {
		t.Fatalf("ModelID() = %q", im.ModelID())
	}

	result, err := im.DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt:      "draw a cat",
		AspectRatio: "1:1",
		Headers:     http.Header{"X-Call": []string{"yes"}},
		ProviderOptions: ProviderOptions{
			"google": map[string]any{"personGeneration": "dont_allow"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if len(f.requests) != 1 {
		t.Fatalf("requests = %d", len(f.requests))
	}
	req := f.requests[0]
	if req.Method != http.MethodPost {
		t.Fatalf("method = %q", req.Method)
	}
	if req.URL.Path != "/v1beta/models/gemini-2.5-flash-image:generateContent" {
		t.Fatalf("path = %q", req.URL.Path)
	}
	if got := req.Header.Get("X-Call"); got != "yes" {
		t.Fatalf("X-Call = %q", got)
	}
	if got := req.Header.Get("x-goog-api-key"); got != "secret" {
		t.Fatalf("x-goog-api-key = %q", got)
	}

	// Inspect the request body.
	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	gen, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("body.generationConfig = %T (%v)", body["generationConfig"], body["generationConfig"])
	}
	rm, ok := gen["responseModalities"].([]any)
	if !ok || len(rm) != 1 || rm[0] != "IMAGE" {
		t.Fatalf("responseModalities = %v", gen["responseModalities"])
	}
	ic, ok := gen["imageConfig"].(map[string]any)
	if !ok {
		t.Fatalf("imageConfig = %T (%v)", gen["imageConfig"], gen["imageConfig"])
	}
	if ic["aspectRatio"] != "1:1" {
		t.Fatalf("imageConfig.aspectRatio = %v", ic["aspectRatio"])
	}

	// passthrough google.personGeneration must NOT be in the body (it was
	// stripped by the image-model recognized-keys filter).
	if p, ok := body["personGeneration"]; ok {
		t.Fatalf("body.personGeneration = %v (should be stripped)", p)
	}

	// Response images.
	if len(result.Images) != 1 || result.Images[0] != "BASE64-OF-IMAGE" {
		t.Fatalf("Images = %v", result.Images)
	}
	// Provider metadata: google.images length 1.
	gm, ok := result.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatalf("ProviderMetadata.google = %T", result.ProviderMetadata["google"])
	}
	imagesMeta, ok := gm["images"].([]map[string]any)
	if !ok || len(imagesMeta) != 1 {
		t.Fatalf("ProviderMetadata.google.images = %v", gm["images"])
	}
	if imagesMeta[0]["mimeType"] != "image/png" {
		t.Fatalf("ProviderMetadata.google.images[0].mimeType = %v", imagesMeta[0]["mimeType"])
	}
	// Usage from the language model must be propagated.
	if result.Response.Headers.Get("x-request-id") != "resp-id" {
		t.Fatalf("Response.Headers x-request-id = %q", result.Response.Headers.Get("x-request-id"))
	}
}

// TestImage_GeminiPath_RejectsMask asserts Mask is rejected.
func TestImage_GeminiPath_RejectsMask(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("gemini-2.5-flash-image").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit",
		Mask:   &ImageFile{Type: "data", Data: []byte{0xff}, MediaType: "image/png"},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var uf UnsupportedFunctionalityError
	if !errors.As(err, &uf) {
		t.Fatalf("err = %T (%v), want UnsupportedFunctionalityError", err, err)
	}
	if !strings.Contains(uf.Error(), "mask") {
		t.Errorf("err message = %q, want it to mention mask", uf.Error())
	}
	if len(f.requests) != 0 {
		t.Fatalf("expected 0 requests, got %d", len(f.requests))
	}
}

// TestImage_GeminiPath_RejectsNGreaterThanOne asserts N > 1 is rejected.
func TestImage_GeminiPath_RejectsNGreaterThanOne(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("gemini-2.5-flash-image").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "many",
		N:      2,
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var uf UnsupportedFunctionalityError
	if !errors.As(err, &uf) {
		t.Fatalf("err = %T (%v), want UnsupportedFunctionalityError", err, err)
	}
}

// TestImage_GeminiPath_WarnsOnSize asserts Size produces an unsupported warning.
func TestImage_GeminiPath_WarnsOnSize(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, geminiImageResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.ImageModel("gemini-2.5-flash-image").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "draw",
		Size:   "1024x1024",
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if !hasWarning(result.Warnings, "unsupported", "size") {
		t.Fatalf("warnings = %v, want one with Type=unsupported Feature=size", result.Warnings)
	}
}

// TestImage_GeminiPath_GoogleSearchToolAdded asserts that when the
// googleSearch provider option is set, the LM call receives a Tool entry
// with the right id.
func TestImage_GeminiPath_GoogleSearchToolAdded(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, geminiImageResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("gemini-2.5-flash-image").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "draw a cat",
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"googleSearch": map[string]any{
					"searchTypes": map[string]any{"webSearch": map[string]any{}},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	// The tools[] entry is added by the language model via prepareTools; we
	// just need to confirm the request body contains a tool entry whose
	// key is googleSearch. Re-read the recorded body bytes directly.
	bodyBytes := readRequestBody(f.requests[0])
	var body map[string]any
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		t.Fatal(err)
	}
	tools, ok := body["tools"].([]any)
	if !ok || len(tools) == 0 {
		t.Fatalf("body.tools = %v (want non-empty)", body["tools"])
	}
	first, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("body.tools[0] = %T", tools[0])
	}
	if _, ok := first["googleSearch"]; !ok {
		t.Fatalf("body.tools[0] = %v (want googleSearch key)", first)
	}
}

// TestImage_GeminiPath_ImageFileInputs verifies the user prompt contains
// image file parts (URL with mediaType "image/*", data with base64 inline).
func TestImage_GeminiPath_ImageFileInputs(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, geminiImageResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	payload := []byte("png-bytes")
	b64 := base64.StdEncoding.EncodeToString(payload)
	_, err := p.ImageModel("gemini-2.5-flash-image").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit",
		Files: []ImageFile{
			{Type: "url", URL: "https://cdn.test/source.png", MediaType: "image/png"},
			{Type: "data", Data: payload, MediaType: "image/jpeg"},
			{Type: "base64", Base64: b64, MediaType: "image/png"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	bodyBytes := readRequestBody(f.requests[0])
	var body map[string]any
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		t.Fatal(err)
	}
	contents, ok := body["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("body.contents = %v", body["contents"])
	}
	parts := contents[0].(map[string]any)["parts"].([]any)
	// 1 text + 3 image parts.
	if len(parts) != 4 {
		t.Fatalf("parts = %d, want 4: %v", len(parts), parts)
	}
	if parts[0].(map[string]any)["text"] != "edit" {
		t.Errorf("parts[0].text = %v", parts[0])
	}
	// URL file -> fileData.
	urlPart := parts[1].(map[string]any)
	fd, ok := urlPart["fileData"].(map[string]any)
	if !ok {
		t.Fatalf("parts[1] = %v (want fileData)", urlPart)
	}
	if fd["fileUri"] != "https://cdn.test/source.png" {
		t.Errorf("fileUri = %v", fd["fileUri"])
	}
	if fd["mimeType"] != "image/png" {
		t.Errorf("mimeType = %v", fd["mimeType"])
	}
	// data file -> inlineData with base64 of the bytes.
	dataPart := parts[2].(map[string]any)
	id, ok := dataPart["inlineData"].(map[string]any)
	if !ok {
		t.Fatalf("parts[2] = %v (want inlineData)", dataPart)
	}
	if id["data"] != b64 {
		t.Errorf("data = %v, want %v", id["data"], b64)
	}
	if id["mimeType"] != "image/jpeg" {
		t.Errorf("mimeType = %v", id["mimeType"])
	}
}

// TestImage_GeminiPath_SkipsNonImageParts asserts that non-image inlineData
// parts in the language model response are NOT returned in the image list.
func TestImage_GeminiPath_SkipsNonImageParts(t *testing.T) {
	body := `{
	  "candidates": [
	    {
	      "content": {
	        "role": "model",
	        "parts": [
		  { "text": "here is your image" },
		  { "inlineData": { "mimeType": "image/png", "data": "PNG" } },
		  { "inlineData": { "mimeType": "application/json", "data": "JSON" } }
		]
	      },
	      "finishReason": "STOP"
	    }
	  ]
	}`
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, body)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.ImageModel("gemini-2.5-flash-image").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "draw",
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if len(result.Images) != 1 || result.Images[0] != "PNG" {
		t.Fatalf("Images = %v", result.Images)
	}
}

// ---- Imagen path tests ----

// TestImage_ImagenPath_PredictEndpoint verifies the Imagen path: DoGenerate
// posts to :predict with the instances/parameters body shape.
func TestImage_ImagenPath_PredictEndpoint(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, imagenResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.ImageModel("imagen-4.0-generate-001").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt:      "draw a dog",
		N:           2,
		AspectRatio: "16:9",
		Headers:     http.Header{"X-Call": []string{"yes"}},
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"personGeneration": "dont_allow",
				"addWatermark":     true,
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if len(f.requests) != 1 {
		t.Fatalf("requests = %d", len(f.requests))
	}
	req := f.requests[0]
	if req.Method != http.MethodPost {
		t.Fatalf("method = %q", req.Method)
	}
	if req.URL.Path != "/v1beta/models/imagen-4.0-generate-001:predict" {
		t.Fatalf("path = %q", req.URL.Path)
	}
	if got := req.Header.Get("X-Call"); got != "yes" {
		t.Fatalf("X-Call = %q", got)
	}

	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	instances, ok := body["instances"].([]any)
	if !ok || len(instances) != 1 {
		t.Fatalf("body.instances = %v", body["instances"])
	}
	inst := instances[0].(map[string]any)
	if inst["prompt"] != "draw a dog" {
		t.Errorf("instances[0].prompt = %v", inst["prompt"])
	}
	params, ok := body["parameters"].(map[string]any)
	if !ok {
		t.Fatalf("body.parameters = %T (%v)", body["parameters"], body["parameters"])
	}
	if params["sampleCount"].(float64) != 2 {
		t.Errorf("sampleCount = %v", params["sampleCount"])
	}
	if params["aspectRatio"] != "16:9" {
		t.Errorf("aspectRatio = %v", params["aspectRatio"])
	}
	if params["personGeneration"] != "dont_allow" {
		t.Errorf("personGeneration = %v", params["personGeneration"])
	}
	if params["addWatermark"] != true {
		t.Errorf("addWatermark = %v", params["addWatermark"])
	}

	if len(result.Images) != 2 || result.Images[0] != "AAA" || result.Images[1] != "BBB" {
		t.Fatalf("Images = %v", result.Images)
	}
	gm, ok := result.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatalf("ProviderMetadata.google = %T", result.ProviderMetadata["google"])
	}
	imagesMeta, ok := gm["images"].([]map[string]any)
	if !ok || len(imagesMeta) != 2 {
		t.Fatalf("ProviderMetadata.google.images = %v", gm["images"])
	}
}

// TestImage_ImagenPath_RejectsFiles asserts Files is rejected for Imagen.
func TestImage_ImagenPath_RejectsFiles(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("imagen-4.0-generate-001").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit",
		Files:  []ImageFile{{Type: "url", URL: "https://cdn.test/img.png"}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var uf UnsupportedFunctionalityError
	if !errors.As(err, &uf) {
		t.Fatalf("err = %T (%v), want UnsupportedFunctionalityError", err, err)
	}
	if !strings.Contains(uf.Error(), "files") && !strings.Contains(uf.Error(), "editing") {
		t.Errorf("err message = %q", uf.Error())
	}
}

// TestImage_ImagenPath_RejectsMask asserts Mask is rejected for Imagen.
func TestImage_ImagenPath_RejectsMask(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("imagen-4.0-generate-001").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit",
		Mask:   &ImageFile{Type: "data", Data: []byte{0xff}, MediaType: "image/png"},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var uf UnsupportedFunctionalityError
	if !errors.As(err, &uf) {
		t.Fatalf("err = %T (%v), want UnsupportedFunctionalityError", err, err)
	}
}

// TestImage_ImagenPath_WarnsOnSizeAndSeed asserts size and seed both produce
// unsupported warnings.
func TestImage_ImagenPath_WarnsOnSizeAndSeed(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, imagenResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	seed := 7
	result, err := p.ImageModel("imagen-4.0-generate-001").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "draw",
		Size:   "1024x1024",
		Seed:   &seed,
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if !hasWarning(result.Warnings, "unsupported", "size") {
		t.Errorf("missing size warning: %v", result.Warnings)
	}
	if !hasWarning(result.Warnings, "unsupported", "seed") {
		t.Errorf("missing seed warning: %v", result.Warnings)
	}
}

// TestImage_ImagenPath_GoogleSearchStrippedAndWarned asserts googleSearch is
// warned and stripped on the Imagen path.
func TestImage_ImagenPath_GoogleSearchStrippedAndWarned(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, imagenResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.ImageModel("imagen-4.0-generate-001").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "draw",
		ProviderOptions: ProviderOptions{
			"google": map[string]any{"googleSearch": map[string]any{}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if !hasWarning(result.Warnings, "unsupported", "googleSearch") {
		t.Fatalf("warnings = %v, want one with Type=unsupported Feature=googleSearch", result.Warnings)
	}
	// googleSearch must NOT appear in the parameters object.
	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	if raw, ok := body["parameters"]; ok {
		if params, ok := raw.(map[string]any); ok {
			if _, ok := params["googleSearch"]; ok {
				t.Errorf("parameters.googleSearch present (should be stripped): %v", params)
			}
		}
	}
}

// ---- Dispatch / MaxImagesPerCall tests ----

// TestImage_IsGeminiImageModel covers the predicate that drives dispatch.
func TestImage_IsGeminiImageModel(t *testing.T) {
	cases := []struct {
		id   string
		want bool
	}{
		{"gemini-2.5-flash-image", true},
		{"gemini-2.5-flash-image-preview", true},
		{"nano-banana-pro-preview", true},
		{"nano-banana", true},
		{"imagen-4.0-generate-001", false},
		{"imagen-4.0-ultra-generate-001", false},
		{"imagen-4.0-fast-generate-001", false},
		{"gemini-2.5-pro", false},
		{"gemini-3-pro-image-preview", true},
	}
	for _, c := range cases {
		if got := isGeminiImageModel(c.id); got != c.want {
			t.Errorf("isGeminiImageModel(%q) = %v, want %v", c.id, got, c.want)
		}
	}
}

// TestImage_MaxImagesPerCall covers the family defaults and the override.
func TestImage_MaxImagesPerCall(t *testing.T) {
	p := CreateGoogle(ProviderSettings{BaseURL: "https://example.test/v1beta", APIKey: "secret"})
	cases := []struct {
		id   string
		want int
	}{
		{"gemini-2.5-flash-image", 10},
		{"nano-banana-pro-preview", 10},
		{"imagen-4.0-generate-001", 4},
		{"imagen-4.0-ultra-generate-001", 4},
	}
	for _, c := range cases {
		got := p.ImageModel(c.id).MaxImagesPerCall()
		if got != c.want {
			t.Errorf("ImageModel(%q).MaxImagesPerCall() = %d, want %d", c.id, got, c.want)
		}
	}
	// Override.
	override := 2
	if got := p.ImageModel("gemini-2.5-flash-image", ImageModelSettings{MaxImagesPerCall: &override}).MaxImagesPerCall(); got != 2 {
		t.Errorf("override MaxImagesPerCall = %d, want 2", got)
	}
}

// TestImage_DispatchGemniniToChat tests that a non-streaming Gemini image
// call flows through the chat path (URL ends with :generateContent).
func TestImage_DispatchGemniniToChat(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, geminiImageResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("gemini-2.5-flash-image").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "draw"})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if !strings.HasSuffix(f.requests[0].URL.Path, ":generateContent") {
		t.Errorf("path = %q (want :generateContent suffix)", f.requests[0].URL.Path)
	}
}

// TestImage_DispatchImagenToPredict tests that an Imagen call flows through
// the :predict endpoint.
func TestImage_DispatchImagenToPredict(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, imagenResponseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("imagen-4.0-generate-001").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "draw"})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if !strings.HasSuffix(f.requests[0].URL.Path, ":predict") {
		t.Errorf("path = %q (want :predict suffix)", f.requests[0].URL.Path)
	}
}

// TestImage_ProviderErrorPropagated asserts that the HTTP error path uses
// the same APICallError machinery as the rest of the provider.
func TestImage_ProviderErrorPropagated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"code":500,"message":"kaboom","status":"INTERNAL"}}`))
	}))
	t.Cleanup(srv.Close)
	p := CreateGoogle(ProviderSettings{
		BaseURL: srv.URL + "/v1beta",
		APIKey:  "secret",
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.ImageModel("imagen-4.0-generate-001").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "draw"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %T, want *APICallError", err)
	}
	if apiErr.Status != http.StatusInternalServerError {
		t.Errorf("Status = %d", apiErr.Status)
	}
}

// ---- helpers ----

func hasWarning(warnings []Warning, wantType, wantFeature string) bool {
	for _, w := range warnings {
		if w.Type == wantType && w.Feature == wantFeature {
			return true
		}
	}
	return false
}
