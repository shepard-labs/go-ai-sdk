package google

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// ---- Test helpers ----

type recordingFetcher struct {
	mu        sync.Mutex
	requests  []*http.Request
	responses []*http.Response
	errors    []error
	calls     int
}

func (f *recordingFetcher) Do(req *http.Request) (*http.Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, cloneRequest(req))
	idx := f.calls
	f.calls++
	var err error
	if idx < len(f.errors) {
		err = f.errors[idx]
	}
	var resp *http.Response
	if idx < len(f.responses) {
		resp = f.responses[idx]
	}
	return resp, err
}

func cloneRequest(req *http.Request) *http.Request {
	clone := req.Clone(req.Context())
	clone.Header = req.Header.Clone()
	if req.Body != nil {
		body, _ := io.ReadAll(req.Body)
		_ = req.Body.Close()
		req.Body = io.NopCloser(strings.NewReader(string(body)))
		clone.Body = io.NopCloser(strings.NewReader(string(body)))
	}
	return clone
}

func testResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"X-Request-Id": []string{"resp-id"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// readRequestBody reads the full body of a recorded request, handling the
// io.ReadCloser → []byte plumbing that the recordingFetcher does not
// surface directly.
func readRequestBody(req *http.Request) []byte {
	if req == nil || req.Body == nil {
		return nil
	}
	body, _ := io.ReadAll(req.Body)
	_ = req.Body.Close()
	return body
}

// ---- Tests ----

// TestEmbeddingDoEmbedSingle verifies the single-value :embedContent path: a
// single value dispatches to :embedContent, the body and URL are correct,
// and the response is decoded into EmbedResult.Embeddings.
func TestEmbeddingDoEmbedSingle(t *testing.T) {
	responseBody := `{"embedding":{"values":[0.1,0.2,0.3]}}`
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, responseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL:    "https://example.test/v1beta",
		APIKey:     "secret",
		Fetch:      f,
		GenerateID: func() string { return "req-id" },
		Retry:      &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	if em.Provider() != defaultProviderName+".embedding" {
		t.Fatalf("Provider() = %q", em.Provider())
	}
	if em.ModelID() != "gemini-embedding-001" {
		t.Fatalf("ModelID() = %q", em.ModelID())
	}

	result, err := em.DoEmbed(context.Background(), EmbedOptions{
		Values:  []string{"hello"},
		Headers: http.Header{"X-Call": []string{"yes"}},
	})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}
	if got := len(f.requests); got != 1 {
		t.Fatalf("requests = %d", got)
	}
	req := f.requests[0]
	if req.Method != http.MethodPost {
		t.Fatalf("method = %q", req.Method)
	}
	if req.URL.Path != "/v1beta/models/gemini-embedding-001:embedContent" {
		t.Fatalf("path = %q", req.URL.Path)
	}
	if got := req.Header.Get("X-Call"); got != "yes" {
		t.Fatalf("X-Call = %q", got)
	}
	if got := req.Header.Get("x-request-id"); got != "req-id" {
		t.Fatalf("x-request-id = %q", got)
	}
	if got := req.Header.Get("x-goog-api-key"); got != "secret" {
		t.Fatalf("x-goog-api-key = %q", got)
	}

	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "models/gemini-embedding-001" {
		t.Errorf("body.model = %v", body["model"])
	}
	content, ok := body["content"].(map[string]any)
	if !ok {
		t.Fatalf("body.content = %T", body["content"])
	}
	parts, ok := content["parts"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("body.content.parts = %v", body["content"])
	}
	if parts[0].(map[string]any)["text"] != "hello" {
		t.Errorf("body.content.parts[0].text = %v", parts[0])
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 || result.Embeddings[0][0] != 0.1 || result.Embeddings[0][2] != 0.3 {
		t.Fatalf("Embeddings[0] = %v", result.Embeddings[0])
	}
	if result.Usage != nil {
		t.Errorf("Usage = %v, want nil", result.Usage)
	}
	if string(result.Response.Body) != responseBody {
		t.Errorf("Response.Body = %q", result.Response.Body)
	}
	if result.Response.Headers.Get("x-request-id") != "resp-id" {
		t.Errorf("Response.Headers x-request-id = %q", result.Response.Headers.Get("x-request-id"))
	}
}

// TestEmbeddingDoEmbedBatch verifies the multi-value :batchEmbedContents
// path: two or more values dispatch to the batch endpoint with a {requests:[...]}
// body, and the response is decoded into a matching number of embeddings.
func TestEmbeddingDoEmbedBatch(t *testing.T) {
	responseBody := `{"embeddings":[{"values":[1,1]},{"values":[2,2]}]}`
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, responseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "secret",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")

	result, err := em.DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"alpha", "beta"},
	})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}
	req := f.requests[0]
	if req.URL.Path != "/v1beta/models/gemini-embedding-001:batchEmbedContents" {
		t.Fatalf("path = %q", req.URL.Path)
	}

	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	requests, ok := body["requests"].([]any)
	if !ok || len(requests) != 2 {
		t.Fatalf("body.requests = %v", body["requests"])
	}
	for i, r := range requests {
		rMap, ok := r.(map[string]any)
		if !ok {
			t.Fatalf("requests[%d] is %T", i, r)
		}
		if rMap["model"] != "models/gemini-embedding-001" {
			t.Errorf("requests[%d].model = %v", i, rMap["model"])
		}
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 2 || result.Embeddings[0][0] != 1 {
		t.Fatalf("Embeddings[0] = %v", result.Embeddings[0])
	}
	if len(result.Embeddings[1]) != 2 || result.Embeddings[1][0] != 2 {
		t.Fatalf("Embeddings[1] = %v", result.Embeddings[1])
	}
}

// TestEmbeddingTooManyValuesForCall verifies that exceeding
// MaxEmbeddingsPerCall returns TooManyEmbeddingValuesForCallError.
func TestEmbeddingTooManyValuesForCall(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "k",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")

	values := make([]string, defaultMaxEmbeddingsPerCall+1)
	_, err := em.DoEmbed(context.Background(), EmbedOptions{Values: values})
	var tm TooManyEmbeddingValuesForCallError
	if !errors.As(err, &tm) {
		t.Fatalf("err = %T (%v), want TooManyEmbeddingValuesForCallError", err, err)
	}
	if tm.Provider != defaultProviderName+".embedding" {
		t.Errorf("tm.Provider = %q", tm.Provider)
	}
	if tm.ModelID != "gemini-embedding-001" {
		t.Errorf("tm.ModelID = %q", tm.ModelID)
	}
	if tm.MaxEmbeddingsPerCall != defaultMaxEmbeddingsPerCall {
		t.Errorf("tm.MaxEmbeddingsPerCall = %d", tm.MaxEmbeddingsPerCall)
	}
	if tm.Values != len(values) {
		t.Errorf("tm.Values = %d", tm.Values)
	}
	if got := f.calls; got != 0 {
		t.Errorf("HTTP calls = %d, want 0 (validation must precede request)", got)
	}
}

// TestEmbedding_MaxEmbeddingsPerCall verifies the default and override.
func TestEmbedding_MaxEmbeddingsPerCall(t *testing.T) {
	p := CreateGoogle(ProviderSettings{APIKey: "k"})
	em := p.EmbeddingModel("gemini-embedding-001")
	if em.MaxEmbeddingsPerCall() != defaultMaxEmbeddingsPerCall {
		t.Errorf("default MaxEmbeddingsPerCall = %d, want %d", em.MaxEmbeddingsPerCall(), defaultMaxEmbeddingsPerCall)
	}
	if !em.SupportsParallelCalls() {
		t.Error("default SupportsParallelCalls = false, want true")
	}

	parallel := false
	p = CreateGoogle(ProviderSettings{
		APIKey:                         "k",
		MaxEmbeddingsPerCall:           4,
		SupportsParallelEmbeddingCalls: &parallel,
	})
	em = p.EmbeddingModel("gemini-embedding-001")
	if em.MaxEmbeddingsPerCall() != 4 {
		t.Errorf("override MaxEmbeddingsPerCall = %d, want 4", em.MaxEmbeddingsPerCall())
	}
	if em.SupportsParallelCalls() {
		t.Error("override SupportsParallelCalls = true, want false")
	}
}

// TestEmbedding_TooManyValuesForCall_Override verifies the override cap is
// respected.
func TestEmbedding_TooManyValuesForCall_Override(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateGoogle(ProviderSettings{
		APIKey:               "k",
		MaxEmbeddingsPerCall: 2,
		Fetch:                f,
		Retry:                &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	_, err := em.DoEmbed(context.Background(), EmbedOptions{Values: []string{"a", "b", "c"}})
	var tm TooManyEmbeddingValuesForCallError
	if !errors.As(err, &tm) {
		t.Fatalf("err = %T (%v)", err, err)
	}
	if tm.MaxEmbeddingsPerCall != 2 || tm.Values != 3 {
		t.Errorf("tm = %+v", tm)
	}
}

// TestEmbeddingMultimodalContent verifies that per-value content parts from
// providerOptions.google.content are appended after the text value, and that
// the lengths must match.
func TestEmbeddingMultimodalContent(t *testing.T) {
	responseBody := `{"embedding":{"values":[1]}}`
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, responseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "k",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	result, err := em.DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"describe this image"},
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"content": []any{
					[]any{
						map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": "AAA="}},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	content := body["content"].(map[string]any)
	parts := content["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts length = %d, want 2", len(parts))
	}
	first := parts[0].(map[string]any)
	if first["text"] != "describe this image" {
		t.Errorf("parts[0].text = %v", first["text"])
	}
	second := parts[1].(map[string]any)
	id, ok := second["inlineData"].(map[string]any)
	if !ok {
		t.Fatalf("parts[1].inlineData is %T", second["inlineData"])
	}
	if id["mimeType"] != "image/png" || id["data"] != "AAA=" {
		t.Errorf("parts[1].inlineData = %v", id)
	}
}

// TestEmbeddingBatchMultimodalContent verifies per-value content in the
// batch path: only the second value has extra parts.
func TestEmbeddingBatchMultimodalContent(t *testing.T) {
	responseBody := `{"embeddings":[{"values":[1]},{"values":[2]}]}`
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, responseBody)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "k",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	_, err := em.DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"alpha", "beta"},
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"content": []any{
					nil, // text-only
					[]any{map[string]any{"fileData": map[string]any{"mimeType": "application/pdf", "fileUri": "files/abc"}}},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(readRequestBody(f.requests[0]), &body); err != nil {
		t.Fatal(err)
	}
	requests := body["requests"].([]any)
	if len(requests) != 2 {
		t.Fatalf("requests length = %d", len(requests))
	}
	first := requests[0].(map[string]any)["content"].(map[string]any)
	if len(first["parts"].([]any)) != 1 {
		t.Errorf("first value should have 1 part, got %v", first["parts"])
	}
	second := requests[1].(map[string]any)["content"].(map[string]any)
	secondParts := second["parts"].([]any)
	if len(secondParts) != 2 {
		t.Fatalf("second value should have 2 parts, got %v", secondParts)
	}
	fd, ok := secondParts[1].(map[string]any)["fileData"].(map[string]any)
	if !ok || fd["fileUri"] != "files/abc" || fd["mimeType"] != "application/pdf" {
		t.Errorf("secondParts[1].fileData = %v", secondParts[1])
	}
}

// TestEmbeddingContentLengthMismatch verifies the validation error returned
// when providerOptions.google.content length doesn't match values length.
func TestEmbeddingContentLengthMismatch(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "k",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	_, err := em.DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a", "b", "c"},
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"content": []any{nil, nil}, // length 2 != 3
			},
		},
	})
	var ip InvalidPromptError
	if !errors.As(err, &ip) {
		t.Fatalf("err = %T (%v), want InvalidPromptError", err, err)
	}
	if got := f.calls; got != 0 {
		t.Errorf("HTTP calls = %d, want 0", got)
	}
}

// TestEmbeddingOutputDimensionalityAndTaskType verifies that recognized
// EmbeddingModelOptions fields are emitted on the wire.
func TestEmbeddingOutputDimensionalityAndTaskType(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, `{"embedding":{"values":[1]}}`)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "k",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	dim := 768
	_, err := em.DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"x"},
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"outputDimensionality": dim,
				"taskType":             "SEMANTIC_SIMILARITY",
			},
		},
	})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(readRequestBody(f.requests[0]), &body); err != nil {
		t.Fatal(err)
	}
	if body["outputDimensionality"].(float64) != 768 {
		t.Errorf("outputDimensionality = %v", body["outputDimensionality"])
	}
	if body["taskType"] != "SEMANTIC_SIMILARITY" {
		t.Errorf("taskType = %v", body["taskType"])
	}
}

// TestEmbeddingBatchWithOutputDimensionality verifies the same fields appear
// in the batch path.
func TestEmbeddingBatchWithOutputDimensionality(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, `{"embeddings":[{"values":[1]},{"values":[2]}]}`)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "k",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	dim := 256
	_, err := em.DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a", "b"},
		ProviderOptions: ProviderOptions{
			"google": map[string]any{"outputDimensionality": dim},
		},
	})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(readRequestBody(f.requests[0]), &body); err != nil {
		t.Fatal(err)
	}
	requests := body["requests"].([]any)
	for i, r := range requests {
		rMap := r.(map[string]any)
		if rMap["outputDimensionality"].(float64) != 256 {
			t.Errorf("requests[%d].outputDimensionality = %v", i, rMap["outputDimensionality"])
		}
	}
}

// TestEmbeddingMissingAPIKey verifies DoEmbed fails fast when the provider
// has no API key.
func TestEmbeddingMissingAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	p := CreateGoogle(ProviderSettings{BaseURL: "https://example.test/v1beta"})
	em := p.EmbeddingModel("gemini-embedding-001")
	_, err := em.DoEmbed(context.Background(), EmbedOptions{Values: []string{"x"}})
	if !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("err = %v, want ErrMissingAPIKey", err)
	}
}

// TestEmbeddingInvalidJSON verifies that a non-JSON body returns
// InvalidResponseDataError.
func TestEmbeddingInvalidJSON(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{testResponse(200, `not-json`)}}
	p := CreateGoogle(ProviderSettings{
		BaseURL: "https://example.test/v1beta",
		APIKey:  "k",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	em := p.EmbeddingModel("gemini-embedding-001")
	_, err := em.DoEmbed(context.Background(), EmbedOptions{Values: []string{"x"}})
	var ir InvalidResponseDataError
	if !errors.As(err, &ir) {
		t.Fatalf("err = %T (%v)", err, err)
	}
}

// TestEmbeddingSingleViaHTTP verifies the single-value flow against a real
// httptest.Server (sanity-check the actual wire request).
func TestEmbeddingSingleViaHTTP(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models/text-embedding-005:embedContent" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"embedding":{"values":[0.5,0.25]}}`)
	}))
	defer srv.Close()

	p := CreateGoogle(ProviderSettings{BaseURL: srv.URL + "/v1beta", APIKey: "k", Retry: &RetryOptions{MaxRetries: 0}})
	em := p.EmbeddingModel("text-embedding-005")
	result, err := em.DoEmbed(context.Background(), EmbedOptions{Values: []string{"hi"}})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}
	if captured["model"] != "models/text-embedding-005" {
		t.Errorf("captured.model = %v", captured["model"])
	}
	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != 2 {
		t.Fatalf("Embeddings = %v", result.Embeddings)
	}
}
