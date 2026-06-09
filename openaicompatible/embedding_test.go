package openaicompatible

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"
)

func TestEmbeddingDoEmbedRequestResponseAndMetadata(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"data":[{"embedding":[1,2]},{"embedding":[3,4]}],"usage":{"prompt_tokens":7},"providerMetadata":{"acme":{"key":"value"}}}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test/v1", Name: "acme-provider", Fetch: f, GenerateID: func() string { return "req-id" }, Retry: &RetryOptions{MaxRetries: 0}})
	dimensions := 3
	result, err := p.Embedding("embed-model").DoEmbed(context.Background(), EmbedOptions{
		Values:  []string{"a", "b"},
		Headers: http.Header{"X-Call": []string{"yes"}},
		ProviderOptions: ProviderOptions{
			"openai-compatible": map[string]any{"dimensions": 1},
			"openaiCompatible":  map[string]any{"dimensions": 2},
			"acme-provider":     map[string]any{"dimensions": &dimensions},
			"acmeProvider":      map[string]any{"user": "user-1"},
		},
	})
	if err != nil {
		t.Fatalf("DoEmbed error = %v", err)
	}
	if len(f.requests) != 1 {
		t.Fatalf("requests = %d", len(f.requests))
	}
	req := f.requests[0]
	if req.URL.Path != "/v1/embeddings" {
		t.Fatalf("path = %q", req.URL.Path)
	}
	if got := req.Header.Get("X-Call"); got != "yes" {
		t.Fatalf("per-call header = %q", got)
	}
	if got := req.Header.Get("x-request-id"); got != "req-id" {
		t.Fatalf("x-request-id = %q", got)
	}
	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "embed-model" || body["encoding_format"] != "float" || body["user"] != "user-1" || body["dimensions"].(float64) != 3 {
		t.Fatalf("body = %#v", body)
	}
	input := body["input"].([]any)
	if input[0] != "a" || input[1] != "b" {
		t.Fatalf("input = %#v", input)
	}
	if string(result.Response.Body) != `{"data":[{"embedding":[1,2]},{"embedding":[3,4]}],"usage":{"prompt_tokens":7},"providerMetadata":{"acme":{"key":"value"}}}` {
		t.Fatalf("response body metadata = %s", result.Response.Body)
	}
	if result.Response.Headers.Get("x-request-id") != "resp-id" {
		t.Fatalf("response headers = %#v", result.Response.Headers)
	}
	if len(result.Embeddings) != 2 || result.Embeddings[0][0] != 1 || result.Embeddings[1][0] != 3 {
		t.Fatalf("embeddings = %#v", result.Embeddings)
	}
	if result.Usage == nil || result.Usage.Tokens != 7 {
		t.Fatalf("usage = %#v", result.Usage)
	}
	metadata := result.ProviderMetadata["acme"].(map[string]any)
	if metadata["key"] != "value" {
		t.Fatalf("metadata = %#v", result.ProviderMetadata)
	}
	if len(result.Warnings) != 1 || result.Warnings[0].Message != deprecatedProviderOptionsWarningMessage {
		t.Fatalf("warnings = %#v", result.Warnings)
	}
}

func TestEmbeddingTooManyValuesAndProperties(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme"})
	m := p.Embedding("embed-model")
	if m.MaxEmbeddingsPerCall() != 2048 {
		t.Fatalf("default max = %d", m.MaxEmbeddingsPerCall())
	}
	if !m.SupportsParallelCalls() {
		t.Fatal("default parallel support = false")
	}
	values := make([]string, 2049)
	_, err := m.DoEmbed(context.Background(), EmbedOptions{Values: values})
	tooMany := TooManyEmbeddingValuesForCallError{}
	if !errors.As(err, &tooMany) || tooMany.Provider != "acme.embedding" || tooMany.ModelID != "embed-model" || tooMany.MaxEmbeddingsPerCall != 2048 || len(tooMany.Values) != 2049 {
		t.Fatalf("error = %#v", err)
	}
	parallel := false
	p = CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", MaxEmbeddingsPerCall: 4, SupportsParallelEmbeddingCalls: &parallel})
	m = p.Embedding("embed-model")
	if m.MaxEmbeddingsPerCall() != 4 || m.SupportsParallelCalls() {
		t.Fatalf("overridden properties max=%d parallel=%v", m.MaxEmbeddingsPerCall(), m.SupportsParallelCalls())
	}
}
