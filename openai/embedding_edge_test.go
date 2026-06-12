package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// TestEmbeddingDoEmbedBasic verifies a basic embed call returns the
// expected number of embeddings.
func TestEmbeddingDoEmbedBasic(t *testing.T) {
	respBody := `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":3,"total_tokens":3}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if len(res.Embeddings) != 1 {
		t.Fatalf("len embeddings = %d", len(res.Embeddings))
	}
	if len(res.Embeddings[0]) != 3 {
		t.Errorf("embedding[0] len = %d", len(res.Embeddings[0]))
	}
	if res.Usage == nil || res.Usage.Tokens != 3 {
		t.Errorf("usage: %+v", res.Usage)
	}
}

// TestEmbeddingDoEmbedMultipleInputs verifies multiple input values
// produce multiple embeddings.
func TestEmbeddingDoEmbedMultipleInputs(t *testing.T) {
	respBody := `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}],"model":"text-embedding-3-small"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a", "b"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if len(res.Embeddings) != 2 {
		t.Errorf("len embeddings = %d", len(res.Embeddings))
	}
}

// TestEmbeddingDoEmbedRequestBody verifies the request body has the
// expected shape (model, input, encoding_format).
func TestEmbeddingDoEmbedRequestBody(t *testing.T) {
	respBody := `{"object":"list","data":[],"model":"text-embedding-3-small"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a", "b"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["model"] != "text-embedding-3-small" {
		t.Errorf("model: %v", body["model"])
	}
	if v, ok := body["input"].([]any); !ok || len(v) != 2 {
		t.Errorf("input: %T %v", body["input"], body["input"])
	}
	if body["encoding_format"] != "float" {
		t.Errorf("encoding_format: %v", body["encoding_format"])
	}
}

// TestEmbeddingDoEmbedInvalidJSON verifies that malformed JSON throws
// InvalidResponseDataError.
func TestEmbeddingDoEmbedInvalidJSON(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, "not json")}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a"},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T: %v", err, err)
	}
}

// TestEmbeddingDoEmbedBase64StringEmbedding verifies that a base64-encoded
// embedding in the response generates a warning and is skipped (not
// included in the result).
func TestEmbeddingDoEmbedBase64StringEmbedding(t *testing.T) {
	respBody := `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"aGVsbG8="}],"model":"text-embedding-3-small"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if len(res.Embeddings) != 0 {
		t.Errorf("expected 0 embeddings (skipped), got %d", len(res.Embeddings))
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected warning for base64 embedding")
	} else if !strings.Contains(res.Warnings[0].Message, "base64") {
		t.Errorf("warning: %v", res.Warnings[0])
	}
}

// TestEmbeddingDoEmbedNonArrayDataItemSkipped verifies that a non-object
// data item is silently skipped.
func TestEmbeddingDoEmbedNonArrayDataItemSkipped(t *testing.T) {
	respBody := `{"object":"list","data":["not-an-object",{"object":"embedding","index":1,"embedding":[0.5]}],"model":"text-embedding-3-small"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if len(res.Embeddings) != 1 {
		t.Errorf("len embeddings = %d, want 1", len(res.Embeddings))
	}
}

// TestEmbeddingNoUsage verifies that a response without a usage block
// leaves Usage nil.
func TestEmbeddingNoUsage(t *testing.T) {
	respBody := `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"model":"text-embedding-3-small"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if res.Usage != nil {
		t.Errorf("Usage: %+v, want nil", res.Usage)
	}
}

// TestEmbeddingDoEmbedProviderError verifies that when the provider has
// a setup error (no API key), DoEmbed returns it.
func TestEmbeddingDoEmbedProviderError(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateOpenAI(ProviderSettings{})
	// Override fetcher so we can be sure no request fires.
	p.(*openaiProvider).fetch = f
	_, err := p.Embedding("x").DoEmbed(context.Background(), EmbedOptions{Values: []string{"a"}})
	if err == nil {
		t.Fatal("expected error")
	}
	if err != ErrMissingAPIKey {
		t.Errorf("err = %v, want ErrMissingAPIKey", err)
	}
	if f.calls != 0 {
		t.Errorf("expected 0 calls, got %d", f.calls)
	}
}
