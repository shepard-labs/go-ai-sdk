package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// TestEmbeddingChunkedBatchMergesInOrder verifies inputs over the byte
// budget split into sequential calls whose embeddings merge in order with
// summed usage.
func TestEmbeddingChunkedBatchMergesInOrder(t *testing.T) {
	big1 := strings.Repeat("a", 200000)
	big2 := strings.Repeat("b", 200000)
	f := &recordingFetcher{responses: []*http.Response{
		response(200, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"usage":{"prompt_tokens":5,"total_tokens":5}}`),
		response(200, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.2]}],"usage":{"prompt_tokens":7,"total_tokens":7}}`),
	}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{big1, big2},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if f.calls != 2 {
		t.Fatalf("calls = %d, want 2", f.calls)
	}
	if len(res.Embeddings) != 2 {
		t.Fatalf("len embeddings = %d, want 2", len(res.Embeddings))
	}
	if res.Embeddings[0][0] != 0.1 || res.Embeddings[1][0] != 0.2 {
		t.Errorf("embeddings out of order: %v", res.Embeddings)
	}
	if res.Usage == nil || res.Usage.Tokens != 12 {
		t.Errorf("usage = %+v, want 12 tokens", res.Usage)
	}
}

// TestEmbeddingOversizedSingleValueStaysWhole verifies a single input over
// the budget still goes out alone in one call.
func TestEmbeddingOversizedSingleValueStaysWhole(t *testing.T) {
	big := strings.Repeat("z", 400000)
	f := &recordingFetcher{responses: []*http.Response{
		response(200, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.9]}],"usage":{"prompt_tokens":9,"total_tokens":9}}`),
	}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{big},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if f.calls != 1 {
		t.Errorf("calls = %d, want 1", f.calls)
	}
	if len(res.Embeddings) != 1 {
		t.Errorf("len embeddings = %d, want 1", len(res.Embeddings))
	}
}

// TestEmbeddingSmallBatchSingleCall verifies inputs within budget keep the
// original single-call behavior.
func TestEmbeddingSmallBatchSingleCall(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		response(200, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}],"usage":{"prompt_tokens":3,"total_tokens":3}}`),
	}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"a", "b"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
	if f.calls != 1 {
		t.Errorf("calls = %d, want 1", f.calls)
	}
	if len(res.Embeddings) != 2 {
		t.Errorf("len embeddings = %d, want 2", len(res.Embeddings))
	}
	if res.Usage == nil || res.Usage.Tokens != 3 {
		t.Errorf("usage = %+v, want 3 tokens", res.Usage)
	}
}

// TestChunkEmbedInputsBoundaries verifies chunk boundary behavior directly.
func TestChunkEmbedInputsBoundaries(t *testing.T) {
	if got := chunkEmbedInputs(nil, 10); len(got) != 1 {
		t.Errorf("empty input: %d chunks, want 1", len(got))
	}
	if got := chunkEmbedInputs([]string{"ab", "cd"}, 4); len(got) != 1 {
		t.Errorf("exact fit: %d chunks, want 1", len(got))
	}
	got := chunkEmbedInputs([]string{"ab", "cde"}, 4)
	if len(got) != 2 || len(got[0]) != 1 || len(got[1]) != 1 {
		t.Errorf("overflow: %q, want 2 single chunks", got)
	}
	got = chunkEmbedInputs([]string{"abcdef"}, 4)
	if len(got) != 1 || got[0][0] != "abcdef" {
		t.Errorf("oversized single: %q, want 1 whole chunk", got)
	}
}
