package openai

import "testing"

// Verifies the hard-coded encoding_format: "float" per spec.
func TestBuildEmbedRequestEncodingFormatFloat(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	got := (&openaiEmbeddingModel{provider: p, modelID: "text-embedding-3-small"}).
		buildEmbedRequest(EmbedOptions{Values: []string{"hi"}})
	if got["encoding_format"] != "float" {
		t.Errorf("encoding_format: %v, want float", got["encoding_format"])
	}
}

// Verifies MaxEmbeddingsPerCall is 2048 per spec.
func TestEmbeddingModelMaxEmbeddingsPerCall(t *testing.T) {
	m := newEmbeddingModel(newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1"), "text-embedding-3-large")
	if got := m.MaxEmbeddingsPerCall(); got != 2048 {
		t.Errorf("MaxEmbeddingsPerCall = %d, want 2048", got)
	}
}

// Verifies SupportsParallelCalls is true per spec.
func TestEmbeddingModelSupportsParallelCalls(t *testing.T) {
	m := newEmbeddingModel(newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1"), "text-embedding-3-large")
	if !m.SupportsParallelCalls() {
		t.Errorf("SupportsParallelCalls should be true")
	}
}

func TestBuildEmbedRequestWithDimensions(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	got := (&openaiEmbeddingModel{provider: p, modelID: "text-embedding-3-small"}).
		buildEmbedRequest(EmbedOptions{
			Values: []string{"hi"},
			ProviderOptions: ProviderOptions{
				"openai": map[string]any{"dimensions": 256},
			},
		})
	if got["dimensions"] != 256 {
		t.Errorf("dimensions: %v", got["dimensions"])
	}
}

func TestBuildEmbedRequestWithUserAndDimensionsFloat(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	got := (&openaiEmbeddingModel{provider: p, modelID: "text-embedding-3-small"}).
		buildEmbedRequest(EmbedOptions{
			Values: []string{"hi"},
			ProviderOptions: ProviderOptions{
				"openai": map[string]any{
					"dimensions": float64(512),
					"user":       "u-1",
				},
			},
		})
	if got["dimensions"] != 512 {
		t.Errorf("dimensions: %v", got["dimensions"])
	}
	if got["user"] != "u-1" {
		t.Errorf("user: %v", got["user"])
	}
}

func TestBuildEmbedRequestNoDimensions(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	got := (&openaiEmbeddingModel{provider: p, modelID: "text-embedding-3-small"}).
		buildEmbedRequest(EmbedOptions{Values: []string{"hi"}})
	if _, has := got["dimensions"]; has {
		t.Errorf("dimensions should not be set: %v", got["dimensions"])
	}
}
