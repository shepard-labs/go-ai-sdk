package openai

import (
	"context"
	"encoding/json"
)

// openaiEmbeddingModel implements EmbeddingModel.
type openaiEmbeddingModel struct {
	provider *openaiProvider
	modelID  string
}

func newEmbeddingModel(p *openaiProvider, modelID string) EmbeddingModel {
	return &openaiEmbeddingModel{provider: p, modelID: modelID}
}

func (m *openaiEmbeddingModel) ModelID() string  { return m.modelID }
func (m *openaiEmbeddingModel) Provider() string { return "openai.embedding" }
func (m *openaiEmbeddingModel) MaxEmbeddingsPerCall() int {
	// text-embedding-3-* supports up to 2048; older ada-002 is 2048 too.
	return 2048
}
func (m *openaiEmbeddingModel) SupportsParallelCalls() bool { return true }

func (m *openaiEmbeddingModel) DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	body := m.buildEmbedRequest(opts)
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointEmbeddings, encoded, opts.Headers)
	if err != nil {
		return nil, err
	}
	return m.parseEmbedResponse(resp.Body, encoded)
}

func (m *openaiEmbeddingModel) buildEmbedRequest(opts EmbedOptions) map[string]any {
	body := map[string]any{
		"model":           m.modelID,
		"input":           opts.Values,
		"encoding_format": "float",
	}
	if v, ok := opts.ProviderOptions["openai"]; ok {
		if dims, ok := v["dimensions"].(float64); ok {
			body["dimensions"] = int(dims)
		} else if dims, ok := v["dimensions"].(int); ok {
			body["dimensions"] = dims
		}
		if user, ok := v["user"].(string); ok {
			body["user"] = user
		}
	}
	return body
}

func (m *openaiEmbeddingModel) parseEmbedResponse(body, requestBody []byte) (*EmbedResult, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse: " + err.Error()}
	}
	result := &EmbedResult{
		Request:  RequestMetadata{Body: requestBody},
		Response: ResponseMetadata{Body: body, ModelID: m.modelID},
	}
	if data, ok := raw["data"].([]any); ok {
		embeddings := make([][]float64, 0, len(data))
		for _, item := range data {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if arr, ok := obj["embedding"].([]any); ok {
				emb := make([]float64, 0, len(arr))
				for _, v := range arr {
					if f, ok := v.(float64); ok {
						emb = append(emb, f)
					}
				}
				embeddings = append(embeddings, emb)
			} else if _, ok := obj["embedding"].(string); ok {
				result.Warnings = append(result.Warnings, Warning{Type: "other", Message: "base64 encoding not yet supported; skipping"})
			}
		}
		result.Embeddings = embeddings
	}
	if usage, ok := raw["usage"].(map[string]any); ok {
		if t, ok := usage["prompt_tokens"].(float64); ok {
			result.Usage = &EmbeddingUsage{Tokens: int(t)}
		}
	}
	return result, nil
}
