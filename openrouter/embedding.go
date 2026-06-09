package openrouter

import (
	"context"
	"encoding/json"
)

type embeddingModel struct {
	provider *openRouterProvider
	modelID  string
	options  EmbeddingOptions
}

func (m *embeddingModel) ModelID() string             { return m.modelID }
func (m *embeddingModel) Provider() string            { return "openrouter.embedding" }
func (m *embeddingModel) MaxEmbeddingsPerCall() int   { return defaultMaxEmbeddings }
func (m *embeddingModel) SupportsParallelCalls() bool { return true }

func (m *embeddingModel) DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error) {
	body := map[string]any{"model": m.modelID, "input": opts.Values}
	if m.options.User != "" {
		body["user"] = m.options.User
	}
	if m.options.Provider != nil {
		body["provider"] = m.options.Provider
	}
	mergeBody(body, m.provider.extraBody)
	mergeBody(body, m.options.ExtraBody)
	mergeOpenRouterOptions(body, opts.ProviderOptions)
	var resp embeddingResponse
	raw, h, err := m.provider.postJSON(ctx, "/embeddings", body, opts.Headers, &resp)
	if err != nil {
		return nil, err
	}
	embs := make([][]float64, len(resp.Data))
	for _, item := range resp.Data {
		if item.Index >= 0 && item.Index < len(embs) {
			embs[item.Index] = item.Embedding
		}
	}
	rb, _ := json.Marshal(body)
	usage := EmbeddingUsage{}
	if resp.Usage != nil {
		usage.Tokens = resp.Usage.PromptTokens
	}
	return &EmbedResult{Embeddings: embs, Usage: usage, ProviderMetadata: providerMetadata(resp.Provider, resp.Usage, nil, nil), Request: RequestMetadata{Body: rb}, Response: ResponseMetadata{ModelID: resp.Model, Headers: h, RawBody: raw}}, nil
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model    string    `json:"model"`
	Provider string    `json:"provider"`
	Usage    *apiUsage `json:"usage"`
}
