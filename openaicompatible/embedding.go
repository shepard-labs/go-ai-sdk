package openaicompatible

import (
	"context"
	"encoding/json"
)

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage *struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
	ProviderMetadata ProviderMetadata `json:"providerMetadata"`
}

func (m *openAICompatibleEmbeddingModel) DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	if len(opts.Values) > m.MaxEmbeddingsPerCall() {
		return nil, TooManyEmbeddingValuesForCallError{Provider: m.Provider(), ModelID: m.modelID, MaxEmbeddingsPerCall: m.MaxEmbeddingsPerCall(), Values: append([]string(nil), opts.Values...)}
	}
	providerOptions, warnings := mergeEmbeddingProviderOptions(m.provider.name, cloneProviderOptions(opts.ProviderOptions))
	bodyMap := map[string]any{
		"model":           m.modelID,
		"input":           append([]string(nil), opts.Values...),
		"encoding_format": "float",
	}
	if dimensions, ok := intProviderOption(providerOptions, "dimensions"); ok {
		bodyMap["dimensions"] = dimensions
	}
	if user, ok := stringProviderOption(providerOptions, "user"); ok && user != "" {
		bodyMap["user"] = user
	}
	body, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointEmbeddings, body, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	var decoded embeddingResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, err
	}
	embeddings := make([][]float64, len(decoded.Data))
	for i, item := range decoded.Data {
		embeddings[i] = append([]float64(nil), item.Embedding...)
	}
	var usage *EmbeddingUsage
	if decoded.Usage != nil {
		usage = &EmbeddingUsage{Tokens: decoded.Usage.PromptTokens}
	}
	return &EmbedResult{
		Embeddings:       embeddings,
		Usage:            usage,
		Warnings:         warnings,
		ProviderMetadata: decoded.ProviderMetadata,
		Request:          RequestMetadata{Body: append([]byte(nil), body...)},
		Response:         ResponseMetadata{Headers: cloneHeader(resp.Headers), Body: append([]byte(nil), resp.Body...)},
	}, nil
}

func intProviderOption(opts map[string]any, key string) (int, bool) {
	v, ok := opts[key]
	if !ok || v == nil {
		return 0, false
	}
	switch n := v.(type) {
	case int:
		return n, true
	case int8:
		return int(n), true
	case int16:
		return int(n), true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case uint:
		return int(n), true
	case uint8:
		return int(n), true
	case uint16:
		return int(n), true
	case uint32:
		return int(n), true
	case uint64:
		return int(n), true
	case float64:
		return int(n), true
	case *int:
		if n == nil {
			return 0, false
		}
		return *n, true
	}
	return 0, false
}

func stringProviderOption(opts map[string]any, key string) (string, bool) {
	v, ok := opts[key]
	if !ok || v == nil {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}
