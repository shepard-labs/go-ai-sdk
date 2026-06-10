package cohere

import (
	"context"
	"encoding/json"
)

func (m *cohereEmbeddingModel) DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	if len(opts.Values) > m.MaxEmbeddingsPerCall() {
		return nil, TooManyEmbeddingValuesForCallError{Provider: m.Provider(), ModelID: m.modelID, MaxEmbeddingsPerCall: m.MaxEmbeddingsPerCall(), Values: append([]string(nil), opts.Values...)}
	}
	co, err := parseEmbeddingOptions(opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	inputType := co.InputType
	if inputType == "" {
		inputType = "search_query"
	}
	bodyMap := map[string]any{"model": m.modelID, "embedding_types": []string{"float"}, "texts": append([]string(nil), opts.Values...), "input_type": inputType}
	if co.Truncate != "" {
		bodyMap["truncate"] = co.Truncate
	}
	if co.OutputDimension != nil {
		bodyMap["output_dimension"] = *co.OutputDimension
	}
	body, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointEmbed, body, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	var decoded cohereEmbeddingResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, InvalidResponseDataError{Message: err.Error(), Data: string(resp.Body)}
	}
	return &EmbedResult{Embeddings: decoded.Embeddings.Float, Usage: &EmbeddingUsage{Tokens: decoded.Meta.BilledUnits.InputTokens}, Warnings: []Warning{}, Request: RequestMetadata{Body: append([]byte(nil), body...)}, Response: ResponseMetadata{Headers: cloneHeader(resp.Headers), Body: append([]byte(nil), resp.Body...)}}, nil
}
