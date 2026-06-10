package cohere

import (
	"context"
	"encoding/json"
	"fmt"
)

func (m *cohereRerankingModel) DoRerank(ctx context.Context, opts RerankOptions) (*RerankResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	co, err := parseRerankingOptions(opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	docs, warnings, err := rerankDocuments(opts.Documents)
	if err != nil {
		return nil, err
	}
	bodyMap := map[string]any{"model": m.modelID, "query": opts.Query, "documents": docs}
	if opts.TopN != nil {
		bodyMap["top_n"] = *opts.TopN
	}
	if co.MaxTokensPerDoc != nil {
		bodyMap["max_tokens_per_doc"] = *co.MaxTokensPerDoc
	}
	if co.Priority != nil {
		bodyMap["priority"] = *co.Priority
	}
	body, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointRerank, body, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	var decoded cohereRerankResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, InvalidResponseDataError{Message: err.Error(), Data: string(resp.Body)}
	}
	ranking := make([]RankedDocument, len(decoded.Results))
	for i, r := range decoded.Results {
		ranking[i] = RankedDocument{Index: r.Index, RelevanceScore: r.RelevanceScore}
	}
	return &RerankResult{Ranking: ranking, Warnings: warnings, Response: ResponseMetadata{ID: decoded.ID, Headers: cloneHeader(resp.Headers), Body: append([]byte(nil), resp.Body...)}}, nil
}

func rerankDocuments(in RerankDocuments) ([]string, []Warning, error) {
	switch in.Type {
	case "", "text":
		out := make([]string, len(in.Values))
		for i, v := range in.Values {
			s, ok := v.(string)
			if !ok {
				return nil, nil, fmt.Errorf("cohere: text document %d is %T", i, v)
			}
			out[i] = s
		}
		return out, nil, nil
	case "object":
		out := make([]string, len(in.Values))
		for i, v := range in.Values {
			b, err := json.Marshal(v)
			if err != nil {
				return nil, nil, err
			}
			out[i] = string(b)
		}
		return out, []Warning{{Type: "compatibility", Feature: "object documents", Details: "Object documents are converted to strings."}}, nil
	default:
		return nil, nil, fmt.Errorf("cohere: unsupported rerank document type %s", in.Type)
	}
}
