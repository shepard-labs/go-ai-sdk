package cohere

import "fmt"

type CohereLanguageModelOptions struct{ Thinking *CohereThinkingOptions }
type CohereThinkingOptions struct {
	Type        string
	TokenBudget *int
}
type CohereImagePartProviderOptions struct{ Detail string }
type CohereEmbeddingModelOptions struct {
	InputType, Truncate string
	OutputDimension     *int
}
type CohereRerankingModelOptions struct{ MaxTokensPerDoc, Priority *int }

func cohereOptions(opts ProviderOptions) map[string]any {
	if opts == nil {
		return nil
	}
	return opts["cohere"]
}
func parseLanguageOptions(opts ProviderOptions) (CohereLanguageModelOptions, error) {
	raw := cohereOptions(opts)
	var out CohereLanguageModelOptions
	v, ok := raw["thinking"]
	if !ok || v == nil {
		return out, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return out, fmt.Errorf("cohere: invalid provider option thinking")
	}
	th := &CohereThinkingOptions{}
	if tv, ok := m["type"]; ok {
		s, ok := tv.(string)
		if !ok || (s != "enabled" && s != "disabled") {
			return out, fmt.Errorf("cohere: invalid thinking type")
		}
		th.Type = s
	}
	if bv, ok := m["tokenBudget"]; ok {
		n, ok := anyInt(bv)
		if !ok {
			return out, fmt.Errorf("cohere: invalid thinking tokenBudget")
		}
		th.TokenBudget = &n
	}
	out.Thinking = th
	return out, nil
}
func parseImageOptions(md ProviderMetadata) (CohereImagePartProviderOptions, error) {
	var out CohereImagePartProviderOptions
	if md == nil {
		return out, nil
	}
	raw, _ := md["cohere"].(map[string]any)
	if raw == nil {
		return out, nil
	}
	if v, ok := raw["detail"]; ok {
		s, ok := v.(string)
		if !ok || (s != "auto" && s != "low" && s != "high") {
			return out, fmt.Errorf("cohere: invalid image detail")
		}
		out.Detail = s
	}
	return out, nil
}
func parseEmbeddingOptions(opts ProviderOptions) (CohereEmbeddingModelOptions, error) {
	raw := cohereOptions(opts)
	var out CohereEmbeddingModelOptions
	if v, ok := raw["inputType"]; ok {
		s, ok := v.(string)
		if !ok || !oneOf(s, "search_document", "search_query", "classification", "clustering") {
			return out, fmt.Errorf("cohere: invalid inputType")
		}
		out.InputType = s
	}
	if v, ok := raw["truncate"]; ok {
		s, ok := v.(string)
		if !ok || !oneOf(s, "NONE", "START", "END") {
			return out, fmt.Errorf("cohere: invalid truncate")
		}
		out.Truncate = s
	}
	if v, ok := raw["outputDimension"]; ok {
		n, ok := anyInt(v)
		if !ok || !oneIntOf(n, 256, 512, 1024, 1536) {
			return out, fmt.Errorf("cohere: invalid outputDimension")
		}
		out.OutputDimension = &n
	}
	return out, nil
}
func parseRerankingOptions(opts ProviderOptions) (CohereRerankingModelOptions, error) {
	raw := cohereOptions(opts)
	var out CohereRerankingModelOptions
	if v, ok := raw["maxTokensPerDoc"]; ok {
		n, ok := anyInt(v)
		if !ok {
			return out, fmt.Errorf("cohere: invalid maxTokensPerDoc")
		}
		out.MaxTokensPerDoc = &n
	}
	if v, ok := raw["priority"]; ok {
		n, ok := anyInt(v)
		if !ok {
			return out, fmt.Errorf("cohere: invalid priority")
		}
		out.Priority = &n
	}
	return out, nil
}
func anyInt(v any) (int, bool) {
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
		if n != nil {
			return *n, true
		}
	}
	return 0, false
}
func oneOf(s string, values ...string) bool {
	for _, v := range values {
		if s == v {
			return true
		}
	}
	return false
}
func oneIntOf(n int, values ...int) bool {
	for _, v := range values {
		if n == v {
			return true
		}
	}
	return false
}
