package openrouter

type apiUsage struct {
	PromptTokens        int `json:"prompt_tokens,omitempty"`
	CompletionTokens    int `json:"completion_tokens,omitempty"`
	TotalTokens         int `json:"total_tokens,omitempty"`
	PromptTokensDetails struct {
		CachedTokens     int `json:"cached_tokens,omitempty"`
		CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	} `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	} `json:"completion_tokens_details,omitempty"`
	Cost        float64 `json:"cost,omitempty"`
	CostDetails struct {
		UpstreamInferenceCost float64 `json:"upstream_inference_cost,omitempty"`
	} `json:"cost_details,omitempty"`
}

func standardUsage(u apiUsage) Usage {
	total := u.TotalTokens
	if total == 0 {
		total = u.PromptTokens + u.CompletionTokens
	}
	return Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  total,
		InputTokensDetails: InputTokensDetails{
			CachedTokens:     u.PromptTokensDetails.CachedTokens,
			CacheWriteTokens: u.PromptTokensDetails.CacheWriteTokens,
		},
		OutputTokensDetails: OutputTokensDetails{ReasoningTokens: u.CompletionTokensDetails.ReasoningTokens},
		Raw:                 u,
	}
}

func openRouterUsage(u apiUsage) OpenRouterUsageAccounting {
	total := u.TotalTokens
	if total == 0 {
		total = u.PromptTokens + u.CompletionTokens
	}
	return OpenRouterUsageAccounting{
		PromptTokens:          u.PromptTokens,
		CachedTokens:          u.PromptTokensDetails.CachedTokens,
		CacheWriteTokens:      u.PromptTokensDetails.CacheWriteTokens,
		CompletionTokens:      u.CompletionTokens,
		ReasoningTokens:       u.CompletionTokensDetails.ReasoningTokens,
		TotalTokens:           total,
		Cost:                  u.Cost,
		UpstreamInferenceCost: u.CostDetails.UpstreamInferenceCost,
		Raw:                   u,
	}
}

func providerMetadata(provider string, usage *apiUsage, reasoning []ReasoningDetail, annotations []FileAnnotation) ProviderMetadata {
	or := map[string]any{"provider": provider}
	if reasoning != nil {
		or["reasoning_details"] = reasoning
	}
	if len(annotations) > 0 {
		or["annotations"] = annotations
	}
	if usage != nil {
		or["usage"] = openRouterUsage(*usage)
	}
	return ProviderMetadata{"openrouter": or}
}
