package openaicompatible

import "encoding/json"

// defaultChatUsage converts an [OpenAICompatibleTokenUsage] to the SDK [Usage]
// type following the spec-defined mapping for chat models.
func defaultChatUsage(usage *OpenAICompatibleTokenUsage) Usage {
	if usage == nil {
		return Usage{}
	}
	promptTokens := intValueOrZero(usage.PromptTokens)
	completionTokens := intValueOrZero(usage.CompletionTokens)
	cacheReadTokens := 0
	if usage.PromptTokensDetails != nil {
		cacheReadTokens = intValueOrZero(usage.PromptTokensDetails.CachedTokens)
	}
	reasoningTokens := 0
	if usage.CompletionTokensDetails != nil {
		reasoningTokens = intValueOrZero(usage.CompletionTokensDetails.ReasoningTokens)
	}
	return Usage{
		InputTokens: TokenCounts{
			Total:     intPtr(promptTokens),
			NoCache:   intPtr(promptTokens - cacheReadTokens),
			CacheRead: intPtr(cacheReadTokens),
		},
		OutputTokens: OutputTokenCounts{
			Total:     intPtr(completionTokens),
			Text:      intPtr(completionTokens - reasoningTokens),
			Reasoning: intPtr(reasoningTokens),
		},
		Raw: cloneRawMessage(usage.Raw),
	}
}

// completionUsage converts an [OpenAICompatibleCompletionUsage] to the SDK
// [Usage] type following the spec-defined mapping for completion models.
func completionUsage(usage *OpenAICompatibleCompletionUsage) Usage {
	if usage == nil {
		return Usage{}
	}
	promptTokens := intValueOrZero(usage.PromptTokens)
	completionTokens := intValueOrZero(usage.CompletionTokens)
	return Usage{
		InputTokens: TokenCounts{
			Total:   intPtr(promptTokens),
			NoCache: intPtr(promptTokens),
		},
		OutputTokens: OutputTokenCounts{
			Total: intPtr(completionTokens),
			Text:  intPtr(completionTokens),
		},
		Raw: cloneRawMessage(usage.Raw),
	}
}

func embeddingUsage(promptTokens *int) *EmbeddingUsage {
	if promptTokens == nil {
		return nil
	}
	return &EmbeddingUsage{Tokens: *promptTokens}
}

func predictionTokenMetadata(name string, opts ProviderOptions, usage *OpenAICompatibleTokenUsage) ProviderMetadata {
	key := metadataKeyForProviderOptions(name, opts)
	metadata := ProviderMetadata{key: map[string]any{}}
	if usage == nil || usage.CompletionTokensDetails == nil {
		return metadata
	}
	inner := metadata[key].(map[string]any)
	if usage.CompletionTokensDetails.AcceptedPredictionTokens != nil {
		inner["acceptedPredictionTokens"] = *usage.CompletionTokensDetails.AcceptedPredictionTokens
	}
	if usage.CompletionTokensDetails.RejectedPredictionTokens != nil {
		inner["rejectedPredictionTokens"] = *usage.CompletionTokensDetails.RejectedPredictionTokens
	}
	return metadata
}

func intValueOrZero(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}

func intPtr(v int) *int { return &v }

func cloneRawMessage(in json.RawMessage) json.RawMessage {
	if in == nil {
		return nil
	}
	return append(json.RawMessage(nil), in...)
}
