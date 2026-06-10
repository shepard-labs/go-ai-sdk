package google

import "encoding/json"

// convertGoogleUsage converts raw Google UsageMetadata to the SDK Usage type.
// Mapping:
//
//	inputTokens.total     = promptTokenCount
//	inputTokens.cacheRead = cachedContentTokenCount
//	inputTokens.noCache   = promptTokenCount - cachedContentTokenCount
//	outputTokens.total    = candidatesTokenCount + thoughtsTokenCount
//	outputTokens.text     = candidatesTokenCount
//	outputTokens.reasoning = thoughtsTokenCount
func convertGoogleUsage(raw *apiUsageMetadata) Usage {
	if raw == nil {
		return Usage{}
	}
	prompt := raw.PromptTokenCount
	cached := raw.CachedContentTokenCount
	candidates := raw.CandidatesTokenCount
	thoughts := raw.ThoughtsTokenCount
	totalOutput := candidates + thoughts
	noCache := prompt - cached
	return Usage{
		InputTokens: TokenCounts{
			Total:     intPtr(prompt),
			NoCache:   intPtr(noCache),
			CacheRead: intPtr(cached),
		},
		OutputTokens: OutputTokenCounts{
			Total:     intPtr(totalOutput),
			Text:      intPtr(candidates),
			Reasoning: intPtr(thoughts),
		},
		Raw: marshalRaw(raw),
	}
}

// defaultChatUsage returns an empty Usage for skeleton stubs that do not yet
// implement usage conversion.
func defaultChatUsage() Usage {
	return Usage{}
}

func intPtr(v int) *int { return &v }

func marshalRaw(v any) json.RawMessage {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}

// apiUsageMetadata is the subset of the Google UsageMetadata wire struct needed
// by convertGoogleUsage. The full struct lives in internal/api-types.go; this
// local alias is used by usage.go to avoid an import cycle.
type apiUsageMetadata = internalUsageMetadata

// internalUsageMetadata matches the fields of internal.apiUsageMetadata used in
// usage conversion.
type internalUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
}
