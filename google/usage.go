package google

import (
	"encoding/json"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// convertGoogleUsage converts raw Google UsageMetadata to the SDK Usage type.
// Mapping:
//
//	inputTokens.total     = promptTokenCount
//	inputTokens.cacheRead = cachedContentTokenCount
//	inputTokens.noCache   = promptTokenCount - cachedContentTokenCount
//	outputTokens.total    = candidatesTokenCount + thoughtsTokenCount
//	outputTokens.text     = candidatesTokenCount
//	outputTokens.reasoning = thoughtsTokenCount
func convertGoogleUsage(raw *internal.APIUsageMetadata) Usage {
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
		Raw: marshalUsageRaw(raw),
	}
}

// defaultChatUsage returns an empty Usage for skeleton stubs that do not yet
// implement usage conversion.
func defaultChatUsage() Usage {
	return Usage{}
}

func intPtr(v int) *int { return &v }

func marshalUsageRaw(v any) json.RawMessage {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}
