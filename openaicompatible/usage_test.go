package openaicompatible

import (
	"encoding/json"
	"testing"
)

func TestUsageConversion(t *testing.T) {
	usage := defaultChatUsage(nil)
	if usage.InputTokens.Total != nil || usage.OutputTokens.Total != nil || usage.Raw != nil {
		t.Fatalf("nil usage = %#v", usage)
	}
	raw := json.RawMessage(`{"prompt_tokens":10}`)
	usage = defaultChatUsage(&OpenAICompatibleTokenUsage{Raw: raw})
	if *usage.InputTokens.Total != 0 || *usage.InputTokens.NoCache != 0 || *usage.InputTokens.CacheRead != 0 || *usage.OutputTokens.Total != 0 || *usage.OutputTokens.Text != 0 || *usage.OutputTokens.Reasoning != 0 || string(usage.Raw) != string(raw) {
		t.Fatalf("missing numeric fields usage = %#v", usage)
	}
	prompt, completion, cached, reasoning, accepted, rejected := 10, 7, 3, 2, 5, 6
	usage = defaultChatUsage(&OpenAICompatibleTokenUsage{
		PromptTokens:     &prompt,
		CompletionTokens: &completion,
		PromptTokensDetails: &struct{ CachedTokens *int }{
			CachedTokens: &cached,
		},
		CompletionTokensDetails: &struct {
			ReasoningTokens          *int
			AcceptedPredictionTokens *int
			RejectedPredictionTokens *int
		}{ReasoningTokens: &reasoning, AcceptedPredictionTokens: &accepted, RejectedPredictionTokens: &rejected},
	})
	if *usage.InputTokens.Total != 10 || *usage.InputTokens.NoCache != 7 || *usage.InputTokens.CacheRead != 3 || *usage.OutputTokens.Total != 7 || *usage.OutputTokens.Text != 5 || *usage.OutputTokens.Reasoning != 2 {
		t.Fatalf("usage = %#v", usage)
	}
	metadata := predictionTokenMetadata("my-provider", ProviderOptions{"myProvider": {}}, &OpenAICompatibleTokenUsage{CompletionTokensDetails: &struct {
		ReasoningTokens          *int
		AcceptedPredictionTokens *int
		RejectedPredictionTokens *int
	}{AcceptedPredictionTokens: &accepted, RejectedPredictionTokens: &rejected}})
	inner := metadata["myProvider"].(map[string]any)
	if inner["acceptedPredictionTokens"] != 5 || inner["rejectedPredictionTokens"] != 6 {
		t.Fatalf("metadata = %#v", metadata)
	}
}

func TestCompletionAndEmbeddingUsage(t *testing.T) {
	prompt, completion := 4, 2
	usage := completionUsage(&OpenAICompatibleCompletionUsage{PromptTokens: &prompt, CompletionTokens: &completion, Raw: json.RawMessage(`{}`)})
	if *usage.InputTokens.Total != 4 || *usage.InputTokens.NoCache != 4 || usage.InputTokens.CacheRead != nil || *usage.OutputTokens.Total != 2 || *usage.OutputTokens.Text != 2 || usage.OutputTokens.Reasoning != nil || string(usage.Raw) != `{}` {
		t.Fatalf("completion usage = %#v", usage)
	}
	if embeddingUsage(nil) != nil {
		t.Fatalf("nil embedding usage should be nil")
	}
	if got := embeddingUsage(&prompt); got == nil || got.Tokens != 4 {
		t.Fatalf("embedding usage = %#v", got)
	}
}
