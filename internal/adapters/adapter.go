// Package adapters translates ai.RouterOptions into the per-provider
// GenerateOptions / StreamOptions types and wraps the dispatch call.
// Each provider has its own file (anthropic.go, openai.go); this file
// holds helpers shared by the per-provider adapters.
package adapters

// MaxTokensAsInt dereferences a *int, returning 0 when nil. It exists
// because the Anthropic and OpenAI provider option types disagree on
// the pointer/value shape of the max-tokens field (anthropic is int,
// openai is *int).
func MaxTokensAsInt(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}

// MaxTokensCopy returns a *int copy of the input, suitable for filling
// the *int-shaped fields on the OpenAI-compatible option types. Returns
// nil when v is nil so the provider's default kicks in.
func MaxTokensCopy(v *int) *int {
	if v == nil {
		return nil
	}
	c := *v
	return &c
}
