package ai

// ProviderCatalog maps a provider's display name (matching the provider's
// Name() return — e.g. "anthropic", "openai") to the list of model IDs
// the router is allowed to choose for that provider. The catalog is used
// to (1) build the system prompt for the routing call to Claude Haiku and
// (2) validate the model the routing call returns.
type ProviderCatalog map[string][]string

// RouterOptions is the provider-agnostic input to Router.Generate and
// Router.Stream. The router translates these to the chosen provider's
// own option types at call time.
type RouterOptions struct {
	// Prompt is the user prompt text. The router does not parse or
	// otherwise inspect it; it is forwarded to the routing model
	// (Claude Haiku) for the model-selection decision, and to the
	// chosen provider for the actual call.
	Prompt string

	// System is the user-supplied system prompt. For the chosen
	// provider, it is included as the system message of the call.
	// It is NOT forwarded to the routing call.
	System string

	// Temperature is the sampling temperature for the chosen
	// provider. Nil leaves the provider default.
	Temperature *float64

	// MaxTokens is the maximum number of tokens the chosen provider
	// may generate. Nil leaves the provider default.
	MaxTokens *int

	// ForceProvider bypasses the Haiku routing call and dispatches
	// directly to the named provider. ForceModel must also be set.
	// Useful for tests and for "always use this model" debug flags.
	ForceProvider string
	// ForceModel is the model to use when ForceProvider is set.
	ForceModel string
}

// Selection describes the (provider, model) pair the router chose. It is
// returned alongside the underlying Generate/Stream result so callers
// can log it or attribute cost.
type Selection struct {
	// Provider is the provider's display name — matches the keys in
	// ProviderCatalog and the Provider's Name() return value.
	Provider string
	// Model is the model ID within the chosen provider.
	Model string
	// Reason is a short one-sentence justification returned by the
	// routing model (Claude Haiku).
	Reason string
}
