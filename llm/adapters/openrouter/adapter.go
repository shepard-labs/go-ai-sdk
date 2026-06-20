// Package openrouter registers the OpenRouter provider with the llm registry.
// Blank-import it to make "openrouter:<model-id>" available to registry.NewClient.
package openrouter

import (
	"github.com/shepard-labs/go-ai-sdk/llm"
	impl "github.com/shepard-labs/go-ai-sdk/llm/adapters"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func init() {
	registry.Register("openrouter", func(modelID string, opts registry.ProviderOptions) (llm.Client, error) {
		return impl.NewOpenRouterClientWithSettings(impl.OpenRouterSettings{APIKey: opts.APIKey, BaseURL: opts.BaseURL, HTTPClient: opts.HTTPClient}, modelID)
	})
}
