// Package openai registers the OpenAI provider with the llm registry.
// Blank-import it to make "openai:<model-id>" available to registry.NewClient.
package openai

import (
	"github.com/shepard-labs/go-ai-sdk/llm"
	impl "github.com/shepard-labs/go-ai-sdk/llm/adapters"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func init() {
	registry.Register("openai", func(modelID string, opts registry.ProviderOptions) (llm.Client, error) {
		return impl.NewOpenAIClientWithSettings(impl.OpenAISettings{APIKey: opts.APIKey, BaseURL: opts.BaseURL, HTTPClient: opts.HTTPClient}, modelID)
	})
}
