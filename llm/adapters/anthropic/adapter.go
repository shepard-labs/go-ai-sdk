// Package anthropic registers the Anthropic provider with the llm registry.
// Blank-import it to make "anthropic:<model-id>" available to registry.NewClient.
package anthropic

import (
	"github.com/shepard-labs/go-ai-sdk/llm"
	impl "github.com/shepard-labs/go-ai-sdk/llm/adapters"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func init() {
	registry.Register("anthropic", func(modelID string, opts registry.ProviderOptions) (llm.Client, error) {
		return impl.NewAnthropicClient(opts.APIKey, impl.AnthropicModelID(modelID))
	})
}
