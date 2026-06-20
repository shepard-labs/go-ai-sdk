// Package openaicompatible registers the OpenAI-compatible provider with the
// llm registry. Blank-import it to make "openaicompatible:<model-id>"
// available to registry.NewClient. The registry's ProviderOptions.BaseURL is
// passed through so the same provider can target Ollama, LM Studio, or any
// OpenAI-compatible endpoint.
package openaicompatible

import (
	"github.com/shepard-labs/go-ai-sdk/llm"
	impl "github.com/shepard-labs/go-ai-sdk/llm/adapters"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func init() {
	registry.Register("openaicompatible", func(modelID string, opts registry.ProviderOptions) (llm.Client, error) {
		return impl.NewOpenAICompatibleClient(impl.OpenAICompatibleSettings{
			Name:       "openaicompatible",
			BaseURL:    opts.BaseURL,
			APIKey:     opts.APIKey,
			HTTPClient: opts.HTTPClient,
		}, modelID)
	})
}
