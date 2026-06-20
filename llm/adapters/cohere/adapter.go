// Package cohere registers the Cohere provider with the llm registry.
// Blank-import it to make "cohere:<model-id>" available to registry.NewClient.
package cohere

import (
	"github.com/shepard-labs/go-ai-sdk/llm"
	impl "github.com/shepard-labs/go-ai-sdk/llm/adapters"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func init() {
	registry.Register("cohere", func(modelID string, opts registry.ProviderOptions) (llm.Client, error) {
		return impl.NewCohereClientWithSettings(impl.CohereSettings{APIKey: opts.APIKey, BaseURL: opts.BaseURL, HTTPClient: opts.HTTPClient}, modelID)
	})
}
