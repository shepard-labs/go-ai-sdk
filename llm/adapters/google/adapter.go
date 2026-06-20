// Package google registers the Google provider with the llm registry.
// Blank-import it to make "google:<model-id>" available to registry.NewClient.
package google

import (
	"github.com/shepard-labs/go-ai-sdk/llm"
	impl "github.com/shepard-labs/go-ai-sdk/llm/adapters"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func init() {
	registry.Register("google", func(modelID string, opts registry.ProviderOptions) (llm.Client, error) {
		return impl.NewGoogleClientWithSettings(impl.GoogleSettings{APIKey: opts.APIKey, BaseURL: opts.BaseURL, HTTPClient: opts.HTTPClient}, modelID)
	})
}
