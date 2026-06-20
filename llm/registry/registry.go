// Package registry provides database/sql-style provider registration so a
// provider can be selected by name at runtime. Blank-import a provider's
// adapter subpackage to register it, then construct a client with NewClient.
package registry

import (
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// ProviderOptions carries the credentials and transport settings passed to a
// provider factory.
type ProviderOptions struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// ProviderFactory builds an llm.Client for a model ID and options.
type ProviderFactory func(modelID string, opts ProviderOptions) (llm.Client, error)

var (
	mu        sync.RWMutex
	factories = make(map[string]ProviderFactory)
)

// Register makes a provider available by name. It panics if name is empty, the
// factory is nil, or the name is already registered.
func Register(name string, factory ProviderFactory) {
	if name == "" {
		panic("registry: empty provider name")
	}
	if factory == nil {
		panic("registry: nil provider factory")
	}
	mu.Lock()
	defer mu.Unlock()
	if _, exists := factories[name]; exists {
		panic(fmt.Sprintf("registry: provider %q already registered", name))
	}
	factories[name] = factory
}

// NewClient constructs a client for a "provider:model-id" string, e.g.
// "anthropic:claude-sonnet-4-6". The model-id portion may itself contain
// colons. It returns a descriptive error if the provider is not registered.
func NewClient(model string, opts ProviderOptions) (llm.Client, error) {
	name, modelID, found := strings.Cut(model, ":")
	if !found || name == "" || modelID == "" {
		return nil, fmt.Errorf("registry: invalid model %q, want \"provider:model-id\"", model)
	}
	mu.RLock()
	factory, ok := factories[name]
	mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("registry: provider %q not registered; blank-import its adapter package (e.g. _ \"github.com/shepard-labs/go-ai-sdk/llm/adapters/%s\")", name, name)
	}
	return factory(modelID, opts)
}
