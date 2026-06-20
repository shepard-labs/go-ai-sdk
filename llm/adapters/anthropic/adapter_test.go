package anthropic_test

import (
	"testing"

	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func TestAnthropicAdapterRegistered(t *testing.T) {
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("NewClient error = %v", err)
	}
	if client == nil {
		t.Fatal("client is nil")
	}
}
