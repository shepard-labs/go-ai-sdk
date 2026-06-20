// Command failover demonstrates llm.WithFailover: wrapping a primary client so
// that, when a call fails with a retryable error, the loop transparently retries
// against a fallback client (here, a different provider). This is the standard
// pattern for surviving a provider outage or rate limit without changing call
// sites — the wrapped client satisfies the same llm.Client interface.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"

	// Blank-import every provider adapter we might fail over to.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/openai"
)

const (
	anthropicAPIKey = "sk-ant-api03-your-api-key"
	openaiAPIKey    = "sk-your-openai-api-key"
)

func main() {
	primary, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: anthropicAPIKey,
	})
	if err != nil {
		log.Fatalf("create primary: %v", err)
	}
	fallback, err := registry.NewClient("openai:gpt-4o", registry.ProviderOptions{
		APIKey: openaiAPIKey,
	})
	if err != nil {
		log.Fatalf("create fallback: %v", err)
	}

	// Wrap the primary. On a retryable error, GetNext supplies the next client to
	// try; ShouldFailover decides which errors are worth retrying.
	client := llm.WithFailover(primary, llm.FailoverConfig{
		MaxAttempts: 2,
		ShouldFailover: func(ctx context.Context, err error) bool {
			// Don't fail over on context cancellation; do for everything else.
			return !errors.Is(err, context.Canceled)
		},
		GetNext: func(attempt int) llm.Client {
			fmt.Printf("primary failed; failing over (attempt %d)\n", attempt)
			return fallback
		},
	})

	result, err := client.Generate(context.Background(), llm.GenerateOptions{
		Messages: []llm.Message{{
			Role:    "user",
			Content: []llm.Content{llm.TextContent{Text: "Say hello in one word."}},
		}},
		MaxTokens: 64,
	})
	if err != nil {
		log.Fatalf("generate (all providers failed): %v", err)
	}
	for _, content := range result.Content {
		if text, ok := content.(llm.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
