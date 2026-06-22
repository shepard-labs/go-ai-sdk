// Command reasoning demonstrates provider-neutral per-request reasoning control
// with llm.GenerateOptions.Reasoning. The same request shape can be used across
// supported adapters; provider-specific reasoning knobs remain available through
// ProviderOptions when needed.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"

	// Blank-import the provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	result, err := client.Generate(context.Background(), llm.GenerateOptions{
		Messages: []llm.Message{{
			Role:    "user",
			Content: []llm.Content{llm.TextContent{Text: "Which data structure fits a small LRU cache? Explain briefly."}},
		}},
		MaxTokens: 512,
		Reasoning: &llm.ReasoningOptions{
			Effort: llm.ReasoningHigh,
		},
		// Use warn policy when you want the same code to run against providers that
		// support only a subset of neutral reasoning controls.
		UnsupportedFeaturePolicy: llm.UnsupportedFeaturePolicyWarn,
	})
	if err != nil {
		log.Fatalf("generate: %v", err)
	}

	for _, warning := range result.Warnings {
		fmt.Printf("warning[%s]: %s\n", warning.Provider, warning.Message)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case llm.ReasoningContent:
			fmt.Printf("Reasoning:\n%s\n\n", c.Text)
		case llm.TextContent:
			fmt.Printf("Answer:\n%s\n", c.Text)
		}
	}

	if result.Usage.ReasoningTokens > 0 {
		fmt.Printf("\nreasoning tokens: %d\n", result.Usage.ReasoningTokens)
	}
}
