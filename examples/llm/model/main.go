// Command model demonstrates provider-local per-request model selection with
// llm.GenerateOptions.ModelID. Empty ModelID uses the client default.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/llm"
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	messages := []llm.Message{{
		Role:    "user",
		Content: []llm.Content{llm.TextContent{Text: "Summarize the tradeoff between latency and quality."}},
	}}

	fast, err := client.Generate(context.Background(), llm.GenerateOptions{
		ModelID:   "claude-haiku-4-5",
		Messages:  messages,
		MaxTokens: 128,
	})
	if err != nil {
		log.Fatalf("fast generate: %v", err)
	}

	strong, err := client.Generate(context.Background(), llm.GenerateOptions{
		ModelID:   "claude-sonnet-4-6",
		Messages:  messages,
		MaxTokens: 128,
	})
	if err != nil {
		log.Fatalf("strong generate: %v", err)
	}

	fmt.Printf("fast model: %s\n", fast.Response.ModelID)
	printText(fast)
	fmt.Printf("\nstrong model: %s\n", strong.Response.ModelID)
	printText(strong)

	if capabilities, ok := client.(llm.ModelCapabilities); ok {
		fmt.Printf("\nhaiku streaming support: %t\n", capabilities.CapabilitiesForModel("claude-haiku-4-5").Streaming)
	}
}

func printText(result *llm.GenerateResult) {
	for _, content := range result.Content {
		if text, ok := content.(llm.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
