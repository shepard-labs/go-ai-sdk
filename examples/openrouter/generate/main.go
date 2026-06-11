package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

// Passing APIKey directly is useful for small examples. In production,
// prefer loading the key from an environment variable or secret manager.
const apiKey = "your-openrouter-api-key"

func main() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Chat("openai/gpt-4o-mini")

	fmt.Println("Sending request...")

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "Write a haiku about programming in Go."},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Println("\nResponse:")
	for _, content := range result.Content {
		if text, ok := content.(openrouter.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
		result.Usage.InputTokens, result.Usage.OutputTokens)
	fmt.Printf("Model: %s\n", result.Response.ModelID)
}
