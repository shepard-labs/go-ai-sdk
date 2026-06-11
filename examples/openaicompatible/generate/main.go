package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

func main() {
	// Passing APIKey directly is useful for small examples. In production,
	// prefer loading the key from an environment variable or secret manager.
	provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
		BaseURL: "https://api.openai.com/v1",
		Name:    "openai",
		APIKey:  apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model("gpt-4o")

	fmt.Println("Sending request...")

	result, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{
			openaicompatible.UserMessage{
				Content: []openaicompatible.UserContent{
					openaicompatible.TextContent{Text: "Write a haiku about programming in Go."},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Println("\nResponse:")
	for _, content := range result.Content {
		if text, ok := content.(openaicompatible.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	if result.Usage.InputTokens.Total != nil {
		fmt.Printf("\nUsage: %d input tokens, ", *result.Usage.InputTokens.Total)
	}
	if result.Usage.OutputTokens.Total != nil {
		fmt.Printf("%d output tokens\n", *result.Usage.OutputTokens.Total)
	}
}
