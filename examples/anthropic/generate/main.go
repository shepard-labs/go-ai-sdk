package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	// Passing APIKey directly is useful for small examples. In production,
	// prefer loading the key from an environment variable or secret manager.
	provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(anthropic.ModelClaudeSonnet46))

	fmt.Println("Sending request to Claude...")

	result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.UserMessage{
				Content: []anthropic.UserContent{
					anthropic.TextContent{Text: "Write a haiku about programming in Go."},
				},
			},
		},
		MaxTokens: 100,
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	// Print the result
	fmt.Println("\nResponse:")
	for _, content := range result.Content {
		if text, ok := content.(anthropic.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
		result.Usage.InputTokens.Total,
		result.Usage.OutputTokens.Total,
	)
}
