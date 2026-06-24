package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/google"
)

const apiKey = "your-google-api-key"

func main() {
	// Passing APIKey directly is useful for small examples. In production,
	// prefer loading the key from an environment variable or secret manager.
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(google.ModelGemini35Flash)

	fmt.Println("Sending request to Gemini...")

	result, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages: []google.Message{
			google.UserMessage{Content: []google.UserContent{
				google.TextContent{Text: "Write a haiku about programming in Go."},
			}},
		},
		MaxOutputTokens: intPtr(100),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Println("\nResponse:")
	printText(result.Content)

	if result.Usage.InputTokens.Total != nil && result.Usage.OutputTokens.Total != nil {
		fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
			*result.Usage.InputTokens.Total,
			*result.Usage.OutputTokens.Total,
		)
	}
}

func printText(contents []google.Content) {
	for _, content := range contents {
		if text, ok := content.(google.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
