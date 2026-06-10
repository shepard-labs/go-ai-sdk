package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/cohere"
)

const apiKey = "your-cohere-api-key"

func main() {
	provider := cohere.CreateCohere(cohere.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(cohere.ModelCommandA032025))

	fmt.Println("Sending request to Cohere...")

	result, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
		Messages: []cohere.Message{
			cohere.UserMessage{
				Content: []cohere.UserContent{
					cohere.TextContent{Text: "Write a haiku about programming in Go."},
				},
			},
		},
		MaxOutputTokens: ptr(100),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Println("\nResponse:")
	for _, content := range result.Content {
		if text, ok := content.(cohere.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
		deref(result.Usage.InputTokens.Total),
		deref(result.Usage.OutputTokens.Total),
	)
}

func ptr(v int) *int { return &v }

func deref(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
