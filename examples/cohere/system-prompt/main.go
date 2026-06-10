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

	result, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
		Messages: []cohere.Message{
			cohere.SystemMessage{
				Content: "You are a concise technical writer. Reply in plain text with no markdown.",
			},
			cohere.UserMessage{
				Content: []cohere.UserContent{
					cohere.TextContent{Text: "Explain what a context in Go is in one paragraph."},
				},
			},
		},
		MaxOutputTokens: ptr(200),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
}

func printText(contents []cohere.Content) {
	for _, content := range contents {
		if text, ok := content.(cohere.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func ptr(v int) *int { return &v }
