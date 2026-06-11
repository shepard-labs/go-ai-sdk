package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
		BaseURL: "https://api.openai.com/v1",
		Name:    "openai",
		APIKey:  apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model("gpt-4o")

	result, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{
			openaicompatible.SystemMessage{
				Content: "You are a terse technical writer. Reply in plain text, one sentence only.",
			},
			openaicompatible.UserMessage{
				Content: []openaicompatible.UserContent{
					openaicompatible.TextContent{Text: "What is a goroutine?"},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		if text, ok := content.(openaicompatible.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
