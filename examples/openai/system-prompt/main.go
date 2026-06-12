package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Chat("gpt-4o")

	result, err := model.DoGenerate(context.Background(), openai.GenerateOptions{
		Messages: []openai.Message{
			openai.SystemMessage{Content: "You are a terse technical writer. Reply in plain text, one sentence only."},
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "What is a goroutine?"},
			}},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
}

func printText(contents []openai.Content) {
	for _, content := range contents {
		if text, ok := content.(openai.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
