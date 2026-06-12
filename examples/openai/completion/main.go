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

	model := provider.Completion("gpt-3.5-turbo-instruct")

	temperature := 0.7
	maxTokens := 100

	result, err := model.DoGenerate(context.Background(), openai.GenerateOptions{
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "The three laws of robotics are:"},
			}},
		},
		Temperature:     &temperature,
		MaxOutputTokens: &maxTokens,
		ProviderOptions: openai.ProviderOptions{
			"openai": {
				"suffix": "\n\nEnd of laws.",
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
	fmt.Printf("\nFinish reason: %s\n", result.FinishReason.Unified)
}

func printText(contents []openai.Content) {
	for _, content := range contents {
		if text, ok := content.(openai.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
