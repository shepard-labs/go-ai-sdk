package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

const apiKey = "your-openrouter-api-key"

func main() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	// Completion targets the /completions endpoint (legacy text completions).
	// LanguageModel automatically routes "openai/gpt-3.5-turbo-instruct" to
	// the completion endpoint; use Completion() explicitly for other instruct
	// models if needed.
	model := provider.Completion("openai/gpt-3.5-turbo-instruct", openrouter.CompletionOptions{
		Suffix: "\n\nEnd of laws.",
	})

	temperature := 0.7
	maxTokens := 120

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "The three laws of robotics are:"},
				},
			},
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		if text, ok := content.(openrouter.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
	fmt.Printf("\nFinish reason: %s\n", result.FinishReason)
}
