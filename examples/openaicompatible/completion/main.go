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

	// CompletionModel targets the /completions endpoint (legacy text completions).
	// Use this for models like gpt-3.5-turbo-instruct.
	model := provider.CompletionModel("gpt-3.5-turbo-instruct")

	temperature := 0.7
	maxTokens := 100

	// CompletionOptions (suffix, logitBias, user) are passed via ProviderOptions
	// using the provider name as the namespace key.
	result, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{
			openaicompatible.UserMessage{
				Content: []openaicompatible.UserContent{
					openaicompatible.TextContent{Text: "The three laws of robotics are:"},
				},
			},
		},
		Temperature:     &temperature,
		MaxOutputTokens: &maxTokens,
		ProviderOptions: openaicompatible.ProviderOptions{
			"openai": {
				"suffix": "\n\nEnd of laws.",
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
	fmt.Printf("\nFinish reason: %s\n", result.FinishReason.Unified)
}
