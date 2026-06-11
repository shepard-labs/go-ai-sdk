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

	model := provider.Chat("openai/gpt-4o-mini")

	temperature := 0.2
	topP := 0.9
	topK := 40
	seed := 42
	freqPenalty := 0.1
	presPenalty := 0.1
	maxTokens := 150

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "List three concise names for a Go SDK. End with DONE."},
				},
			},
		},
		MaxTokens:        &maxTokens,
		Temperature:      &temperature,
		TopP:             &topP,
		TopK:             &topK,
		Seed:             &seed,
		FrequencyPenalty: &freqPenalty,
		PresencePenalty:  &presPenalty,
		Stop:             []string{"DONE"},
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
