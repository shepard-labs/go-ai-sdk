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

	temperature := 0.2
	topP := 0.9
	seed := 42
	freqPenalty := 0.1
	presPenalty := 0.1
	maxTokens := 150

	result, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{
			openaicompatible.UserMessage{
				Content: []openaicompatible.UserContent{
					openaicompatible.TextContent{Text: "List three concise names for a Go SDK. End with DONE."},
				},
			},
		},
		MaxOutputTokens:  &maxTokens,
		Temperature:      &temperature,
		TopP:             &topP,
		Seed:             &seed,
		FrequencyPenalty: &freqPenalty,
		PresencePenalty:  &presPenalty,
		StopSequences:    []string{"DONE"},
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
