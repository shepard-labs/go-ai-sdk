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

	temperature := 0.2
	topP := 0.9
	topK := 40
	freqPenalty := 0.1
	presPenalty := 0.1
	seed := 42

	result, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
		Messages: []cohere.Message{
			cohere.UserMessage{
				Content: []cohere.UserContent{
					cohere.TextContent{Text: "Write three concise product names for a Go SDK. End with END."},
				},
			},
		},
		MaxOutputTokens:  ptr(150),
		Temperature:      &temperature,
		TopP:             &topP,
		TopK:             &topK,
		FrequencyPenalty: &freqPenalty,
		PresencePenalty:  &presPenalty,
		Seed:             &seed,
		StopSequences:    []string{"END"},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
	fmt.Printf("\nFinish reason: %s\n", result.FinishReason.Unified)
}

func printText(contents []cohere.Content) {
	for _, content := range contents {
		if text, ok := content.(cohere.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func ptr(v int) *int { return &v }
