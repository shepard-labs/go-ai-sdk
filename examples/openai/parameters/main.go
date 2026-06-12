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

	temperature := 0.2
	topP := 0.9
	seed := 42

	result, err := model.DoGenerate(context.Background(), openai.GenerateOptions{
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "Write three concise product names for a Go SDK. End with END."},
			}},
		},
		MaxOutputTokens:  intPtr(150),
		Temperature:      &temperature,
		TopP:             &topP,
		Seed:             &seed,
		StopSequences:    []string{"END"},
		FrequencyPenalty: floatPtr(0.5),
		PresencePenalty:  floatPtr(0.2),
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

func intPtr(v int) *int           { return &v }
func floatPtr(v float64) *float64 { return &v }
