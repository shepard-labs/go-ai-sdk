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

	effort := "high"
	summary := "auto"
	model := provider.Responses("o4-mini")

	result, err := model.DoGenerate(context.Background(), openai.ResponsesGenerateOptions{
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "Which data structure fits a small LRU cache? Explain briefly."},
			}},
		},
		Reasoning: &openai.ReasoningConfig{
			Effort:  &effort,
			Summary: &summary,
		},
		MaxOutputTokens: intPtr(1200),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case openai.ReasoningContent:
			fmt.Printf("Reasoning:\n%s\n\n", c.Text)
		case openai.TextContent:
			fmt.Printf("Answer:\n%s\n", c.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
