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

	// Thinking is enabled via ProviderOptions using a token budget.
	model := provider.Model(string(cohere.ModelCommandAReasoning082025))

	tokenBudget := 1024
	result, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
		Messages: []cohere.Message{
			cohere.UserMessage{
				Content: []cohere.UserContent{
					cohere.TextContent{Text: "Choose the better data structure for a small LRU cache and explain why."},
				},
			},
		},
		MaxOutputTokens: ptr(1600),
		ProviderOptions: cohere.ProviderOptions{
			"cohere": {
				"thinking": map[string]any{
					"type":        "enabled",
					"tokenBudget": tokenBudget,
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case cohere.ReasoningContent:
			fmt.Printf("Reasoning:\n%s\n\n", c.Text)
		case cohere.TextContent:
			fmt.Printf("Answer:\n%s\n", c.Text)
		}
	}
}

func ptr(v int) *int { return &v }
