package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/google"
)

const apiKey = "your-google-api-key"

func main() {
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	// Enable thinking via ProviderOptions["google"].thinkingConfig.
	// includeThoughts = true makes the model return reasoning content blocks.
	includeThoughts := true
	model := provider.Model(google.ModelGemini35Flash)
	result, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages: []google.Message{
			google.UserMessage{Content: []google.UserContent{
				google.TextContent{Text: "Choose the better data structure for a small LRU cache and explain why."},
			}},
		},
		MaxOutputTokens: intPtr(1600),
		ProviderOptions: google.ProviderOptions{
			"google": {
				"thinkingConfig": map[string]any{
					"includeThoughts": includeThoughts,
					"thinkingLevel":   "medium",
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case google.ReasoningContent:
			fmt.Printf("Reasoning:\n%s\n\n", c.Text)
		case google.TextContent:
			fmt.Printf("Answer:\n%s\n", c.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
