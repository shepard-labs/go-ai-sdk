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

	model := provider.Model(google.ModelGemini25Flash)
	result, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages: []google.Message{
			google.SystemMessage{
				Content: "You are a senior Go engineer. Be precise, practical, and concise.",
			},
			google.UserMessage{Content: []google.UserContent{
				google.TextContent{Text: "Explain when to use context.Context in HTTP clients."},
			}},
		},
		MaxOutputTokens: intPtr(250),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
}

func printText(contents []google.Content) {
	for _, content := range contents {
		if text, ok := content.(google.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
