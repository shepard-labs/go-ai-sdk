package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(anthropic.ModelClaudeSonnet46))
	result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.SystemMessage{
				Content: "You are a senior Go engineer. Be precise, practical, and concise.",
			},
			anthropic.UserMessage{Content: []anthropic.UserContent{
				anthropic.TextContent{Text: "Explain when to use context.Context in HTTP clients."},
			}},
		},
		MaxTokens: 250,
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
}

func printText(contents []anthropic.Content) {
	for _, content := range contents {
		if text, ok := content.(anthropic.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
