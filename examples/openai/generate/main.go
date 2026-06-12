package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	// Chat completions API (/chat/completions).
	model := provider.Chat("gpt-4o")

	fmt.Println("Sending request to OpenAI...")

	result, err := model.DoGenerate(context.Background(), openai.GenerateOptions{
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "Write a haiku about programming in Go."},
			}},
		},
		MaxOutputTokens: intPtr(100),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Println("\nResponse:")
	printText(result.Content)

	if result.Usage.InputTokens.Total != nil && result.Usage.OutputTokens.Total != nil {
		fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
			*result.Usage.InputTokens.Total,
			*result.Usage.OutputTokens.Total,
		)
	}
}

func printText(contents []openai.Content) {
	for _, content := range contents {
		if text, ok := content.(openai.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
