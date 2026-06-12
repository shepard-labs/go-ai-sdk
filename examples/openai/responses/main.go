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

	// Provider.Model / LanguageModel / Call are aliases for Responses.
	model := provider.Responses("gpt-4o")

	fmt.Println("Sending request via the Responses API...")

	result, err := model.DoGenerate(context.Background(), openai.ResponsesGenerateOptions{
		Instructions: "Be concise.",
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "Name three strengths of the Go programming language."},
			}},
		},
		MaxOutputTokens: intPtr(200),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("\nResponse ID: %s\n\n", result.Response.ID)
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

func intPtr(v int) *int { return &v }
