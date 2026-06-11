package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

const apiKey = "your-openrouter-api-key"

func main() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	// Reasoning models expose their chain-of-thought via ReasoningContent.
	// Enable reasoning and set an effort level via ChatOptions.
	includeReasoning := true
	model := provider.Chat("anthropic/claude-sonnet-4-5", openrouter.ChatOptions{
		IncludeReasoning: &includeReasoning,
		Reasoning: &openrouter.ReasoningOptions{
			Effort: openrouter.ReasoningEffortHigh,
		},
	})

	fmt.Println("Sending request to reasoning model...")

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "How many r's are in the word 'strawberry'? Show your reasoning."},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case openrouter.ReasoningContent:
			fmt.Println("=== Reasoning ===")
			fmt.Println(c.Text)
			fmt.Println("=================")
		case openrouter.TextContent:
			fmt.Println("\nAnswer:", c.Text)
		}
	}

	fmt.Printf("\nUsage: %d input tokens, %d output tokens (reasoning: %d)\n",
		result.Usage.InputTokens,
		result.Usage.OutputTokens,
		result.Usage.OutputTokensDetails.ReasoningTokens,
	)
}
