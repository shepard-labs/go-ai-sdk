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

	model := provider.Model(string(anthropic.ModelClaudeSonnet46), anthropic.ModelOptions{
		SendReasoning: true,
		Thinking: &anthropic.ThinkingConfig{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 1024,
		},
		Effort: "high",
	})

	result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.UserMessage{Content: []anthropic.UserContent{
				anthropic.TextContent{Text: "Choose the better data structure for a small LRU cache and explain why."},
			}},
		},
		MaxTokens: 1600,
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case anthropic.ReasoningContent:
			fmt.Printf("Reasoning:\n%s\n\n", c.Text)
		case anthropic.TextContent:
			fmt.Printf("Answer:\n%s\n", c.Text)
		}
	}
}
