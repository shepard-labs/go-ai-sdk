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
		Metadata: &anthropic.Metadata{UserID: "example-user-123"},
	})

	result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.UserMessage{Content: []anthropic.UserContent{
				anthropic.TextContent{Text: "Write three concise product names for a Go SDK. End with END."},
			}},
		},
		MaxTokens:     150,
		StopSequences: []string{"END"},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
	fmt.Printf("\nFinish reason: %s\n", result.FinishReason)
}

func printText(contents []anthropic.Content) {
	for _, content := range contents {
		if text, ok := content.(anthropic.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
