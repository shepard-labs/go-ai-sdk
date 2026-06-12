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

	// Provider tools are serialized by the Responses (and Chat) models.
	tools := []openai.Tool{
		provider.Tools().WebSearch(openai.WebSearchArgs{}),
		provider.Tools().CodeInterpreter(nil),
	}

	model := provider.Responses("gpt-4o")
	result, err := model.DoGenerate(context.Background(), openai.ResponsesGenerateOptions{
		Instructions: "Use tools when helpful. Be brief.",
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "What is 19 * 23? You may use the code interpreter if you want to verify."},
			}},
		},
		Tools:           tools,
		MaxOutputTokens: intPtr(500),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case openai.TextContent:
			fmt.Println(c.Text)
		case openai.ToolCallContent:
			fmt.Printf("[tool %s executed=%v]\n", c.ToolName, c.ProviderExecuted)
		case openai.ReasoningContent:
			if c.Text != "" {
				fmt.Printf("[reasoning] %s\n", c.Text)
			}
		}
	}
}

func intPtr(v int) *int { return &v }
