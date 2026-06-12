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

	// web_search is a provider-executed tool: OpenAI runs search and may return citations.
	searchTool := provider.Tools().WebSearch(openai.WebSearchArgs{})

	model := provider.Responses("gpt-4o")
	result, err := model.DoGenerate(context.Background(), openai.ResponsesGenerateOptions{
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "What is the latest stable release of Go? Cite sources if available."},
			}},
		},
		Tools:           []openai.Tool{searchTool},
		MaxOutputTokens: intPtr(400),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Println("Answer:")
	for _, content := range result.Content {
		switch c := content.(type) {
		case openai.TextContent:
			fmt.Println(c.Text)
		case openai.SourceContent:
			fmt.Printf("  Source: %s — %s\n", c.Title, c.URL)
		case openai.ToolCallContent:
			if c.ProviderExecuted {
				fmt.Printf("  (provider executed tool: %s)\n", c.ToolName)
			}
		}
	}
}

func intPtr(v int) *int { return &v }
