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

	// OpenRouter's web search plugin lets the model fetch live information
	// from the web before generating a response. Pass it via ChatOptions.Plugins
	// when creating the model.
	maxResults := 5
	model := provider.Chat("openai/gpt-4o-mini", openrouter.ChatOptions{
		Plugins: []openrouter.Plugin{
			openrouter.WebPlugin{
				MaxResults: &maxResults,
				Engine:     openrouter.EngineAuto,
			},
		},
	})

	fmt.Println("Sending request with web search plugin...")

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "What are the latest Go release notes?"},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		switch c := content.(type) {
		case openrouter.TextContent:
			fmt.Println(c.Text)
		case openrouter.SourceContent:
			fmt.Printf("\nSource: %s — %s\n", c.Title, c.URL)
		}
	}

	// Alternatively, use the web_search provider-defined tool directly.
	fmt.Println("\n--- Using web_search tool ---")

	tools := provider.Tools()
	wsModel := provider.Chat("openai/gpt-4o-mini")

	result2, err := wsModel.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "What is the current Go version?"},
				},
			},
		},
		Tools: []openrouter.Tool{tools.WebSearch(openrouter.WebSearchToolArgs{
			MaxResults: &maxResults,
		})},
	})
	if err != nil {
		log.Fatalf("Error generating response with web_search tool: %v", err)
	}

	for _, content := range result2.Content {
		if text, ok := content.(openrouter.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
