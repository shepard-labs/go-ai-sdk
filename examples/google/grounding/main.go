package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/google"
)

// grounding demonstrates the googleSearch provider tool. Gemini answers
// with citations pulled from the public web, returned as StreamSource parts
// alongside the text response.
const apiKey = "your-google-api-key"

func main() {
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	// googleSearch is a provider-executed tool — the model calls Google's
	// search backend itself and returns the answer with grounding sources.
	searchTool := provider.Tools().GoogleSearch()

	model := provider.Model(google.ModelGemini31ProPreview)
	result, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages: []google.Message{
			google.UserMessage{Content: []google.UserContent{
				google.TextContent{Text: "What is the latest stable release of Go and when was it published?"},
			}},
		},
		Tools:           []google.Tool{searchTool},
		MaxOutputTokens: intPtr(1000),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Println("Answer:")
	for _, content := range result.Content {
		if text, ok := content.(google.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	fmt.Println("\nSources:")
	if md, ok := result.ProviderMetadata["google"]; ok {
		if meta, ok := md.(map[string]any); ok {
			if sources, ok := meta["groundingMetadata"]; ok {
				fmt.Printf("%+v\n", sources)
			}
		}
	}
}

func intPtr(v int) *int { return &v }
