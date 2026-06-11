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

	// ProviderRouting controls which upstream providers OpenRouter may use.
	// Order specifies a preferred list; AllowFallbacks=false pins to that list.
	allowFallbacks := false
	model := provider.Chat("openai/gpt-4o-mini", openrouter.ChatOptions{
		Provider: &openrouter.ProviderRouting{
			Order:          []string{"OpenAI", "Azure"},
			AllowFallbacks: &allowFallbacks,
			// DataCollection denies storing prompts on provider infrastructure.
			DataCollection: openrouter.DataCollectionDeny,
			// Quantizations restricts to specific weight precisions.
			Quantizations: []openrouter.Quantization{
				openrouter.QuantizationFP16,
				openrouter.QuantizationBF16,
			},
			// Sort by lowest price when multiple providers match.
			Sort: openrouter.ProviderSortPrice,
		},
	})

	fmt.Println("Sending request with provider routing constraints...")

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "Name the capital of France in one word."},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		if text, ok := content.(openrouter.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	// The provider that actually served the request is available in metadata.
	if or, ok := result.ProviderMetadata["openrouter"].(map[string]any); ok {
		fmt.Printf("\nServed by: %v\n", or["provider"])
	}
}
