// Command vision demonstrates multimodal input with the llm package: a single
// user message can carry both TextContent and ImageContent. The image is
// supplied here by URL via ImageURLSource; inline bytes work too via
// ImageInlineSource{Data: ...} (e.g. for a local file read with os.ReadFile).
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"

	// Blank-import the provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	result, err := client.Generate(context.Background(), llm.GenerateOptions{
		Messages: []llm.Message{{
			Role: "user",
			Content: []llm.Content{
				llm.TextContent{Text: "Describe this image in one short sentence."},
				llm.ImageContent{
					Source: llm.ImageURLSource{URL: "https://images.unsplash.com/photo-1780476895954-b934fb2e480b"},
				},
				// For local bytes instead of a URL:
				//   data, _ := os.ReadFile("photo.jpg")
				//   llm.ImageContent{Source: llm.ImageInlineSource{Data: data}, MIME: "image/jpeg"}
			},
		}},
		MaxTokens: 200,
	})
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	for _, content := range result.Content {
		if text, ok := content.(llm.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
