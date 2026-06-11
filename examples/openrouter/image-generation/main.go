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

	// OpenRouter routes image generation through the chat completions endpoint
	// using the "image" modality. Use Image() to get an ImageModel.
	model := provider.Image("google/gemini-2.5-flash-preview-05-20")

	result, err := model.DoGenerate(context.Background(), openrouter.ImageGenerateOptions{
		Prompt:      "A serene mountain lake at sunrise, photorealistic, 4K",
		AspectRatio: "16:9",
	})
	if err != nil {
		log.Fatalf("Error generating image: %v", err)
	}

	fmt.Printf("Generated %d image(s)\n", len(result.Images))
	for i, img := range result.Images {
		// Images are returned as base64-encoded strings.
		if len(img) > 60 {
			fmt.Printf("  [%d] base64 data (%d bytes), prefix: %s...\n", i, len(img), img[:60])
		} else {
			fmt.Printf("  [%d] %s\n", i, img)
		}
	}

	for _, w := range result.Warnings {
		fmt.Printf("Warning: %s — %s\n", w.Type, w.Message)
	}
}
