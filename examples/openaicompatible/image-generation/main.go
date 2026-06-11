package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
		BaseURL: "https://api.openai.com/v1",
		Name:    "openai",
		APIKey:  apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.ImageModel("dall-e-3")

	result, err := model.DoGenerate(context.Background(), openaicompatible.ImageGenerateOptions{
		Prompt: "A serene mountain lake at sunrise, photorealistic, 4K",
		N:      1,
		Size:   "1024x1024",
	})
	if err != nil {
		log.Fatalf("Error generating image: %v", err)
	}

	fmt.Printf("Generated %d image(s)\n", len(result.Images))
	for i, img := range result.Images {
		fmt.Printf("  [%d] %s\n", i, img)
	}

	for _, w := range result.Warnings {
		fmt.Printf("Warning: %s — %s\n", w.Feature, w.Details)
	}
}
