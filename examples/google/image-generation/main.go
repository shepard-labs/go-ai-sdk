package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/google"
)

const apiKey = "your-google-api-key"

func main() {
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.ImageModel(google.ModelGemini31FlashImage)
	result, err := model.DoGenerate(context.Background(), google.ImageGenerateOptions{
		Prompt:      "A serene mountain lake at sunrise, photorealistic, 4K",
		N:           1,
		AspectRatio: "16:9",
		ProviderOptions: google.ProviderOptions{
			"google": {
				"personGeneration": "dont_allow",
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating image: %v", err)
	}

	fmt.Printf("Generated %d image(s)\n", len(result.Images))
	for i, b64 := range result.Images {
		data, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			log.Printf("image %d: failed to decode base64: %v", i, err)
			continue
		}
		filename := fmt.Sprintf("image-%d.png", i)
		if err := os.WriteFile(filename, data, 0o644); err != nil {
			log.Printf("image %d: failed to write %s: %v", i, filename, err)
			continue
		}
		fmt.Printf("  [%d] %d bytes -> %s\n", i, len(data), filename)
	}

	for _, w := range result.Warnings {
		fmt.Printf("Warning: %s — %s\n", w.Feature, w.Details)
	}
}
