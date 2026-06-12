package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Image("dall-e-3")

	result, err := model.DoGenerate(context.Background(), openai.ImageGenerateOptions{
		Prompt: "A serene mountain lake at sunrise, photorealistic",
		N:      1,
		Size:   "1024x1024",
	})
	if err != nil {
		log.Fatalf("Error generating image: %v", err)
	}

	fmt.Printf("Generated %d image(s)\n", len(result.Images))
	for i, img := range result.Images {
		if strings.HasPrefix(img, "http://") || strings.HasPrefix(img, "https://") {
			fmt.Printf("  [%d] URL: %s\n", i, img)
			continue
		}
		// Many models return base64 data URLs or raw base64.
		payload := img
		if strings.HasPrefix(img, "data:") {
			if idx := strings.Index(img, ","); idx >= 0 {
				payload = img[idx+1:]
			}
		}
		raw, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			fmt.Printf("  [%d] (could not decode as base64, length %d)\n", i, len(img))
			continue
		}
		name := fmt.Sprintf("openai-image-%d.png", i)
		if err := os.WriteFile(name, raw, 0o644); err != nil {
			log.Fatalf("write %s: %v", name, err)
		}
		fmt.Printf("  [%d] Wrote %d bytes to %s\n", i, len(raw), name)
	}

	for _, w := range result.Warnings {
		fmt.Printf("Warning: %s — %s\n", w.Feature, w.Message)
	}
}
