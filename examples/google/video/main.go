package main

import (
	"context"
	"fmt"
	"log"

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

	model := provider.VideoModel(google.ModelVeo31GeneratePreview)

	// Veo is a long-running operation: DoGenerate submits the request and
	// polls the operation until completion, then returns download URIs.
	result, err := model.DoGenerate(context.Background(), google.VideoGenerateOptions{
		Prompt:      "A drone shot over a foggy coastline at golden hour, waves crashing on rocks.",
		N:           1,
		AspectRatio: "16:9",
		ProviderOptions: google.ProviderOptions{
			"google": {
				"negativePrompt":   "blurry, low quality",
				"personGeneration": "dont_allow",
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating video: %v", err)
	}

	fmt.Printf("Generated %d video(s)\n", len(result.Videos))
	for i, uri := range result.Videos {
		fmt.Printf("  [%d] %s\n", i, uri)
	}

	for _, w := range result.Warnings {
		fmt.Printf("Warning: %s — %s\n", w.Feature, w.Details)
	}
}
