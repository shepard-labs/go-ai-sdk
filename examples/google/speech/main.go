package main

import (
	"context"
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

	model := provider.SpeechModel(google.ModelGemini25FlashPreviewTTS)

	result, err := model.DoGenerate(context.Background(), google.SpeechGenerateOptions{
		Text:         "Hello! Welcome to the Google Generative AI speech synthesis example.",
		Voice:        "Kore",
		OutputFormat: "wav",
	})
	if err != nil {
		log.Fatalf("Error generating speech: %v", err)
	}

	filename := "speech.wav"
	if err := os.WriteFile(filename, result.Audio, 0o644); err != nil {
		log.Fatalf("failed to write %s: %v", filename, err)
	}

	fmt.Printf("Wrote %d bytes of audio to %s\n", len(result.Audio), filename)
	for _, w := range result.Warnings {
		fmt.Printf("Warning: %s — %s\n", w.Feature, w.Details)
	}
}
