package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Speech("tts-1")

	result, err := model.DoGenerate(context.Background(), openai.SpeechGenerateOptions{
		Text:         "Hello! This is the OpenAI text to speech example for the go-ai-sdk.",
		Voice:        openai.VoiceNova,
		OutputFormat: "mp3",
	})
	if err != nil {
		log.Fatalf("Error generating speech: %v", err)
	}

	filename := "speech.mp3"
	if err := os.WriteFile(filename, result.Audio, 0o644); err != nil {
		log.Fatalf("failed to write %s: %v", filename, err)
	}

	fmt.Printf("Wrote %d bytes of audio to %s\n", len(result.Audio), filename)
	for _, w := range result.Warnings {
		fmt.Printf("Warning: %s — %s\n", w.Feature, w.Message)
	}
}
