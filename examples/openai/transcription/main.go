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
	if len(os.Args) < 2 {
		log.Fatalf("usage: %s <path-to-audio.mp3>\n", os.Args[0])
	}
	audio, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatalf("read audio: %v", err)
	}

	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Transcription("whisper-1")

	result, err := model.DoGenerate(context.Background(), openai.TranscriptionOptions{
		Audio:     audio,
		Filename:  os.Args[1],
		MediaType: "audio/mpeg",
	})
	if err != nil {
		log.Fatalf("Error transcribing: %v", err)
	}

	fmt.Printf("Transcript:\n%s\n", result.Text)
	if result.Language != "" {
		fmt.Printf("\nLanguage (ISO): %s\n", result.Language)
	}
	if len(result.Segments) > 0 {
		fmt.Printf("\nFirst segment: %q (%.2f–%.2fs)\n",
			result.Segments[0].Text,
			result.Segments[0].StartSecond,
			result.Segments[0].EndSecond,
		)
	}
}
