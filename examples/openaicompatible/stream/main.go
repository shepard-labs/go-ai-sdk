package main

import (
	"context"
	"fmt"
	"log"
	"os"

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

	model := provider.Model("gpt-4o")

	fmt.Println("Sending streaming request...")

	result, err := model.DoStream(context.Background(), openaicompatible.StreamOptions{
		GenerateOptions: openaicompatible.GenerateOptions{
			Messages: []openaicompatible.Message{
				openaicompatible.UserMessage{
					Content: []openaicompatible.UserContent{
						openaicompatible.TextContent{Text: "Write a short story about a robot learning to paint."},
					},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error starting stream: %v", err)
	}

	fmt.Print("\nResponse: ")

	for part := range result.Stream {
		switch p := part.(type) {
		case openaicompatible.StreamTextDelta:
			fmt.Print(p.Text)
		case openaicompatible.StreamFinish:
			if p.Usage.InputTokens.Total != nil {
				fmt.Printf("\n\nUsage: %d input tokens, ", *p.Usage.InputTokens.Total)
			}
			if p.Usage.OutputTokens.Total != nil {
				fmt.Printf("%d output tokens\n", *p.Usage.OutputTokens.Total)
			}
		case openaicompatible.StreamError:
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
		}
	}

	fmt.Println("Stream finished.")
}
