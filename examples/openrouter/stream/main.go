package main

import (
	"context"
	"fmt"
	"log"
	"os"

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

	model := provider.Chat("openai/gpt-4o-mini")

	fmt.Println("Sending streaming request...")

	result, err := model.DoStream(context.Background(), openrouter.StreamOptions{
		GenerateOptions: openrouter.GenerateOptions{
			Messages: []openrouter.Message{
				openrouter.UserMessage{
					Content: []openrouter.UserContent{
						openrouter.TextContent{Text: "Write a short story about a robot learning to paint."},
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
		case openrouter.StreamTextDelta:
			fmt.Print(p.Delta)
		case openrouter.StreamFinish:
			fmt.Printf("\n\nUsage: %d input tokens, %d output tokens\n",
				p.Usage.InputTokens, p.Usage.OutputTokens)
		case openrouter.StreamError:
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
		}
	}

	fmt.Println("Stream finished.")
}
