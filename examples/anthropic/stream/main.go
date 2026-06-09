package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(anthropic.ModelClaudeSonnet46))

	fmt.Println("Sending streaming request to Claude...")

	result, err := model.DoStream(context.Background(), anthropic.StreamOptions{
		Messages: []anthropic.Message{
			anthropic.UserMessage{
				Content: []anthropic.UserContent{
					anthropic.TextContent{Text: "Write a short story about a robot learning to paint, stream the response."},
				},
			},
		},
		MaxTokens: 300,
	})
	if err != nil {
		log.Fatalf("Error starting stream: %v", err)
	}

	fmt.Print("\nResponse: ")

	// Process the streaming chunks as they arrive
	for part := range result.Stream {
		switch p := part.(type) {
		case anthropic.StreamTextDelta:
			fmt.Print(p.Text)
		case anthropic.StreamError:
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
		}
	}
	fmt.Println("\n\nStream finished.")
}
