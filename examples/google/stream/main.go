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

	model := provider.Model(google.ModelGemini35Flash)

	fmt.Println("Sending streaming request to Gemini...")

	result, err := model.DoStream(context.Background(), google.StreamOptions{
		GenerateOptions: google.GenerateOptions{
			Messages: []google.Message{
				google.UserMessage{Content: []google.UserContent{
					google.TextContent{Text: "Write a short story about a robot learning to paint, stream the response."},
				}},
			},
			MaxOutputTokens: intPtr(3000),
		},
	})
	if err != nil {
		log.Fatalf("Error starting stream: %v", err)
	}

	fmt.Print("\nResponse: ")

	// Process the streaming chunks as they arrive.
	for part := range result.Stream {
		switch p := part.(type) {
		case google.StreamTextDelta:
			fmt.Print(p.Text)
		case google.StreamError:
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
		}
	}
	fmt.Println("\n\nStream finished.")
}

func intPtr(v int) *int { return &v }
