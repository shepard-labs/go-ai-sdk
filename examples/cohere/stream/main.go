package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/cohere"
)

const apiKey = "your-cohere-api-key"

func main() {
	provider := cohere.CreateCohere(cohere.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(cohere.ModelCommandA032025))

	result, err := model.DoStream(context.Background(), cohere.StreamOptions{
		GenerateOptions: cohere.GenerateOptions{
			Messages: []cohere.Message{
				cohere.UserMessage{
					Content: []cohere.UserContent{
						cohere.TextContent{Text: "Write a short story about a robot learning to paint, stream the response."},
					},
				},
			},
			MaxOutputTokens: ptr(300),
		},
	})
	if err != nil {
		log.Fatalf("Error starting stream: %v", err)
	}

	fmt.Print("\nResponse: ")
	for part := range result.Stream {
		switch p := part.(type) {
		case cohere.StreamTextDelta:
			fmt.Print(p.Text)
		case cohere.StreamError:
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
		}
	}
	fmt.Println("\n\nStream finished.")
}

func ptr(v int) *int { return &v }
