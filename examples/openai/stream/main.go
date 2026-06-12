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

	model := provider.Chat("gpt-4o")

	fmt.Println("Sending streaming chat request...")

	result, err := model.DoStream(context.Background(), openai.StreamOptions{
		GenerateOptions: openai.GenerateOptions{
			Messages: []openai.Message{
				openai.UserMessage{Content: []openai.UserContent{
					openai.TextContent{Text: "Write a short story about a robot learning to paint."},
				}},
			},
			MaxOutputTokens: intPtr(300),
		},
	})
	if err != nil {
		log.Fatalf("Error starting stream: %v", err)
	}

	fmt.Print("\nResponse: ")
	for part := range result.Stream {
		switch p := part.(type) {
		case openai.StreamTextDelta:
			fmt.Print(p.Text)
		case openai.StreamFinish:
			if p.Usage.InputTokens.Total != nil && p.Usage.OutputTokens.Total != nil {
				fmt.Printf("\n\nUsage: %d input, %d output tokens\n",
					*p.Usage.InputTokens.Total, *p.Usage.OutputTokens.Total)
			}
		case openai.StreamError:
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
		}
	}
	fmt.Println("Stream finished.")
}

func intPtr(v int) *int { return &v }
