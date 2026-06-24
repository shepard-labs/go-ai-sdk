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

	model := provider.Model(google.ModelGemini35Flash)
	result, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages: []google.Message{
			google.UserMessage{Content: []google.UserContent{
				google.TextContent{Text: "Describe this image in one short paragraph."},
				google.ImageContent{Source: google.ImageSource{
					Type: "url",
					URL:  "https://images.unsplash.com/photo-1780476895954-b934fb2e480b",
				}},
			}},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
}

func printText(contents []google.Content) {
	for _, content := range contents {
		if text, ok := content.(google.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
