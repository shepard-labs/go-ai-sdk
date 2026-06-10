package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/shepard-labs/go-ai-sdk/cohere"
)

const apiKey = "your-cohere-api-key"

func main() {
	provider := cohere.CreateCohere(cohere.ProviderSettings{
		APIKey:  apiKey,
		BaseURL: "https://api.cohere.com/v2",
		Headers: http.Header{
			"X-Example-App": {"go-ai-sdk-settings-example"},
		},
		Retry: &cohere.RetryOptions{
			MaxRetries: 3,
			BaseDelay:  250 * time.Millisecond,
			MaxDelay:   2 * time.Second,
			Jitter:     true,
		},
		MaxResponseBodyBytes:  64 << 20,
		MaxErrorResponseBytes: 2 << 20,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(cohere.ModelCommandA032025))
	result, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
		Messages: []cohere.Message{
			cohere.UserMessage{
				Content: []cohere.UserContent{
					cohere.TextContent{Text: "Reply with one sentence about custom client settings."},
				},
			},
		},
		MaxOutputTokens: ptr(100),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("Provider name: %s\n\n", provider.Name())
	printText(result.Content)
}

func printText(contents []cohere.Content) {
	for _, content := range contents {
		if text, ok := content.(cohere.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func ptr(v int) *int { return &v }
