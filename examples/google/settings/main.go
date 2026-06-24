package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/shepard-labs/go-ai-sdk/google"
)

const apiKey = "your-google-api-key"

func main() {
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey:  apiKey,
		BaseURL: "https://generativelanguage.googleapis.com/v1beta",
		Headers: http.Header{
			"X-Example-App": {"go-ai-sdk-settings-example"},
		},
		QueryParams: map[string]string{
			"alt": "json",
		},
		Name: "google-with-custom-settings",
		Retry: &google.RetryOptions{
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

	model := provider.Model(google.ModelGemini35Flash)
	result, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages: []google.Message{
			google.UserMessage{Content: []google.UserContent{
				google.TextContent{Text: "Reply with one sentence about custom client settings."},
			}},
		},
		MaxOutputTokens: intPtr(1000),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("Provider name: %s\n\n", provider.Name())
	printText(result.Content)
}

func printText(contents []google.Content) {
	for _, content := range contents {
		if text, ok := content.(google.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
