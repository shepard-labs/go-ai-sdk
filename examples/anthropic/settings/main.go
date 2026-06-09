package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
		APIKey:  apiKey,
		BaseURL: "https://api.anthropic.com/v1",
		Headers: http.Header{
			"X-Example-App": {"go-ai-sdk-settings-example"},
		},
		Name: "anthropic-with-custom-settings",
		Retry: &anthropic.RetryOptions{
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

	model := provider.Model(string(anthropic.ModelClaudeSonnet46))
	result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.UserMessage{Content: []anthropic.UserContent{
				anthropic.TextContent{Text: "Reply with one sentence about custom client settings."},
			}},
		},
		MaxTokens: 100,
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("Provider name: %s\n\n", provider.Name())
	printText(result.Content)
}

func printText(contents []anthropic.Content) {
	for _, content := range contents {
		if text, ok := content.(anthropic.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
