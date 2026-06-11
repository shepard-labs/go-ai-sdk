package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

const apiKey = "your-openrouter-api-key"

// stdoutLogger implements openrouter.Logger and prints to stdout.
type stdoutLogger struct{}

func (l stdoutLogger) Debug(msg string, kv ...any) { fmt.Printf("[DEBUG] %s %v\n", msg, kv) }
func (l stdoutLogger) Info(msg string, kv ...any)  { fmt.Printf("[INFO]  %s %v\n", msg, kv) }
func (l stdoutLogger) Warn(msg string, kv ...any)  { fmt.Printf("[WARN]  %s %v\n", msg, kv) }
func (l stdoutLogger) Error(msg string, kv ...any) { fmt.Printf("[ERROR] %s %v\n", msg, kv) }

func main() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
		APIKey: apiKey,
		// AppName and AppURL are sent as X-OpenRouter-Title and HTTP-Referer headers,
		// used by OpenRouter for analytics and ranking.
		AppName: "go-ai-sdk-settings-example",
		AppURL:  "https://example.com",
		Headers: http.Header{
			"X-Custom-Header": {"my-value"},
		},
		Retry: &openrouter.RetryOptions{
			MaxRetries: 3,
			BaseDelay:  250 * time.Millisecond,
			MaxDelay:   2 * time.Second,
			Jitter:     true,
		},
		MaxResponseBodyBytes:  64 << 20,
		MaxErrorResponseBytes: 2 << 20,
		Logger:                stdoutLogger{},
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Chat("openai/gpt-4o-mini")

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "Reply with one sentence about custom provider settings."},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("\nProvider: %s\n\n", provider.Name())
	for _, content := range result.Content {
		if text, ok := content.(openrouter.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
		result.Usage.InputTokens, result.Usage.OutputTokens)
}
