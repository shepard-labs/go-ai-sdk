package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

// stdoutLogger implements openaicompatible.Logger and prints to stdout.
type stdoutLogger struct{}

func (l stdoutLogger) Debug(msg string, kv ...any) { fmt.Printf("[DEBUG] %s %v\n", msg, kv) }
func (l stdoutLogger) Info(msg string, kv ...any)  { fmt.Printf("[INFO]  %s %v\n", msg, kv) }
func (l stdoutLogger) Warn(msg string, kv ...any)  { fmt.Printf("[WARN]  %s %v\n", msg, kv) }
func (l stdoutLogger) Error(msg string, kv ...any) { fmt.Printf("[ERROR] %s %v\n", msg, kv) }

func main() {
	provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
		BaseURL: "https://api.openai.com/v1",
		Name:    "openai",
		APIKey:  apiKey,
		Headers: http.Header{
			"X-App-Name": {"go-ai-sdk-settings-example"},
		},
		QueryParams: map[string]string{
			"example_param": "value",
		},
		Retry: &openaicompatible.RetryOptions{
			MaxRetries: 3,
			BaseDelay:  250 * time.Millisecond,
			MaxDelay:   2 * time.Second,
			Jitter:     true,
		},
		MaxResponseBodyBytes:  64 << 20,
		MaxErrorResponseBytes: 2 << 20,
		IncludeUsage:          true,
		Logger:                stdoutLogger{},
		// TransformRequestBody lets you inject or rewrite any field in the
		// outgoing JSON body before it is sent to the API.
		TransformRequestBody: func(body map[string]any) map[string]any {
			body["user"] = "example-user-123"
			return body
		},
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model("gpt-4o")

	result, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{
			openaicompatible.UserMessage{
				Content: []openaicompatible.UserContent{
					openaicompatible.TextContent{Text: "Reply with one sentence about custom provider settings."},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("\nProvider: %s\n\n", provider.Name())
	for _, content := range result.Content {
		if text, ok := content.(openaicompatible.TextContent); ok {
			fmt.Println(text.Text)
		}
	}

	if result.Usage.InputTokens.Total != nil {
		fmt.Printf("\nUsage: %d input tokens, ", *result.Usage.InputTokens.Total)
	}
	if result.Usage.OutputTokens.Total != nil {
		fmt.Printf("%d output tokens\n", *result.Usage.OutputTokens.Total)
	}
}
