package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

type stdoutLogger struct{}

func (stdoutLogger) Debug(msg string, kv ...any) { fmt.Printf("[DEBUG] %s %v\n", msg, kv) }
func (stdoutLogger) Info(msg string, kv ...any)  { fmt.Printf("[INFO]  %s %v\n", msg, kv) }
func (stdoutLogger) Warn(msg string, kv ...any)  { fmt.Printf("[WARN]  %s %v\n", msg, kv) }
func (stdoutLogger) Error(msg string, kv ...any) { fmt.Printf("[ERROR] %s %v\n", msg, kv) }

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{
		APIKey:       apiKey,
		Organization: "org_example",
		Project:      "proj_example",
		BaseURL:      "https://api.openai.com/v1",
		Name:         "openai-with-custom-settings",
		Headers: http.Header{
			"X-Example-App": {"go-ai-sdk-openai-settings"},
		},
		Retry: &openai.RetryOptions{
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

	model := provider.Chat("gpt-4o")
	result, err := model.DoGenerate(context.Background(), openai.GenerateOptions{
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "Reply with one sentence about custom OpenAI client settings."},
			}},
		},
		MaxOutputTokens: intPtr(80),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("Provider name: %s\n\n", provider.Name())
	printText(result.Content)
}

func printText(contents []openai.Content) {
	for _, content := range contents {
		if text, ok := content.(openai.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
