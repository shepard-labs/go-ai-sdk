package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title":   map[string]any{"type": "string"},
			"summary": map[string]any{"type": "string"},
			"keywords": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
			},
		},
		"required": []string{"title", "summary", "keywords"},
	}

	// Structured outputs work on both Chat and Responses; Responses is the default entrypoint.
	model := provider.Responses("gpt-4o")

	result, err := model.DoGenerate(context.Background(), openai.ResponsesGenerateOptions{
		Messages: []openai.Message{
			openai.UserMessage{Content: []openai.UserContent{
				openai.TextContent{Text: "Summarize why Go is popular for backend services."},
			}},
		},
		StructuredOutput: &openai.StructuredOutput{
			Name:        "backend_summary",
			Description: "Structured summary for engineering notes.",
			Schema:      schema,
		},
		MaxOutputTokens: intPtr(300),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

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
