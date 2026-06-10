package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/cohere"
)

const apiKey = "your-cohere-api-key"

func main() {
	provider := cohere.CreateCohere(cohere.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(cohere.ModelCommandA032025))

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{
				"type":        "string",
				"description": "A short title for the summary.",
			},
			"summary": map[string]any{
				"type":        "string",
				"description": "A two sentence summary.",
			},
			"keywords": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "string",
				},
			},
		},
		"required": []string{"title", "summary", "keywords"},
	}

	result, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
		Messages: []cohere.Message{
			cohere.UserMessage{
				Content: []cohere.UserContent{
					cohere.TextContent{Text: "Summarize why Go is popular for backend services."},
				},
			},
		},
		StructuredOutput: &cohere.StructuredOutput{
			Name:        "backend_summary",
			Description: "A structured summary for backend engineering notes.",
			Schema:      schema,
		},
		MaxOutputTokens: ptr(300),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

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
