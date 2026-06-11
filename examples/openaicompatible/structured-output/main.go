package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

func main() {
	// SupportsStructuredOutputs enables strict JSON schema enforcement on
	// providers that support it (e.g. OpenAI gpt-4o and later).
	provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
		BaseURL:                   "https://api.openai.com/v1",
		Name:                      "openai",
		APIKey:                    apiKey,
		SupportsStructuredOutputs: true,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model("gpt-4o")

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{
				"type":        "string",
				"description": "A short title for the summary.",
			},
			"summary": map[string]any{
				"type":        "string",
				"description": "A two-sentence summary.",
			},
			"keywords": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
			},
		},
		"required":             []string{"title", "summary", "keywords"},
		"additionalProperties": false,
	}

	result, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{
			openaicompatible.UserMessage{
				Content: []openaicompatible.UserContent{
					openaicompatible.TextContent{Text: "Summarize why Go is popular for backend services."},
				},
			},
		},
		StructuredOutput: &openaicompatible.StructuredOutput{
			Name:        "backend_summary",
			Description: "Structured summary for backend engineering notes.",
			Schema:      schema,
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		if text, ok := content.(openaicompatible.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
