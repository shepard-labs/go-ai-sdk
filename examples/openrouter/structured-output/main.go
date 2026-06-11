package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

const apiKey = "your-openrouter-api-key"

func main() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	// StructuredOutputs.Strict enables strict JSON schema enforcement.
	strict := true
	model := provider.Chat("openai/gpt-4o-mini", openrouter.ChatOptions{
		StructuredOutputs: &openrouter.StructuredOutputsOptions{Strict: &strict},
	})

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

	result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{
				Content: []openrouter.UserContent{
					openrouter.TextContent{Text: "Summarize why Go is popular for backend services."},
				},
			},
		},
		ResponseFormat: &openrouter.ResponseFormat{
			Type:        "json",
			Schema:      schema,
			Name:        "backend_summary",
			Description: "Structured summary for backend engineering notes.",
		},
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	for _, content := range result.Content {
		if text, ok := content.(openrouter.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
