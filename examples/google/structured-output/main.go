package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/google"
)

const apiKey = "your-google-api-key"

func main() {
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(google.ModelGemini25Flash)

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

	result, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages: []google.Message{
			google.UserMessage{Content: []google.UserContent{
				google.TextContent{Text: "Summarize why Go is popular for backend services."},
			}},
		},
		StructuredOutput: &google.StructuredOutput{
			Name:        "backend_summary",
			Description: "A structured summary for backend engineering notes.",
			Schema:      schema,
		},
		MaxOutputTokens: intPtr(300),
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

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
