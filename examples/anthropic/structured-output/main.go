package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model(string(anthropic.ModelClaudeSonnet46), anthropic.ModelOptions{
		StructuredOutputMode: anthropic.StructuredOutputModeAuto,
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

	result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.UserMessage{Content: []anthropic.UserContent{
				anthropic.TextContent{Text: "Summarize why Go is popular for backend services."},
			}},
		},
		StructuredOutput: &anthropic.StructuredOutput{
			Name:        "backend_summary",
			Description: "A structured summary for backend engineering notes.",
			Schema:      schema,
		},
		MaxTokens: 300,
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	printText(result.Content)
}

func printText(contents []anthropic.Content) {
	for _, content := range contents {
		if text, ok := content.(anthropic.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
