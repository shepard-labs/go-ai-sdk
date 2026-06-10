package main

import (
	"context"
	"encoding/json"
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

	weatherTool := cohere.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a specific location.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city and state, e.g., San Francisco, CA",
				},
			},
			"required": []string{"location"},
		},
	}

	messages := []cohere.Message{
		cohere.UserMessage{
			Content: []cohere.UserContent{
				cohere.TextContent{Text: "What is the weather like in Seattle, WA right now?"},
			},
		},
	}

	fmt.Println("Sending request with tool definition...")

	result1, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
		Messages:        messages,
		Tools:           []cohere.Tool{weatherTool},
		MaxOutputTokens: ptr(500),
	})
	if err != nil {
		log.Fatalf("Error in first generation: %v", err)
	}

	var assistantContents []cohere.AssistantContent
	for _, content := range result1.Content {
		if ac, ok := content.(cohere.AssistantContent); ok {
			assistantContents = append(assistantContents, ac)
		}
	}
	messages = append(messages, cohere.AssistantMessage{Content: assistantContents})

	var toolCall *cohere.ToolCallContent
	for _, content := range result1.Content {
		if tc, ok := content.(cohere.ToolCallContent); ok {
			toolCall = &tc
			break
		}
	}

	if toolCall != nil && toolCall.ToolName == "get_weather" {
		fmt.Printf("\nModel requested tool call: %s\n", toolCall.ToolName)
		fmt.Printf("Tool arguments: %s\n", string(toolCall.Input))

		var args struct {
			Location string `json:"location"`
		}
		if err := json.Unmarshal(toolCall.Input, &args); err != nil {
			log.Fatalf("Error parsing tool input: %v", err)
		}

		weatherResult := fmt.Sprintf("The current weather in %s is 65F and partly cloudy.", args.Location)
		fmt.Printf("Simulated tool result: %s\n", weatherResult)

		toolResultMessage := cohere.ToolMessage{
			Content: []cohere.ToolContent{
				cohere.ToolResultContent{
					ToolCallID: toolCall.ToolCallID,
					Output: cohere.ToolResultOutput{
						Type:  "text",
						Value: weatherResult,
					},
				},
			},
		}

		messages = append(messages, toolResultMessage)

		fmt.Println("\nSending the tool result back to the model...")

		result2, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
			Messages:        messages,
			Tools:           []cohere.Tool{weatherTool},
			MaxOutputTokens: ptr(500),
		})
		if err != nil {
			log.Fatalf("Error in follow-up generation: %v", err)
		}

		fmt.Println("\nFinal Response:")
		for _, content := range result2.Content {
			if text, ok := content.(cohere.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
	} else {
		fmt.Println("The model did not call the expected tool.")
		for _, content := range result1.Content {
			if text, ok := content.(cohere.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
	}
}

func ptr(v int) *int { return &v }
