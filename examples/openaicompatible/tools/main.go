package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
		BaseURL: "https://api.openai.com/v1",
		Name:    "openai",
		APIKey:  apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Model("gpt-4o")

	weatherTool := openaicompatible.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a specific location.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
			},
			"required": []string{"location"},
		},
	}

	messages := []openaicompatible.Message{
		openaicompatible.UserMessage{
			Content: []openaicompatible.UserContent{
				openaicompatible.TextContent{Text: "What is the weather like in Seattle, WA right now?"},
			},
		},
	}

	fmt.Println("Sending request with tool definition...")

	result1, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: messages,
		Tools:    []openaicompatible.Tool{weatherTool},
	})
	if err != nil {
		log.Fatalf("Error in first generation: %v", err)
	}

	// Collect assistant content for history.
	var assistantContents []openaicompatible.AssistantContent
	for _, content := range result1.Content {
		if ac, ok := content.(openaicompatible.AssistantContent); ok {
			assistantContents = append(assistantContents, ac)
		}
	}
	messages = append(messages, openaicompatible.AssistantMessage{Content: assistantContents})

	// Find the tool call.
	var toolCall *openaicompatible.ToolCallContent
	for _, content := range result1.Content {
		if tc, ok := content.(openaicompatible.ToolCallContent); ok {
			toolCall = &tc
			break
		}
	}

	if toolCall == nil || toolCall.ToolName != "get_weather" {
		fmt.Println("Model did not call the expected tool.")
		for _, content := range result1.Content {
			if text, ok := content.(openaicompatible.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
		return
	}

	fmt.Printf("\nModel requested tool call: %s\n", toolCall.ToolName)
	fmt.Printf("Arguments: %s\n", string(toolCall.Input))

	var args struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal(toolCall.Input, &args); err != nil {
		log.Fatalf("Error parsing tool input: %v", err)
	}

	weatherResult := fmt.Sprintf("The current weather in %s is 62°F and overcast.", args.Location)
	fmt.Printf("Simulated result: %s\n", weatherResult)

	// Send the tool result back.
	messages = append(messages, openaicompatible.ToolMessage{
		Content: []openaicompatible.ToolContent{
			openaicompatible.ToolResultContent{
				ToolCallID: toolCall.ToolCallID,
				Output: openaicompatible.ToolResultOutput{
					Type:  "text",
					Value: weatherResult,
				},
			},
		},
	})

	fmt.Println("\nSending tool result back to the model...")

	result2, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: messages,
		Tools:    []openaicompatible.Tool{weatherTool},
	})
	if err != nil {
		log.Fatalf("Error in follow-up generation: %v", err)
	}

	fmt.Println("\nFinal response:")
	for _, content := range result2.Content {
		if text, ok := content.(openaicompatible.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
