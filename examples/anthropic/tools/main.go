package main

import (
	"context"
	"encoding/json"
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

	model := provider.Model(string(anthropic.ModelClaudeSonnet46))

	// Define a custom tool for getting the weather
	weatherTool := anthropic.Tool{
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

	messages := []anthropic.Message{
		anthropic.UserMessage{
			Content: []anthropic.UserContent{
				anthropic.TextContent{Text: "What is the weather like in Seattle, WA right now?"},
			},
		},
	}

	fmt.Println("Sending request with tool definition...")

	// Make the first request
	result1, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages:  messages,
		Tools:     []anthropic.Tool{weatherTool},
		MaxTokens: 500,
	})
	if err != nil {
		log.Fatalf("Error in first generation: %v", err)
	}

	// Append assistant's response to the conversation history
	var assistantContents []anthropic.AssistantContent
	for _, content := range result1.Content {
		if ac, ok := content.(anthropic.AssistantContent); ok {
			assistantContents = append(assistantContents, ac)
		}
	}
	messages = append(messages, anthropic.AssistantMessage{Content: assistantContents})

	// Check if the model decided to call our tool
	var toolCall *anthropic.ToolCallContent
	for _, content := range result1.Content {
		if tc, ok := content.(anthropic.ToolCallContent); ok {
			toolCall = &tc
			break
		}
	}

	if toolCall != nil && toolCall.ToolName == "get_weather" {
		fmt.Printf("\nModel requested tool call: %s\n", toolCall.ToolName)
		fmt.Printf("Tool arguments: %s\n", string(toolCall.Input))

		// Parse the input
		var args struct {
			Location string `json:"location"`
		}
		if err := json.Unmarshal(toolCall.Input, &args); err != nil {
			log.Fatalf("Error parsing tool input: %v", err)
		}

		weatherResult := fmt.Sprintf("The current weather in %s is 65F and partly cloudy.", args.Location)
		fmt.Printf("Simulated tool result: %s\n", weatherResult)

		// Create a tool result message
		toolResultMessage := anthropic.UserMessage{
			Content: []anthropic.UserContent{
				anthropic.ToolResultContent{
					ToolCallID: toolCall.ToolCallID,
					Result: []anthropic.ToolResultPart{
						anthropic.ToolResultText{Text: weatherResult},
					},
				},
			},
		}

		// Append the tool result back to the messages list
		messages = append(messages, toolResultMessage)

		fmt.Println("\nSending the tool result back to the model...")

		// Make the follow-up request
		result2, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
			Messages:  messages,
			Tools:     []anthropic.Tool{weatherTool},
			MaxTokens: 500,
		})
		if err != nil {
			log.Fatalf("Error in follow-up generation: %v", err)
		}

		fmt.Println("\nFinal Response:")
		for _, content := range result2.Content {
			if text, ok := content.(anthropic.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
	} else {
		fmt.Println("The model did not call the expected tool.")
		for _, content := range result1.Content {
			if text, ok := content.(anthropic.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
	}
}
