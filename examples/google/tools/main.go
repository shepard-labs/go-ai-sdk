package main

import (
	"context"
	"encoding/json"
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

	model := provider.Model(google.ModelGemini35Flash)

	// Define a custom tool for getting the weather.
	weatherTool := google.Tool{
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

	messages := []google.Message{
		google.UserMessage{Content: []google.UserContent{
			google.TextContent{Text: "What is the weather like in San Francisco, CA right now?"},
		}},
	}

	fmt.Println("Sending request with tool definition...")

	// First turn: model decides whether to call the tool.
	result1, err := model.DoGenerate(context.Background(), google.GenerateOptions{
		Messages:        messages,
		Tools:           []google.Tool{weatherTool},
		MaxOutputTokens: intPtr(500),
	})
	if err != nil {
		log.Fatalf("Error in first generation: %v", err)
	}

	// Replay the assistant turn into the conversation history.
	var assistantContents []google.AssistantContent
	for _, content := range result1.Content {
		if ac, ok := content.(google.AssistantContent); ok {
			assistantContents = append(assistantContents, ac)
		}
	}
	messages = append(messages, google.AssistantMessage{Content: assistantContents})

	var toolCall *google.ToolCallContent
	for _, content := range result1.Content {
		if tc, ok := content.(google.ToolCallContent); ok {
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

		// Send the tool result back to the model.
		toolResultMessage := google.ToolMessage{
			Content: []google.ToolContent{
				google.ToolResultContent{
					ToolCallID: toolCall.ToolCallID,
					Output: google.ToolResultOutput{
						Type:  "text",
						Value: weatherResult,
					},
				},
			},
		}
		messages = append(messages, toolResultMessage)

		fmt.Println("\nSending the tool result back to the model...")

		result2, err := model.DoGenerate(context.Background(), google.GenerateOptions{
			Messages:        messages,
			Tools:           []google.Tool{weatherTool},
			MaxOutputTokens: intPtr(500),
		})
		if err != nil {
			log.Fatalf("Error in follow-up generation: %v", err)
		}

		fmt.Println("\nFinal Response:")
		printText(result2.Content)
	} else {
		fmt.Println("The model did not call the expected tool.")
		printText(result1.Content)
	}
}

func printText(contents []google.Content) {
	for _, content := range contents {
		if text, ok := content.(google.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
