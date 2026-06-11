package main

import (
	"context"
	"encoding/json"
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

	model := provider.Chat("openai/gpt-4o-mini")

	weatherTool := openrouter.Tool{
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

	messages := []openrouter.Message{
		openrouter.UserMessage{
			Content: []openrouter.UserContent{
				openrouter.TextContent{Text: "What is the weather like in Seattle, WA right now?"},
			},
		},
	}

	fmt.Println("Sending request with tool definition...")

	result1, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: messages,
		Tools:    []openrouter.Tool{weatherTool},
	})
	if err != nil {
		log.Fatalf("Error in first generation: %v", err)
	}

	// Collect assistant content for conversation history.
	var assistantContents []openrouter.AssistantContent
	for _, content := range result1.Content {
		if ac, ok := content.(openrouter.AssistantContent); ok {
			assistantContents = append(assistantContents, ac)
		}
	}
	messages = append(messages, openrouter.AssistantMessage{Content: assistantContents})

	// Find the tool call.
	var toolCall *openrouter.ToolCallContent
	for _, content := range result1.Content {
		if tc, ok := content.(openrouter.ToolCallContent); ok {
			toolCall = &tc
			break
		}
	}

	if toolCall == nil || toolCall.ToolName != "get_weather" {
		fmt.Println("Model did not call the expected tool.")
		for _, content := range result1.Content {
			if text, ok := content.(openrouter.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
		return
	}

	fmt.Printf("\nModel requested tool call: %s\n", toolCall.ToolName)

	inputBytes, _ := json.Marshal(toolCall.Input)
	fmt.Printf("Arguments: %s\n", string(inputBytes))

	var args struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal(inputBytes, &args); err != nil {
		log.Fatalf("Error parsing tool input: %v", err)
	}

	weatherResult := fmt.Sprintf("The current weather in %s is 62°F and overcast.", args.Location)
	fmt.Printf("Simulated result: %s\n", weatherResult)

	// Send the tool result back.
	messages = append(messages, openrouter.ToolMessage{
		Content: []openrouter.ToolContent{
			openrouter.ToolResultContent{
				ToolCallID: toolCall.ToolCallID,
				ToolName:   toolCall.ToolName,
				Output:     weatherResult,
			},
		},
	})

	fmt.Println("\nSending tool result back to the model...")

	result2, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: messages,
		Tools:    []openrouter.Tool{weatherTool},
	})
	if err != nil {
		log.Fatalf("Error in follow-up generation: %v", err)
	}

	fmt.Println("\nFinal response:")
	for _, content := range result2.Content {
		if text, ok := content.(openrouter.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
