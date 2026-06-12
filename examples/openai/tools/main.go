package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openai"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.Responses("gpt-4o")

	weatherTool := openai.Tool{
		Type:        "function",
		Name:        "get_weather",
		Description: "Get the current weather for a specific location.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "City and region, e.g. Seattle, WA",
				},
			},
			"required": []string{"location"},
		},
	}

	messages := []openai.Message{
		openai.UserMessage{Content: []openai.UserContent{
			openai.TextContent{Text: "What is the weather like in Seattle, WA right now?"},
		}},
	}

	fmt.Println("Sending request with tool definition...")

	result1, err := model.DoGenerate(context.Background(), openai.ResponsesGenerateOptions{
		Messages:        messages,
		Tools:           []openai.Tool{weatherTool},
		MaxOutputTokens: intPtr(500),
	})
	if err != nil {
		log.Fatalf("Error in first generation: %v", err)
	}

	var assistantContents []openai.AssistantContent
	for _, content := range result1.Content {
		if ac, ok := content.(openai.AssistantContent); ok {
			assistantContents = append(assistantContents, ac)
		}
	}
	messages = append(messages, openai.AssistantMessage{Content: assistantContents})

	var toolCall *openai.ToolCallContent
	for _, content := range result1.Content {
		if tc, ok := content.(openai.ToolCallContent); ok {
			toolCall = &tc
			break
		}
	}

	if toolCall == nil || toolCall.ToolName != "get_weather" {
		fmt.Println("The model did not call the expected tool.")
		printText(result1.Content)
		return
	}

	fmt.Printf("\nModel requested tool call: %s\n", toolCall.ToolName)
	fmt.Printf("Tool arguments: %s\n", string(toolCall.Input))

	var args struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal(toolCall.Input, &args); err != nil {
		log.Fatalf("Error parsing tool input: %v", err)
	}

	weatherResult := fmt.Sprintf("The current weather in %s is 62°F and overcast.", args.Location)
	fmt.Printf("Simulated tool result: %s\n", weatherResult)

	messages = append(messages, openai.ToolMessage{
		Content: []openai.ToolContent{
			openai.ToolResultContent{
				ToolResultContent: openaicompatible.ToolResultContent{
					ToolCallID: toolCall.ToolCallID,
					Output: openaicompatible.ToolResultOutput{
						Type:  "text",
						Value: weatherResult,
					},
				},
			},
		},
	})

	fmt.Println("\nSending the tool result back to the model...")

	result2, err := model.DoGenerate(context.Background(), openai.ResponsesGenerateOptions{
		Messages:        messages,
		Tools:           []openai.Tool{weatherTool},
		MaxOutputTokens: intPtr(500),
	})
	if err != nil {
		log.Fatalf("Error in follow-up generation: %v", err)
	}

	fmt.Println("\nFinal response:")
	printText(result2.Content)
}

func printText(contents []openai.Content) {
	for _, content := range contents {
		if text, ok := content.(openai.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}

func intPtr(v int) *int { return &v }
