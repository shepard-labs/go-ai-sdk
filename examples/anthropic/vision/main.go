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

	model := provider.Model(string(anthropic.ModelClaudeSonnet46))
	result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.UserMessage{Content: []anthropic.UserContent{
				anthropic.TextContent{Text: "Describe this image in one short paragraph."},
				anthropic.ImageContent{Source: anthropic.ImageSource{
					Type: "url",
					URL:  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
				}},
			}},
		},
		MaxTokens: 200,
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
