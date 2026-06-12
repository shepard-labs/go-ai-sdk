package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/shepard-labs/go-ai-sdk"
	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/openai"
)

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	r := ai.CreateRouter(ai.RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: envOrDefault("ANTHROPIC_API_KEY", "sk-ant-api03-your-api-key"),
		}),
		OpenAI: openai.CreateOpenAI(openai.ProviderSettings{
			APIKey: envOrDefault("OPENAI_API_KEY", "sk-your-openai-key"),
		}),
		Catalog: ai.ProviderCatalog{
			"anthropic": {
				"claude-haiku-4-5-20251001",
				"claude-sonnet-4-6",
				"claude-opus-4-8",
			},
			"openai": {
				"gpt-5",
				"gpt-5-mini",
				"gpt-4.1",
			},
		},
	})
	if err := r.Err(); err != nil {
		log.Fatalf("Error creating router: %v", err)
	}

	res, sel, err := r.Generate(context.Background(), ai.RouterOptions{
		Prompt: "Write a haiku about programming in Go.",
	})
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	fmt.Printf("Routed to: %s / %s\n", sel.Provider, sel.Model)
	fmt.Printf("Reason:    %s\n\n", sel.Reason)

	fmt.Println("Response:")
	switch res.Kind {
	case ai.ResultKindAnthropic:
		for _, content := range res.Anthropic.Value.Content {
			if text, ok := content.(anthropic.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
		fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
			res.Anthropic.Value.Usage.InputTokens.Total,
			res.Anthropic.Value.Usage.OutputTokens.Total,
		)
	case ai.ResultKindOpenAI:
		for _, content := range res.OpenAI.Value.Content {
			if text, ok := content.(openai.TextContent); ok {
				fmt.Println(text.Text)
			}
		}
		if res.OpenAI.Value.Usage.InputTokens.Total != nil {
			fmt.Printf("\nUsage: %d input tokens, %d output tokens\n",
				*res.OpenAI.Value.Usage.InputTokens.Total,
				*res.OpenAI.Value.Usage.OutputTokens.Total,
			)
		}
	}
}
