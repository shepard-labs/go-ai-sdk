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

	env, sel, err := r.Stream(context.Background(), ai.RouterOptions{
		Prompt: "Stream a short paragraph about programming in Go.",
	})
	if err != nil {
		log.Fatalf("Error starting stream: %v", err)
	}

	fmt.Printf("Routed to: %s / %s\n", sel.Provider, sel.Model)
	fmt.Printf("Reason:    %s\n\n", sel.Reason)
	fmt.Print("Response: ")

	// The underlying stream channel is provider-specific, so switch
	// on the envelope Kind and iterate the matching channel. Each
	// provider's StreamPart union has its own set of delta types
	// (anthropic.StreamTextDelta vs openai.StreamTextDelta).
	switch env.Kind {
	case ai.StreamKindAnthropic:
		for part := range env.Anthropic.Stream {
			switch p := part.(type) {
			case anthropic.StreamTextDelta:
				fmt.Print(p.Text)
			case anthropic.StreamError:
				fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
			}
		}
	case ai.StreamKindOpenAI:
		for part := range env.OpenAI.Stream {
			switch p := part.(type) {
			case openai.StreamTextDelta:
				fmt.Print(p.Text)
			case openai.StreamError:
				fmt.Fprintf(os.Stderr, "\nStream error: %v\n", p.Err)
			}
		}
	default:
		log.Fatalf("unknown stream kind: %s", env.Kind)
	}
	fmt.Println("\n\nStream finished.")
}
