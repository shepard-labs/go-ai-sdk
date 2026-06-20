// Command stream demonstrates streaming a completion with the llm package's
// Client.Stream: the model's output arrives incrementally as StreamPart values
// on a channel, so text can be rendered token-by-token instead of waiting for
// the full response. The channel emits exactly one StreamFinish (success) or
// StreamError (failure) before it closes.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"

	// Blank-import the provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	parts, err := client.Stream(context.Background(), llm.GenerateOptions{
		Messages: []llm.Message{{
			Role:    "user",
			Content: []llm.Content{llm.TextContent{Text: "Write a short haiku about streaming data."}},
		}},
		MaxTokens: 256,
	})
	if err != nil {
		// ErrStreamNotImplemented is returned here (not as a StreamError part)
		// when the provider adapter doesn't support streaming.
		log.Fatalf("stream: %v", err)
	}

	// Consume parts in order until the channel closes. Reasoning (thinking) text
	// and final-answer text arrive as separate delta streams.
	for part := range parts {
		switch p := part.(type) {
		case llm.StreamReasoningDelta:
			fmt.Fprint(os.Stderr, p.Text) // reasoning to stderr
		case llm.StreamTextDelta:
			fmt.Print(p.Text) // answer to stdout, as it arrives
		case llm.StreamFinish:
			fmt.Printf("\n\n[finished: %s — %d in / %d out tokens]\n",
				p.FinishReason, p.Usage.InputTokens, p.Usage.OutputTokens)
		case llm.StreamError:
			log.Fatalf("\nstream error: %v", p.Err)
		}
	}
}
