// Command store demonstrates the real-world use of the llm/store package:
// running a live agent turn, persisting the resulting transcript, then resuming
// the same run in a fresh state from the store and continuing the conversation.
// This is the durable multi-turn agent pattern — the store carries the run
// across what could be separate processes or requests. The file backend is used
// here; postgres, gcs, and r2 submodule backends are drop-in replacements.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
	"github.com/shepard-labs/go-ai-sdk/llm/store"
	"github.com/shepard-labs/go-ai-sdk/llm/store/file"

	// Blank-import the provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	ctx := context.Background()
	const runID = "demo-run"

	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	// File backend: one JSON file per run ID under a directory.
	dir, err := os.MkdirTemp("", "agent-runs-*")
	if err != nil {
		log.Fatalf("temp dir: %v", err)
	}
	defer os.RemoveAll(dir)
	rs, err := file.New(dir)
	if err != nil {
		log.Fatalf("file store: %v", err)
	}

	const system = "You are a concise assistant. Answer in one short sentence."

	// Turn 1: ask a question, then persist the whole transcript so the run can be
	// resumed later (in this process, another process, or a later request).
	turn(ctx, client, rs, runID, system, "My name is Sam and I'm planning a trip to Kyoto.")

	// Turn 2: a fresh load from the store rehydrates the prior messages, so the
	// model still has the context from turn 1 — note we never re-send it.
	turn(ctx, client, rs, runID, system, "What's my name, and where am I going?")
}

// turn loads the run, appends a user message, generates one reply, and saves the
// updated transcript back to the store.
func turn(ctx context.Context, client llm.Client, rs store.RunStore, runID, system, userText string) {
	// Load existing state, or start a new run. Load returns (nil, nil) when the
	// run does not exist yet.
	state, err := rs.Load(ctx, runID)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	if state == nil {
		state = &store.RunState{ID: runID}
	}

	state.Messages = append(state.Messages, llm.Message{
		Role:    "user",
		Content: []llm.Content{llm.TextContent{Text: userText}},
	})

	result, err := client.Generate(ctx, llm.GenerateOptions{
		System:    system,
		Messages:  state.Messages,
		MaxTokens: 256,
	})
	if err != nil {
		log.Fatalf("generate: %v", err)
	}

	// Append the assistant reply and persist the run for the next turn.
	state.Messages = append(state.Messages, llm.Message{Role: "assistant", Content: result.Content})
	if err := rs.Save(ctx, state); err != nil {
		log.Fatalf("save: %v", err)
	}

	fmt.Printf("User:  %s\n", userText)
	for _, content := range result.Content {
		if text, ok := content.(llm.TextContent); ok {
			fmt.Printf("Agent: %s\n\n", text.Text)
		}
	}
}
