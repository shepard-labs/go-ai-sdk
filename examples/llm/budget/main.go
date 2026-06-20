// Command budget demonstrates AgentLoopOptions.TokenBudget: bounding how much
// conversation history is sent on each turn of a long tool-calling loop. As the
// transcript grows past the budget, the loop drops the oldest completed
// tool_use/tool_result message pairs before the next Generate call, keeping the
// request within a token ceiling. TokenCounter customizes how tokens are
// measured; omit it to use the built-in estimator (provider usage, else chars/4).
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
	"github.com/shepard-labs/go-ai-sdk/llm/toolkit"

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

	// A workspace with several files so the model makes multiple read_file calls,
	// growing the transcript turn over turn.
	workspace, err := os.MkdirTemp("", "agent-budget-*")
	if err != nil {
		log.Fatalf("temp dir: %v", err)
	}
	defer os.RemoveAll(workspace)
	for name, body := range map[string]string{
		"a.txt": "alpha\n", "b.txt": "bravo\n", "c.txt": "charlie\n", "d.txt": "delta\n",
	} {
		if err := os.WriteFile(workspace+"/"+name, []byte(body), 0o600); err != nil {
			log.Fatalf("seed %s: %v", name, err)
		}
	}

	files := toolkit.Files(toolkit.FilesConfig{Roots: []string{workspace}})

	const budget = 400 // small ceiling so trimming engages mid-run

	// A deterministic ~4-chars-per-token counter, so the demo doesn't depend on
	// provider-reported usage. It counts every content kind the transcript can
	// hold — including tool_use inputs and tool_result text — because those are
	// exactly the message pairs the loop drops to stay under budget. Keep this
	// function pure: the loop calls it repeatedly while trimming a single turn.
	counter := func(msgs []llm.Message) int {
		chars := 0
		for _, m := range msgs {
			for _, c := range m.Content {
				switch v := c.(type) {
				case llm.TextContent:
					chars += len(v.Text)
				case llm.ToolUseContent:
					chars += len(v.Name) + len(v.Input)
				case llm.ToolResultContent:
					chars += len(v.Text)
				}
			}
		}
		return chars / 4
	}

	transcript, _, err := llm.AgentLoopWithOptions(context.Background(), client, llm.GenerateOptions{
		System: "You inspect files one at a time using read_file, then summarize.",
		Messages: []llm.Message{{
			Role: "user",
			Content: []llm.Content{llm.TextContent{Text: fmt.Sprintf(
				"List the files in %s with list_dir, then read each one and tell me what each contains.",
				workspace)}},
		}},
		Tools:     files.Tools(),
		MaxTokens: 1024,
	}, files, llm.AgentLoopOptions{
		MaxTurns:     12,
		TokenBudget:  budget,
		TokenCounter: counter,
	})
	if err != nil {
		log.Fatalf("agent loop: %v", err)
	}

	// The returned transcript is the final (already-trimmed) history. Its size
	// staying near the budget — despite many tool round-trips — is the budget at
	// work: older tool_use/tool_result pairs were dropped along the way.
	fmt.Printf("loop completed; final history is ~%d tokens against a %d-token budget\n", counter(transcript), budget)
}
