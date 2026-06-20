// Command toolkit demonstrates the real-world use of the llm/toolkit package:
// constructing scoped file and shell toolkits, exposing their schemas to a live
// model, and letting an agent loop drive the model through actual tool calls to
// complete a task in a sandboxed workspace. The toolkit is both the tool set
// (Tools) and the dispatcher (Merge) the loop executes against.
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

	workspace, err := os.MkdirTemp("", "agent-workspace-*")
	if err != nil {
		log.Fatalf("temp dir: %v", err)
	}
	defer os.RemoveAll(workspace)

	// Seed the workspace with a file for the agent to work on.
	notesPath := workspace + "/notes.txt"
	const notes = "buy milk\nfix the login bug\nreply to Dana\nrenew domain\n"
	if err := os.WriteFile(notesPath, []byte(notes), 0o600); err != nil {
		log.Fatalf("seed file: %v", err)
	}

	// Files is scoped to Roots: path traversal and symlink escapes are rejected.
	// Shell only permits the listed base commands, each bounded by a timeout.
	files := toolkit.Files(toolkit.FilesConfig{Roots: []string{workspace}})
	shell := toolkit.Shell(toolkit.ShellConfig{Cwd: workspace, AllowedCmds: []string{"ls", "wc", "sort"}})

	// Tools flattens the schemas to advertise to the model; Merge builds the
	// dispatcher the loop calls when the model requests a tool.
	tools := toolkit.Tools(files, shell)
	dispatcher := toolkit.Merge(files, shell)

	// Run a real agent loop. The model reads notes.txt, sorts the lines, and
	// writes the result back — using the toolkit's tools to do actual work.
	transcript, _, err := llm.AgentLoopWithOptions(context.Background(), client, llm.GenerateOptions{
		System: "You are a file-tidying assistant. Use the available tools to complete the task, " +
			"then briefly confirm what you did.",
		Messages: []llm.Message{{
			Role: "user",
			Content: []llm.Content{llm.TextContent{Text: fmt.Sprintf(
				"Read %s, sort its lines alphabetically, and write the sorted lines back to the same file.",
				notesPath)}},
		}},
		Tools:     tools,
		MaxTokens: 2048,
	}, dispatcher, llm.AgentLoopOptions{MaxTurns: 10})
	if err != nil {
		log.Fatalf("agent loop: %v", err)
	}

	// Show the model's final words and the actual on-disk result.
	if last := transcript[len(transcript)-1]; last.Role == "assistant" {
		for _, content := range last.Content {
			if text, ok := content.(llm.TextContent); ok {
				fmt.Printf("Agent: %s\n", text.Text)
			}
		}
	}
	final, err := os.ReadFile(notesPath)
	if err != nil {
		log.Fatalf("read result: %v", err)
	}
	fmt.Printf("\nnotes.txt is now:\n%s", final)
}
