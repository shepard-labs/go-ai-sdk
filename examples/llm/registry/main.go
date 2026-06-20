// Command registry demonstrates selecting an LLM provider by name at runtime
// using the llm/registry package. Providers become available by blank-importing
// their adapter subpackages; switching providers is then a one-line string
// change.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"

	// Blank-import each provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/cohere"
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/google"
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/openai"
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/openaicompatible"
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/openrouter"
)

const apiKey = "sk-ant-api03-your-api-key"

func main() {
	// model format is "provider:model-id".
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	// For an OpenAI-compatible endpoint (Ollama, LM Studio), pass BaseURL:
	//   registry.NewClient("openaicompatible:llama3", registry.ProviderOptions{
	//       BaseURL: "http://localhost:11434/v1",
	//   })

	result, err := client.Generate(context.Background(), llm.GenerateOptions{
		Messages:  []llm.Message{{Role: "user", Content: []llm.Content{llm.TextContent{Text: "Say hello in one word."}}}},
		MaxTokens: 64,
	})
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	for _, content := range result.Content {
		if text, ok := content.(llm.TextContent); ok {
			fmt.Println(text.Text)
		}
	}
}
