package main

import (
	"context"
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

	model := provider.Embedding("openai/text-embedding-3-small")

	texts := []string{
		"Go is a statically typed, compiled programming language.",
		"Python is a dynamically typed, interpreted language.",
		"Rust focuses on memory safety without a garbage collector.",
	}

	result, err := model.DoEmbed(context.Background(), openrouter.EmbedOptions{
		Values: texts,
	})
	if err != nil {
		log.Fatalf("Error generating embeddings: %v", err)
	}

	fmt.Printf("Generated %d embeddings\n", len(result.Embeddings))
	for i, emb := range result.Embeddings {
		fmt.Printf("  [%d] %q — %d dimensions, first value: %.6f\n",
			i, texts[i], len(emb), emb[0])
	}

	fmt.Printf("\nUsage: %d tokens\n", result.Usage.Tokens)
}
