package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
		BaseURL: "https://api.openai.com/v1",
		Name:    "openai",
		APIKey:  apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.EmbeddingModel("text-embedding-3-small")

	texts := []string{
		"Go is a statically typed, compiled programming language.",
		"Python is a dynamically typed, interpreted language.",
		"Rust focuses on memory safety without a garbage collector.",
	}

	// EmbeddingOptions (dimensions, user) can be passed via ProviderOptions
	// using the provider name as the namespace key.
	dimensions := 256
	result, err := model.DoEmbed(context.Background(), openaicompatible.EmbedOptions{
		Values: texts,
		ProviderOptions: openaicompatible.ProviderOptions{
			"openai": {
				"dimensions": dimensions,
			},
		},
	})
	if err != nil {
		log.Fatalf("Error generating embeddings: %v", err)
	}

	fmt.Printf("Generated %d embeddings\n", len(result.Embeddings))
	for i, emb := range result.Embeddings {
		fmt.Printf("  [%d] %q — %d dimensions, first value: %.6f\n",
			i, texts[i], len(emb), emb[0])
	}

	if result.Usage != nil {
		fmt.Printf("\nUsage: %d tokens\n", result.Usage.Tokens)
	}
}
