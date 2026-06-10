package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/cohere"
)

const apiKey = "your-cohere-api-key"

func main() {
	provider := cohere.CreateCohere(cohere.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.EmbeddingModel(string(cohere.ModelEmbedEnglishV30))

	texts := []string{
		"Go is a statically typed, compiled programming language.",
		"Python is a dynamically typed, interpreted language.",
		"Rust focuses on memory safety without a garbage collector.",
	}

	result, err := model.DoEmbed(context.Background(), cohere.EmbedOptions{
		Values: texts,
		ProviderOptions: cohere.ProviderOptions{
			"cohere": {
				"inputType": "search_document",
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
