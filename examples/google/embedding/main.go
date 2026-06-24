package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/google"
)

const apiKey = "your-google-api-key"

func main() {
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	model := provider.EmbeddingModel(google.ModelGeminiEmbedding2Preview)

	texts := []string{
		"Go is a statically typed, compiled programming language.",
		"Python is a dynamically typed, interpreted language.",
		"Rust focuses on memory safety without a garbage collector.",
	}

	dimensions := 768
	result, err := model.DoEmbed(context.Background(), google.EmbedOptions{
		Values: texts,
		ProviderOptions: google.ProviderOptions{
			"google": {
				"outputDimensionality": dimensions,
				"taskType":             "SEMANTIC_SIMILARITY",
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
