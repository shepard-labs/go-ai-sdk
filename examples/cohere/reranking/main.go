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

	model := provider.RerankingModel(string(cohere.ModelRerankV35))

	query := "What programming language is best for systems programming?"
	documents := cohere.TextDocuments(
		"Go is a statically typed language designed for building scalable systems.",
		"Python is great for data science and machine learning workflows.",
		"Rust provides memory safety guarantees ideal for systems programming.",
		"JavaScript is widely used for web frontend and Node.js backends.",
		"C++ offers fine-grained control over memory and is used in OS and game development.",
	)

	topN := 3
	result, err := model.DoRerank(context.Background(), cohere.RerankOptions{
		Query:     query,
		Documents: documents,
		TopN:      &topN,
	})
	if err != nil {
		log.Fatalf("Error reranking documents: %v", err)
	}

	fmt.Printf("Query: %q\n\nTop %d results:\n", query, topN)
	for rank, doc := range result.Ranking {
		fmt.Printf("  #%d  index=%d  score=%.4f  %q\n",
			rank+1, doc.Index, doc.RelevanceScore,
			documents.Values[doc.Index])
	}
}
