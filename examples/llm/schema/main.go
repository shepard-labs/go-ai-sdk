// Command schema demonstrates the real-world use of the llm/schema package:
// generating an llm.Tool from a Go struct, handing that tool to a live model as
// the agent loop's terminal "submit result" tool, and decoding the model's tool
// call straight back into the typed struct. This is the canonical way to get
// validated, structured output out of an LLM without hand-authoring JSON Schema.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
	"github.com/shepard-labs/go-ai-sdk/llm/schema"

	// Blank-import the provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

// Sentiment is the typed output we want from the model. Struct tags drive the
// generated JSON Schema: json sets the property name, description documents it,
// enum restricts values, and minimum/maximum bound numbers. Because none of the
// fields are pointers, all are marked required in the schema.
type Sentiment struct {
	Label      string  `json:"label" description:"overall sentiment" enum:"positive,negative,neutral"`
	Confidence float64 `json:"confidence" description:"confidence from 0 to 1" minimum:"0" maximum:"1"`
	Summary    string  `json:"summary" description:"one-sentence summary of the review"`
}

func main() {
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	// Generate the tool schema from the struct instead of writing JSON by hand.
	tool, err := schema.Tool("submit_sentiment", "Submit the sentiment analysis of the review.", Sentiment{})
	if err != nil {
		log.Fatalf("build tool schema: %v", err)
	}

	const review = "I was skeptical at first, but this turned out to be the best purchase I've made all year. Setup was effortless and it just works."

	// Run the loop with the generated tool as the terminal tool. The model reads
	// the review, then calls submit_sentiment to finish — its tool input is the
	// structured result.
	result, err := llm.AgentLoopResultWithOptions(context.Background(), client, llm.GenerateOptions{
		System: "You analyze product reviews. Call submit_sentiment exactly once with your analysis.",
		Messages: []llm.Message{{
			Role:    "user",
			Content: []llm.Content{llm.TextContent{Text: "Analyze this review:\n\n" + review}},
		}},
		Tools:     []llm.Tool{tool},
		MaxTokens: 1024,
	}, nil, llm.AgentLoopOptions{
		SubmitResultTool: tool.Name,
		MaxTurns:         4,
	})
	if err != nil {
		log.Fatalf("agent loop: %v", err)
	}

	// The terminal tool input decodes directly into the struct the schema was
	// built from — typed, validated output with no manual parsing.
	var sentiment Sentiment
	if err := json.Unmarshal(result.Input, &sentiment); err != nil {
		log.Fatalf("decode result: %v", err)
	}
	fmt.Printf("Label:      %s\n", sentiment.Label)
	fmt.Printf("Confidence: %.2f\n", sentiment.Confidence)
	fmt.Printf("Summary:    %s\n", sentiment.Summary)
}
