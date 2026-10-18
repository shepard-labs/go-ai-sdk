// Command adapter uses the provider-neutral llm/decision API backed by the
// typesafe adapter. This is the recommended path when you might swap providers
// later — the call shape is identical across decision providers.
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/adapter
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/llm/decision"
	decisiontypesafe "github.com/shepard-labs/go-ai-sdk/llm/decision/typesafe"
	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	if os.Getenv("TYPESAFE_AI_API_KEY") == "" {
		log.Fatal("TYPESAFE_AI_API_KEY is not set")
	}

	// NewClient checks the key up front; NewAdapter wraps an existing model.
	client, err := decisiontypesafe.NewClient(typesafe.ProviderSettings{}, "jev-latest")
	if err != nil {
		log.Fatalf("create client: %v", err)
	}
	fmt.Println("supported question types:", client.SupportedQuestionTypes())

	result, err := client.Decide(context.Background(), decision.CallOptions{
		State: []decision.StatePart{
			decision.TextStatePart{Text: "Customer: I was charged twice for order #4821."},
		},
		Questions: map[string]decision.Question{
			"department": decision.ChoiceQuestion{
				Instructions: ptr(decision.StringInput("Which team should handle this?")),
				Criteria: map[string]*decision.Input{
					"billing":   ptr(decision.StringInput("Charges, invoices, and refunds")),
					"technical": ptr(decision.StringInput("Bugs and outages")),
					"other":     ptr(decision.NullInput()),
				},
			},
			"refund": decision.BooleanQuestion{
				Instructions: ptr(decision.StringInput("Is the customer requesting a refund?")),
			},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}

	for id, a := range result.Answers {
		fmt.Printf("%s: %#v\n", id, a)
	}
	// TypeSafe-specific extras still surface in ProviderMetadata.
	if c, ok := result.ProviderMetadata["typesafe"]["confidence"].(map[string]float64); ok {
		fmt.Println("confidence:", c)
	}
}

func ptr[T any](v T) *T { return &v }
