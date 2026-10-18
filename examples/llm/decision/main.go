// Command decision demonstrates the provider-neutral decision API in
// llm/decision using the TypeSafe AI adapter: a refund-triage scenario that
// answers a choice question (which department) and a boolean question (did
// the customer request a refund) against shared state in a single call.
//
// Run with a real key:
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/llm/decision
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
		log.Fatal("TYPESAFE_AI_API_KEY is not set: export a TypeSafe AI API key to run this example")
	}

	client, err := decisiontypesafe.NewClient(typesafe.ProviderSettings{}, "jev-latest")
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	result, err := client.Decide(context.Background(), decision.CallOptions{
		State: []decision.StatePart{
			decision.TextStatePart{Text: "Customer: I was charged twice for order #4821. Please refund the duplicate charge."},
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
			"requestsRefund": decision.BooleanQuestion{
				Instructions: ptr(decision.StringInput("Is the customer requesting a refund?")),
			},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}

	if answer, ok := result.Answers["department"].(decision.ChoiceAnswer); ok {
		fmt.Println("department:", answer.Choice)
		fmt.Println("department probabilities:", answer.Probabilities)
	}
	if answer, ok := result.Answers["requestsRefund"].(decision.BooleanAnswer); ok {
		fmt.Printf("requestsRefund: P(true) = %.2f\n", answer.Probability)
	}

	// TypeSafe confidence (distinct from probabilities) lives in provider
	// metadata. The API rounds probabilities and scores to 2 decimals, as
	// declared by result.Rounding.
	if confidence, ok := result.ProviderMetadata["typesafe"]["confidence"].(map[string]float64); ok {
		fmt.Println("confidence:", confidence)
	}
	fmt.Printf("rounding: %d decimals (probabilities and scores)\n", result.Rounding.ProbabilityDecimals)
	if result.Usage != nil {
		var in, out string
		if result.Usage.InputTokens != nil {
			in = fmt.Sprintf("%d", *result.Usage.InputTokens)
		}
		if result.Usage.OutputTokens != nil {
			out = fmt.Sprintf("%d", *result.Usage.OutputTokens)
		}
		fmt.Printf("usage: %s input / %s output tokens\n", orUnknown(in), orUnknown(out))
	}
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

func ptr[T any](v T) *T { return &v }
