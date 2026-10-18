// Command decision is the minimal direct-package example: create a TypeSafe
// AI provider, build a decision model, and answer structured questions against
// shared text state in a single call.
//
// Run with a real key:
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/decision
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	if os.Getenv("TYPESAFE_AI_API_KEY") == "" {
		log.Fatal("TYPESAFE_AI_API_KEY is not set: export a TypeSafe AI API key to run this example")
	}

	// ProviderSettings{} reads APIKey from TYPESAFE_AI_API_KEY lazily.
	provider := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{})
	model := provider.DecisionModel("jev-latest")

	result, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.TextStatePart{Text: "Customer: I was charged twice for order #4821. Please refund the duplicate charge."},
		},
		Questions: map[string]typesafe.Question{
			"department": typesafe.ChoiceQuestion{
				Instructions: ptr(typesafe.StringInput("Which team should handle this?")),
				Criteria: map[string]*typesafe.Input{
					"billing":   ptr(typesafe.StringInput("Charges, invoices, and refunds")),
					"technical": ptr(typesafe.StringInput("Bugs and outages")),
					"other":     ptr(typesafe.NullInput()),
				},
			},
			"requestsRefund": typesafe.BooleanQuestion{
				Instructions: ptr(typesafe.StringInput("Is the customer requesting a refund?")),
			},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}

	for id, answer := range result.Answers {
		switch a := answer.(type) {
		case typesafe.ChoiceAnswer:
			fmt.Printf("%s: %s (probabilities %v)\n", id, a.Choice, a.Probabilities)
		case typesafe.BooleanAnswer:
			fmt.Printf("%s: P(true) = %.2f\n", id, a.Probability)
		}
	}
	printMeta(result)
}

func printMeta(r *typesafe.DecisionResult) {
	if c, ok := r.ProviderMetadata["typesafe"]["confidence"].(map[string]float64); ok && len(c) > 0 {
		fmt.Println("confidence:", c)
	}
	fmt.Printf("rounding: %d decimals\n", r.Rounding.ProbabilityDecimals)
	if u := r.Usage; u != nil {
		fmt.Printf("usage: %d in / %d out\n", deref(u.InputTokens), deref(u.OutputTokens))
	}
}

func ptr[T any](v T) *T { return &v }

func deref(p *int64) int64 {
	if p == nil {
		return 0
	}
	return *p
}
