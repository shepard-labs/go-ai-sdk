// Command state covers the three state part types — text, JSON, and how the
// wire encoding chooses between a raw JSON value and a newline-joined string.
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/state
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	if os.Getenv("TYPESAFE_AI_API_KEY") == "" {
		log.Fatal("TYPESAFE_AI_API_KEY is not set")
	}
	model := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{}).DecisionModel("jev-latest")

	// 1. A single JSONStatePart is sent as the raw JSON value — the API sees a
	//    structured object, not a string containing JSON.
	r1, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.JSONStatePart{Value: json.RawMessage(`{
				"message": "charged twice, please refund",
				"order":   {"id": 4821, "amount": 49},
				"history": ["opened dispute", "contacted support"]
			}`)},
		},
		Questions: map[string]typesafe.Question{
			"duplicate": typesafe.BooleanQuestion{Instructions: ptr(typesafe.StringInput("Was the customer charged twice?"))},
		},
	})
	if err != nil {
		log.Fatalf("decide (json): %v", err)
	}
	if a, ok := r1.Answers["duplicate"].(typesafe.BooleanAnswer); ok {
		fmt.Printf("json state → duplicate: P(true)=%.2f\n", a.Probability)
	}

	// 2. Mixed text + JSON parts are compact-encoded and joined with "\n" —
	//    the API sees one string containing both.
	r2, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.TextStatePart{Text: "Ticket #42"},
			typesafe.JSONStatePart{Value: json.RawMessage(`{"priority_hint":"urgent","age_days":14}`)},
			typesafe.TextStatePart{Text: "Customer is threatening to churn."},
		},
		Questions: map[string]typesafe.Question{
			"escalate": typesafe.BooleanQuestion{Instructions: ptr(typesafe.StringInput("Should this be escalated to a manager?"))},
		},
	})
	if err != nil {
		log.Fatalf("decide (mixed): %v", err)
	}
	if a, ok := r2.Answers["escalate"].(typesafe.BooleanAnswer); ok {
		fmt.Printf("mixed state → escalate: P(true)=%.2f\n", a.Probability)
	}
}

func ptr[T any](v T) *T { return &v }
