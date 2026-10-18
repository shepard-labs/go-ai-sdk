// Command question-types exercises every supported question type in one
// call: choice, score, and boolean. (Box requires an image — see the box
// and images examples.)
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/question-types
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
		log.Fatal("TYPESAFE_AI_API_KEY is not set")
	}
	model := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{}).DecisionModel("jev-latest")

	result, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.TextStatePart{Text: "Ticket #9823 — app crashes on launch after the 4.2 update. Two users affected, workaround is to reinstall."},
		},
		Questions: map[string]typesafe.Question{
			// Choice: pick one named option. Up to 255 criteria.
			"priority": typesafe.ChoiceQuestion{
				Instructions: ptr(typesafe.StringInput("What priority should this ticket be?")),
				Criteria: map[string]*typesafe.Input{
					"low":    ptr(typesafe.StringInput("Cosmetic or minor annoyance")),
					"medium": ptr(typesafe.StringInput("Functionality impaired but workaround exists")),
					"high":   ptr(typesafe.StringInput("Blocking; no workaround")),
				},
			},
			// Score: ordered numeric levels. At most 10 criteria.
			"severity": typesafe.ScoreQuestion{
				Instructions: ptr(typesafe.StringInput("Rate severity.")),
				Criteria: []*typesafe.Input{
					ptr(typesafe.StringInput("Cosmetic; functionality works")),
					ptr(typesafe.StringInput("Functionality impaired; workaround exists")),
					ptr(typesafe.StringInput("Blocking; no workaround")),
				},
			},
			// Boolean: yes/no, returned as P(true). Sent as type "noul".
			"regression": typesafe.BooleanQuestion{
				Instructions: ptr(typesafe.StringInput("Is this a regression introduced by the 4.2 update?")),
			},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}

	if a, ok := result.Answers["priority"].(typesafe.ChoiceAnswer); ok {
		fmt.Printf("priority: %s  P=%v\n", a.Choice, a.Probabilities)
	}
	if a, ok := result.Answers["severity"].(typesafe.ScoreAnswer); ok {
		fmt.Printf("severity: %.2f  per-level=%v\n", a.Score, a.Probabilities)
	}
	if a, ok := result.Answers["regression"].(typesafe.BooleanAnswer); ok {
		fmt.Printf("regression: P(true)=%.2f\n", a.Probability)
	}
	if c, ok := result.ProviderMetadata["typesafe"]["confidence"].(map[string]float64); ok {
		fmt.Println("confidence:", c)
	}
}

func ptr[T any](v T) *T { return &v }
