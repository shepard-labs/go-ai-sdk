// Command inputs demonstrates the four ways to construct a typesafe.Input:
// StringInput, JSONInput, RawInput, and NullInput. Inputs describe question
// instructions and criteria; the API treats them as structured context, not
// just strings.
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/inputs
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

	result, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.TextStatePart{Text: "Review comment: works but the diff adds a panic path; not covered by tests."},
		},
		Questions: map[string]typesafe.Question{
			// JSONInput marshals arbitrary Go values — nested maps, slices, numbers.
			"verdict": typesafe.ChoiceQuestion{
				Instructions: ptr(typesafe.JSONInput(map[string]any{
					"question": "What is the overall verdict on this code change?",
					"context":  "This is a blocking review for a hotfix release.",
				})),
				Criteria: map[string]*typesafe.Input{
					// Mix and match input kinds per criterion.
					"approve":   ptr(typesafe.StringInput("Safe to merge as-is")),
					"changes":   ptr(typesafe.JSONInput([]any{"requests changes", "must fix"})),
					"uncertain": ptr(typesafe.NullInput()), // null = "no description"
				},
			},
			// RawInput embeds literal JSON — useful when criteria are generated
			// elsewhere or stored as bytes.
			"blocking": typesafe.BooleanQuestion{
				Instructions: ptr(typesafe.RawInput(json.RawMessage(`"Does this block the release?"`))),
				Criteria: &typesafe.BooleanCriteria{
					True:  ptr(typesafe.StringInput("Release must wait for a fix")),
					False: ptr(typesafe.NullInput()),
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}
	for id, a := range result.Answers {
		fmt.Printf("%s: %#v\n", id, a)
	}
}

func ptr[T any](v T) *T { return &v }
