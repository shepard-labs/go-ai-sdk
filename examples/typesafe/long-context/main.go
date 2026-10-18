// Command long-context enables ProviderOptions["typesafe"]["longContext"]
// which sets the TypeSafe-Danger-Zone-Long-Context header and lets the API
// accept state larger than the standard limit.
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/long-context
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	if os.Getenv("TYPESAFE_AI_API_KEY") == "" {
		log.Fatal("TYPESAFE_AI_API_KEY is not set")
	}
	model := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{}).DecisionModel("jev-latest")

	// A large text state — long context mode tolerates inputs that would
	// exceed the normal request limit.
	large := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 500)
	large += "IMPORTANT: the secret code is ZEBRA-7."

	r, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{typesafe.TextStatePart{Text: large}},
		Questions: map[string]typesafe.Question{
			"code": typesafe.ChoiceQuestion{
				Instructions: ptr(typesafe.StringInput("What is the secret code?")),
				Criteria: map[string]*typesafe.Input{
					"ZEBRA-7":   ptr(typesafe.StringInput("ZEBRA-7")),
					"LION-3":    ptr(typesafe.StringInput("LION-3")),
					"not-found": ptr(typesafe.NullInput()),
				},
			},
		},
		ProviderOptions: map[string]map[string]any{
			"typesafe": {"longContext": true},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}
	if a, ok := r.Answers["code"].(typesafe.ChoiceAnswer); ok {
		fmt.Printf("code: %s  P=%v\n", a.Choice, a.Probabilities)
	}
}

func ptr[T any](v T) *T { return &v }
