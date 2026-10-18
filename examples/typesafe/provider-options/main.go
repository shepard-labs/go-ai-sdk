// Command provider-options shows the "typesafe" namespace: longContext is
// the only supported key; unknown keys produce warnings, wrong types produce
// *InvalidArgumentError, and other namespaces are ignored.
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/provider-options
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	if os.Getenv("TYPESAFE_AI_API_KEY") == "" {
		log.Fatal("TYPESAFE_AI_API_KEY is not set")
	}
	model := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{}).DecisionModel("jev-latest")

	base := typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{typesafe.TextStatePart{Text: "Test."}},
		Questions: map[string]typesafe.Question{
			"ok": typesafe.BooleanQuestion{Instructions: ptr(typesafe.StringInput("Is this a test?"))},
		},
	}

	// Unknown typesafe.* key → Warning, call still succeeds.
	opts := base
	opts.ProviderOptions = map[string]map[string]any{
		"typesafe": {"temperature": 0, "longContext": false},
	}
	r, err := model.DoDecide(context.Background(), opts)
	if err != nil {
		log.Fatalf("decide: %v", err)
	}
	fmt.Println("warnings:", r.Warnings)

	// Wrong type for longContext → *InvalidArgumentError before any I/O.
	opts = base
	opts.ProviderOptions = map[string]map[string]any{
		"typesafe": {"longContext": "yes"}, // string, not bool
	}
	_, err = model.DoDecide(context.Background(), opts)
	var iaErr *typesafe.InvalidArgumentError
	if errors.As(err, &iaErr) {
		fmt.Printf("invalid arg (expected): %s → %s\n", iaErr.Argument, iaErr.Message)
	}

	// Other namespaces are ignored entirely — no warnings.
	opts = base
	opts.ProviderOptions = map[string]map[string]any{
		"anthropic": {"temperature": 0},
	}
	opts.Headers = http.Header{"X-Debug": {"1"}} // arbitrary per-call header
	r, err = model.DoDecide(context.Background(), opts)
	if err != nil {
		log.Fatalf("decide: %v", err)
	}
	fmt.Printf("foreign-namespace warnings (want none): %v\n", r.Warnings)
}

func ptr[T any](v T) *T { return &v }
