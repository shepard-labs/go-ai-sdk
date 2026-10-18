// Command errors demonstrates each error type the package produces —
// InvalidArgumentError (local validation, no I/O), UnsupportedFunctionalityError
// (media types the API can't accept), APICallError (non-2xx from the server),
// and ErrMissingAPIKey (no key configured).
//
// The validation errors run without a key; the API call errors need one.
package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	ctx := context.Background()

	// ── Local validation errors — no HTTP, no API key required ──

	// Nil question inside a non-nil interface.
	p := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{APIKey: "k"})
	var nilQ *typesafe.ChoiceQuestion
	_, err := p.DecisionModel("jev-latest").DoDecide(ctx, typesafe.DecisionCallOptions{
		State:     []typesafe.StatePart{typesafe.TextStatePart{Text: "x"}},
		Questions: map[string]typesafe.Question{"bad": nilQ},
	})
	fmt.Printf("nil question → %v\n", err)

	// Too many choice criteria (>255) or score levels (>10).
	tooMany := typesafe.ScoreQuestion{Criteria: make([]*typesafe.Input, 11)}
	_, err = p.DecisionModel("jev-latest").DoDecide(ctx, typesafe.DecisionCallOptions{
		State:     []typesafe.StatePart{typesafe.TextStatePart{Text: "x"}},
		Questions: map[string]typesafe.Question{"score": tooMany},
	})
	var iaErr *typesafe.InvalidArgumentError
	if errors.As(err, &iaErr) {
		fmt.Printf("score>10 → invalid arg %q: %s\n", iaErr.Argument, iaErr.Message)
	}

	// Unsupported image media type (only image/png and image/jpeg).
	_, err = p.DecisionModel("jev-latest").DoDecide(ctx, typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{typesafe.FileStatePart{
			MediaType: "image/webp", Data: []byte{1},
		}},
		Questions: map[string]typesafe.Question{},
	})
	var ufErr *typesafe.UnsupportedFunctionalityError
	if errors.As(err, &ufErr) {
		fmt.Printf("webp → unsupported: %s\n", ufErr.Functionality)
	}

	// ── Errors that need a configured key ──

	if os.Getenv("TYPESAFE_AI_API_KEY") == "" {
		fmt.Println("\nTYPESAFE_AI_API_KEY not set — skipping live error cases")
		return
	}
	model := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{}).DecisionModel("jev-latest")

	// 401 from the API → *APICallError with Status + Retryable=false.
	_, err = model.DoDecide(ctx, typesafe.DecisionCallOptions{
		State:     []typesafe.StatePart{typesafe.TextStatePart{Text: "x"}},
		Questions: map[string]typesafe.Question{"q": typesafe.BooleanQuestion{}},
		Headers:   http.Header{"Authorization": {"Bearer ts-invalid"}},
	})
	var apiErr *typesafe.APICallError
	if errors.As(err, &apiErr) {
		fmt.Printf("401 → status=%d retryable=%v msg=%q\n", apiErr.Status, apiErr.Retryable, apiErr.Message)
	}
}
