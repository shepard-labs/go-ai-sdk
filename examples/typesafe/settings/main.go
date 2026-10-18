// Command settings covers ProviderSettings: explicit API key vs the
// TYPESAFE_AI_API_KEY env var, custom BaseURL / TYPESAFE_AI_BASE_URL, default
// headers, body caps, and the Err() pre-flight check.
//
// Runs without an API key — Err() reports the missing key before any I/O.
package main

import (
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	// 1. Environment-driven: APIKey falls back to TYPESAFE_AI_API_KEY at
	//    request time. Err() is a cheap pre-flight check.
	p := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{})
	if err := p.Err(); err != nil {
		fmt.Println("env provider Err():", err)
	} else {
		fmt.Println("env provider: key present (from TYPESAFE_AI_API_KEY)")
	}

	// 2. Explicit settings override the env var. Headers apply to every call;
	//    Authorization from Headers takes precedence over APIKey.
	p2 := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{
		APIKey:  "ts-explicit-key",
		BaseURL: "https://staging.typesafe.example/v2", // also TYPESAFE_AI_BASE_URL
		Headers: http.Header{
			"X-Team":        {"platform"},
			"Authorization": {"Bearer ts-from-headers"}, // wins over APIKey
		},
		MaxResponseBodyBytes:  8 << 20, // cap response bodies (default 32 MiB)
		MaxErrorResponseBytes: 64 << 10,
	})
	fmt.Println("explicit provider Err():", p2.Err()) // nil

	// 3. Missing key is reported lazily, not at construction.
	saved := os.Getenv("TYPESAFE_AI_API_KEY")
	os.Unsetenv("TYPESAFE_AI_API_KEY")
	defer os.Setenv("TYPESAFE_AI_API_KEY", saved)

	p3 := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{})
	if err := p3.Err(); errors.Is(err, typesafe.ErrMissingAPIKey) {
		fmt.Println("missing-key provider Err():", err)
	}
}
