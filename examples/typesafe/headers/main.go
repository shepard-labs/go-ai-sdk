// Command headers shows the three header layers: provider defaults
// (ProviderSettings.Headers), per-call overrides (DecisionCallOptions.Headers),
// and the headers the SDK sets itself (Authorization, User-Agent,
// TypeSafe-Danger-Zone-*). A recording Fetcher lets you inspect what was sent.
//
// Runs without an API key — uses a mock Fetcher.
package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

// echo records the outgoing request and replies with a canned answer.
type echo struct{ last *http.Request }

func (e *echo) Do(r *http.Request) (*http.Response, error) {
	e.last = r
	body := `{"model":"jev-latest","answers":{"ok":{"type":"noul","noul":1}}}`
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}, nil
}

func main() {
	f := &echo{}
	provider := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{
		APIKey: "ts-demo",
		Fetch:  f,
		Headers: http.Header{
			"X-Team": {"platform"},
			"X-Env":  {"prod"},
		},
	})
	model := provider.DecisionModel("jev-latest")

	_, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{typesafe.TextStatePart{Text: "ping"}},
		Questions: map[string]typesafe.Question{
			"ok": typesafe.BooleanQuestion{Instructions: ptr(typesafe.StringInput("ok?"))},
		},
		// Per-call headers replace provider headers with the same name.
		Headers: http.Header{
			"X-Env":     {"canary"},             // overrides provider X-Env
			"X-Request": {"123"},                // additional header
			"X-Trace":   {"will-be-overridden"}, // provider wins if set later
		},
		ProviderOptions: map[string]map[string]any{"typesafe": {"longContext": true}},
	})
	if err != nil {
		fmt.Println("err:", err)
		return
	}

	fmt.Println("Headers actually sent:")
	for k, v := range f.last.Header {
		fmt.Printf("  %s: %s\n", k, strings.Join(v, ", "))
	}
}

func ptr[T any](v T) *T { return &v }
