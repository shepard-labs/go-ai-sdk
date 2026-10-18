package typesafe

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

type fetchFunc func(*http.Request) (*http.Response, error)

func (f fetchFunc) Do(r *http.Request) (*http.Response, error) { return f(r) }

type recordedRequest struct {
	Method string
	URL    string
	Header http.Header
	Body   []byte
}

type recorder struct {
	requests []recordedRequest
	respond  func(*http.Request) *http.Response
}

func (rc *recorder) Do(r *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(r.Body)
	rc.requests = append(rc.requests, recordedRequest{
		Method: r.Method,
		URL:    r.URL.String(),
		Header: r.Header.Clone(),
		Body:   body,
	})
	return rc.respond(r), nil
}

func jsonResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

var minimalOptions = DecisionCallOptions{
	State: []StatePart{TextStatePart{Text: "The request used the configured endpoint."}},
	Questions: map[string]Question{
		"configured": BooleanQuestion{Instructions: ptr(StringInput("Did the request use the configured endpoint?"))},
	},
}

const minimalResponse = `{"model":"jev-latest","answers":{"configured":{"type":"noul","noul":1}}}`

func ptr[T any](v T) *T { return &v }

func TestProviderDecisionCapability(t *testing.T) {
	model := TypeSafeAI.DecisionModel("jev-latest")
	if model.Provider() != "typesafe.decision" {
		t.Fatalf("provider = %q", model.Provider())
	}
	if model.ModelID() != "jev-latest" {
		t.Fatalf("modelID = %q", model.ModelID())
	}
	want := []QuestionType{QuestionTypeChoice, QuestionTypeScore, QuestionTypeBoolean, QuestionTypeBox}
	got := model.SupportedQuestionTypes()
	if len(got) != len(want) {
		t.Fatalf("supported question types = %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("supported question types = %v", got)
		}
	}
}

func TestProviderUnsupportedFactories(t *testing.T) {
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "k"})
	for name, call := range map[string]func() (any, error){
		"languageModel":  func() (any, error) { return p.LanguageModel("unknown") },
		"embeddingModel": func() (any, error) { return p.EmbeddingModel("unknown") },
		"imageModel":     func() (any, error) { return p.ImageModel("unknown") },
	} {
		if _, err := call(); err == nil {
			t.Fatalf("%s: expected error", name)
		} else if nsErr, ok := errors.AsType[*NoSuchModelError](err); !ok {
			t.Fatalf("%s: expected *NoSuchModelError, got %T", name, err)
		} else if nsErr.ModelID != "unknown" {
			t.Fatalf("%s: model ID = %q", name, nsErr.ModelID)
		}
	}
}

func TestProviderBaseURL(t *testing.T) {
	cases := []struct {
		name     string
		explicit string
		env      string
		want     string
	}{
		{"default", "", "", "https://api.typesafe.ai/v1/systemone"},
		{"env trailing slash trimmed", "", "https://proxy.example/v1/", "https://proxy.example/v1/systemone"},
		{"explicit wins over env", "https://option.example/v1/", "https://environment.example/v1", "https://option.example/v1/systemone"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("TYPESAFE_AI_BASE_URL", tc.env)
			rc := &recorder{respond: func(*http.Request) *http.Response {
				return jsonResponse(200, minimalResponse)
			}}
			p := CreateTypeSafeAI(ProviderSettings{BaseURL: tc.explicit, APIKey: "test-api-key", Fetch: rc})
			if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), minimalOptions); err != nil {
				t.Fatal(err)
			}
			if len(rc.requests) != 1 {
				t.Fatalf("requests = %d", len(rc.requests))
			}
			if rc.requests[0].URL != tc.want {
				t.Fatalf("url = %q, want %q", rc.requests[0].URL, tc.want)
			}
		})
	}
}

func TestProviderLazyAPIKey(t *testing.T) {
	rc := &recorder{respond: func(*http.Request) *http.Response {
		return jsonResponse(200, minimalResponse)
	}}
	p := CreateTypeSafeAI(ProviderSettings{Fetch: rc})
	t.Setenv("TYPESAFE_AI_API_KEY", "environment-key")
	if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), minimalOptions); err != nil {
		t.Fatal(err)
	}
	if got := rc.requests[0].Header.Get("Authorization"); got != "Bearer environment-key" {
		t.Fatalf("authorization = %q", got)
	}
}

func TestProviderExplicitAPIKeyWinsOverEnv(t *testing.T) {
	t.Setenv("TYPESAFE_AI_API_KEY", "environment-key")
	rc := &recorder{respond: func(*http.Request) *http.Response {
		return jsonResponse(200, minimalResponse)
	}}
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "explicit-key", Fetch: rc})
	if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), minimalOptions); err != nil {
		t.Fatal(err)
	}
	if got := rc.requests[0].Header.Get("Authorization"); got != "Bearer explicit-key" {
		t.Fatalf("authorization = %q", got)
	}
}

// TestProviderDefaultVarConfiguredFromEnv exercises the package-level
// TypeSafeAI provider built with zero settings: its key comes from the
// environment at call time.
func TestProviderDefaultVarConfiguredFromEnv(t *testing.T) {
	t.Setenv("TYPESAFE_AI_API_KEY", "default-env-key")
	if err := TypeSafeAI.Err(); err != nil {
		t.Fatalf("TypeSafeAI.Err() = %v", err)
	}
}

func TestProviderMissingAPIKey(t *testing.T) {
	t.Setenv("TYPESAFE_AI_API_KEY", "")
	rc := &recorder{respond: func(*http.Request) *http.Response {
		return jsonResponse(200, minimalResponse)
	}}
	p := CreateTypeSafeAI(ProviderSettings{Fetch: rc})
	if err := p.Err(); !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("Err() = %v", err)
	}
	if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), minimalOptions); !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("DoDecide err = %v", err)
	}
	if len(rc.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(rc.requests))
	}
}
