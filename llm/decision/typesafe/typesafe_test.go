package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm/decision"
	typesafesdk "github.com/shepard-labs/go-ai-sdk/typesafe"
)

const coreResponse = `{
	"model": "jev-1.13.0",
	"answers": {
		"department": {"type": "choice", "choice": "billing", "confidence": 1.0, "probabilities": {"technical": 0.0, "other": 0.0, "billing": 1.0}},
		"severity": {"type": "score", "score": 0.97, "confidence": 0.64, "probabilities": {"0": 0.13, "1": 0.76, "2": 0.11}},
		"requestsRefund": {"type": "noul", "noul": 0.99}
	},
	"usage": {"input_tokens": 471, "output_tokens": 71}
}`

// recordedRequest captures one HTTP request observed by the test server.
type recordedRequest struct {
	Header http.Header
	Body   []byte
}

// decisionServer serves body for every request and records what it received.
func decisionServer(t *testing.T, status int, body string) (*httptest.Server, *[]recordedRequest) {
	t.Helper()
	var requests []recordedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		requests = append(requests, recordedRequest{Header: r.Header.Clone(), Body: b})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &requests
}

func coreCallOptions() decision.CallOptions {
	return decision.CallOptions{
		State: []decision.StatePart{
			decision.JSONStatePart{Value: json.RawMessage(`{"message":"I was charged twice. Please refund the duplicate.","order":{"amount":49}}`)},
		},
		Questions: map[string]decision.Question{
			"department": decision.ChoiceQuestion{
				Instructions: ptr(decision.JSONInput(map[string]any{"question": "Which team should handle this?"})),
				Criteria: map[string]*decision.Input{
					"billing":   ptr(decision.JSONInput(map[string]any{"includes": []any{"Charges", "Invoices", "Refunds"}})),
					"technical": ptr(decision.JSONInput([]any{"Bugs", "Outages"})),
					"other":     ptr(decision.NullInput()),
				},
			},
			"severity": decision.ScoreQuestion{
				Instructions: ptr(decision.JSONInput([]any{"How severe is the issue?"})),
				Criteria: []*decision.Input{
					ptr(decision.JSONInput(map[string]any{"meaning": "Cosmetic; functionality works"})),
					ptr(decision.StringInput("Functionality impaired; workaround exists")),
					ptr(decision.StringInput("Blocking; no workaround")),
				},
			},
			"requestsRefund": decision.BooleanQuestion{
				Instructions: ptr(decision.StringInput("Is the customer requesting a refund?")),
				Criteria: &decision.BooleanCriteria{
					True:  ptr(decision.JSONInput(map[string]any{"meaning": "Explicit request for money back"})),
					False: ptr(decision.NullInput()),
				},
			},
		},
	}
}

func ptr[T any](v T) *T { return &v }

func TestDecideCoreScenario(t *testing.T) {
	srv, requests := decisionServer(t, 200, coreResponse)
	client, err := NewClient(typesafesdk.ProviderSettings{APIKey: "test-api-key", BaseURL: srv.URL}, "jev-latest")
	if err != nil {
		t.Fatal(err)
	}

	opts := coreCallOptions()
	opts.Headers = map[string]string{"X-Custom": "call"}
	opts.ProviderOptions = decision.ProviderOptions{"typesafe": {"temperature": 0}}
	result, err := client.Decide(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}

	// Wire request: one call, JSON body, boolean renamed to noul on the wire
	// even though the neutral API says boolean.
	if len(*requests) != 1 {
		t.Fatalf("requests = %d", len(*requests))
	}
	req := (*requests)[0]
	if got := req.Header.Get("Authorization"); got != "Bearer test-api-key" {
		t.Fatalf("authorization = %q", got)
	}
	if got := req.Header.Get("Content-Type"); got != "application/json" {
		t.Fatalf("content-type = %q", got)
	}
	if got := req.Header.Get("X-Custom"); got != "call" {
		t.Fatalf("x-custom = %q", got)
	}
	if got := req.Header.Get("TypeSafe-Danger-Zone-Images"); got != "" {
		t.Fatalf("images header = %q, want absent", got)
	}
	var wire map[string]any
	if err := json.Unmarshal(req.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire["model"] != "jev-latest" {
		t.Fatalf("model = %v", wire["model"])
	}
	wantState := map[string]any{"message": "I was charged twice. Please refund the duplicate.", "order": map[string]any{"amount": float64(49)}}
	if !reflect.DeepEqual(wire["state"], wantState) {
		t.Fatalf("state = %v", wire["state"])
	}
	questions, ok := wire["questions"].(map[string]any)
	if !ok {
		t.Fatalf("questions = %v", wire["questions"])
	}
	if got := questions["department"].(map[string]any)["type"]; got != "choice" {
		t.Fatalf("department type = %v", got)
	}
	if got := questions["severity"].(map[string]any)["type"]; got != "score" {
		t.Fatalf("severity type = %v", got)
	}
	if got := questions["requestsRefund"].(map[string]any)["type"]; got != "noul" {
		t.Fatalf("requestsRefund type = %v, want noul on the wire", got)
	}
	if strings.Contains(string(req.Body), `"boolean"`) {
		t.Fatal("wire body must not contain the neutral boolean type name")
	}

	// Neutral result mapping.
	wantAnswers := map[string]decision.Answer{
		"department":     decision.ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"technical": 0, "other": 0, "billing": 1}},
		"severity":       decision.ScoreAnswer{Score: 0.97, Probabilities: map[string]float64{"0": 0.13, "1": 0.76, "2": 0.11}},
		"requestsRefund": decision.BooleanAnswer{Probability: 0.99},
	}
	if !reflect.DeepEqual(result.Answers, wantAnswers) {
		t.Fatalf("answers = %#v", result.Answers)
	}
	if result.Rounding != (decision.Rounding{ProbabilityDecimals: 2, ScoreDecimals: 2}) {
		t.Fatalf("rounding = %+v", result.Rounding)
	}
	confidence, ok := result.ProviderMetadata["typesafe"]["confidence"].(map[string]float64)
	if !ok {
		t.Fatalf("confidence metadata = %#v", result.ProviderMetadata)
	}
	if !reflect.DeepEqual(confidence, map[string]float64{"department": 1, "severity": 0.64}) {
		t.Fatalf("confidence = %v", confidence)
	}
	if result.Usage == nil || *result.Usage.InputTokens != 471 || *result.Usage.OutputTokens != 71 {
		t.Fatalf("usage = %+v", result.Usage)
	}
	if result.Response.ModelID != "jev-1.13.0" {
		t.Fatalf("response model = %q", result.Response.ModelID)
	}
	if len(result.Response.Body) == 0 {
		t.Fatal("response body is empty")
	}
	wantWarnings := []decision.Warning{{Type: "unsupported", Feature: "providerOptions.typesafe.temperature"}}
	if !reflect.DeepEqual(result.Warnings, wantWarnings) {
		t.Fatalf("warnings = %+v", result.Warnings)
	}
}

func TestDecideImageScenario(t *testing.T) {
	const imageResponse = `{
		"model": "jev-1.14.0",
		"answers": {
			"logo": {"type": "box", "box": {"top": 0.1, "bottom": 0.4, "left": 0.2, "right": 0.5}, "confidence": 0.9},
			"hasCat": {"type": "noul", "noul": 0.75}
		},
		"usage": {"input_tokens": 100, "output_tokens": 10}
	}`
	srv, requests := decisionServer(t, 200, imageResponse)
	client, err := NewClient(typesafesdk.ProviderSettings{APIKey: "test-api-key", BaseURL: srv.URL}, "jev-latest")
	if err != nil {
		t.Fatal(err)
	}

	png := []byte{0x89, 0x50, 0x4E, 0x47}
	result, err := client.Decide(t.Context(), decision.CallOptions{
		State: []decision.StatePart{
			decision.TextStatePart{Text: "Look at this photo."},
			decision.FileStatePart{MediaType: "image/png", Data: png, Reference: "cat-photo"},
		},
		Questions: map[string]decision.Question{
			"logo":   decision.BoxQuestion{Instructions: decision.StringInput("Locate the logo.")},
			"hasCat": decision.BooleanQuestion{Instructions: ptr(decision.StringInput("Is there a cat?"))},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(*requests) != 1 {
		t.Fatalf("requests = %d", len(*requests))
	}
	req := (*requests)[0]
	if got := req.Header.Get("TypeSafe-Danger-Zone-Images"); got != "true" {
		t.Fatalf("images header = %q", got)
	}

	// Parse the multipart request: a request.json part plus one image part.
	mediaType, params, err := mime.ParseMediaType(req.Header.Get("Content-Type"))
	if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
		t.Fatalf("content type %q: %v", req.Header.Get("Content-Type"), err)
	}
	mr := multipart.NewReader(bytes.NewReader(req.Body), params["boundary"])
	parts := map[string]struct {
		filename string
		data     []byte
	}{}
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Fatal(err)
		}
		parts[part.FormName()] = struct {
			filename string
			data     []byte
		}{part.FileName(), data}
	}
	requestPart, ok := parts["request"]
	if !ok || requestPart.filename != "request.json" {
		t.Fatalf("request part = %+v", parts)
	}
	imagePart, ok := parts["image-1"]
	if !ok {
		t.Fatalf("image part missing: %v", parts)
	}
	if imagePart.filename != "image-1.png" {
		t.Fatalf("image filename = %q", imagePart.filename)
	}
	if string(imagePart.data) != string(png) {
		t.Fatalf("image bytes = %v", imagePart.data)
	}

	var wire map[string]any
	if err := json.Unmarshal(requestPart.data, &wire); err != nil {
		t.Fatal(err)
	}
	wantAssets := map[string]any{"cat-photo": map[string]any{"type": "image", "part": "image-1"}}
	if !reflect.DeepEqual(wire["assets"], wantAssets) {
		t.Fatalf("assets = %v", wire["assets"])
	}
	if wire["state"] != "Look at this photo.\ncat-photo" {
		t.Fatalf("state = %v", wire["state"])
	}
	questions := wire["questions"].(map[string]any)
	wantBox := map[string]any{"type": "box", "instructions": "Locate the logo."}
	if !reflect.DeepEqual(questions["logo"], wantBox) {
		t.Fatalf("box question = %v", questions["logo"])
	}
	if got := questions["hasCat"].(map[string]any)["type"]; got != "noul" {
		t.Fatalf("hasCat type = %v, want noul on the wire", got)
	}

	// Neutral result mapping.
	if got, ok := result.Answers["logo"].(decision.BoxAnswer); !ok {
		t.Fatalf("logo answer = %#v", result.Answers["logo"])
	} else if got.Box != (decision.Box{Top: 0.1, Bottom: 0.4, Left: 0.2, Right: 0.5}) {
		t.Fatalf("box = %+v", got.Box)
	}
	if got, ok := result.Answers["hasCat"].(decision.BooleanAnswer); !ok {
		t.Fatalf("hasCat answer = %#v", result.Answers["hasCat"])
	} else if got.Probability != 0.75 {
		t.Fatalf("probability = %v", got.Probability)
	}
	confidence := result.ProviderMetadata["typesafe"]["confidence"].(map[string]float64)
	if !reflect.DeepEqual(confidence, map[string]float64{"logo": 0.9}) {
		t.Fatalf("confidence = %v, want box only (boolean has none)", confidence)
	}
	if result.Response.ModelID != "jev-1.14.0" {
		t.Fatalf("response model = %q", result.Response.ModelID)
	}
}

// stubModel is a typesafe.DecisionModel with a restricted supported set.
type stubModel struct {
	supported []typesafesdk.QuestionType
	calls     int
}

func (m *stubModel) ModelID() string  { return "stub" }
func (m *stubModel) Provider() string { return "typesafe.decision" }
func (m *stubModel) SupportedQuestionTypes() []typesafesdk.QuestionType {
	return m.supported
}
func (m *stubModel) DoDecide(ctx context.Context, opts typesafesdk.DecisionCallOptions) (*typesafesdk.DecisionResult, error) {
	m.calls++
	return &typesafesdk.DecisionResult{Answers: map[string]typesafesdk.Answer{}}, nil
}

func TestDecideUnsupportedQuestionTypeRejectedBeforeIO(t *testing.T) {
	model := &stubModel{supported: []typesafesdk.QuestionType{typesafesdk.QuestionTypeChoice}}
	client := NewAdapter(model)
	_, err := client.Decide(t.Context(), decision.CallOptions{
		State: []decision.StatePart{decision.TextStatePart{Text: "test"}},
		Questions: map[string]decision.Question{
			"requestsRefund": decision.BooleanQuestion{Instructions: ptr(decision.StringInput("Refund?"))},
		},
	})
	var unsupported *decision.UnsupportedQuestionTypeError
	if !errors.As(err, &unsupported) {
		t.Fatalf("err = %v (%T)", err, err)
	}
	if unsupported.QuestionID != "requestsRefund" || unsupported.Type != decision.QuestionTypeBoolean {
		t.Fatalf("unsupported = %+v", unsupported)
	}
	if model.calls != 0 {
		t.Fatalf("model calls = %d, want 0 (rejected before I/O)", model.calls)
	}
}

func TestSupportedQuestionTypes(t *testing.T) {
	model := &stubModel{supported: []typesafesdk.QuestionType{typesafesdk.QuestionTypeChoice, typesafesdk.QuestionTypeBox}}
	got := NewAdapter(model).SupportedQuestionTypes()
	want := []decision.QuestionType{decision.QuestionTypeChoice, decision.QuestionTypeBox}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("supported = %v, want %v", got, want)
	}
}

func TestNewClientAPIKeyHandling(t *testing.T) {
	t.Run("missing key", func(t *testing.T) {
		t.Setenv("TYPESAFE_AI_API_KEY", "")
		_, err := NewClient(typesafesdk.ProviderSettings{}, "jev-latest")
		if !errors.Is(err, typesafesdk.ErrMissingAPIKey) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("explicit key", func(t *testing.T) {
		client, err := NewClient(typesafesdk.ProviderSettings{APIKey: "test-api-key"}, "jev-latest")
		if err != nil {
			t.Fatal(err)
		}
		if got := client.SupportedQuestionTypes(); len(got) != 4 {
			t.Fatalf("supported = %v", got)
		}
	})
}

func TestDecidePropagatesAPIError(t *testing.T) {
	srv, requests := decisionServer(t, 429, `{"message":"slow down"}`)
	client, err := NewClient(typesafesdk.ProviderSettings{APIKey: "test-api-key", BaseURL: srv.URL}, "jev-latest")
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Decide(t.Context(), coreCallOptions())
	var apiErr *typesafesdk.APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v (%T)", err, err)
	}
	if apiErr.Status != 429 || !apiErr.Retryable {
		t.Fatalf("apiErr = %+v", apiErr)
	}
	if len(*requests) != 1 {
		t.Fatalf("requests = %d", len(*requests))
	}
}
