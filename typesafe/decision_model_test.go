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
	"os"
	"reflect"
	"strings"
	"testing"
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func requireJSONEqual(t *testing.T, want, got []byte) {
	t.Helper()
	var w, g any
	if err := json.Unmarshal(want, &w); err != nil {
		t.Fatalf("want is not JSON: %v", err)
	}
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("got is not JSON: %v\n%s", err, got)
	}
	if !reflect.DeepEqual(w, g) {
		t.Fatalf("JSON mismatch\nwant: %s\ngot:  %s", want, got)
	}
}

type multipartPart struct {
	Name        string
	Filename    string
	ContentType string
	Data        []byte
}

func parseMultipart(t *testing.T, contentType string, body []byte) []multipartPart {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
		t.Fatalf("content type %q: %v", contentType, err)
	}
	mr := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	var parts []multipartPart
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			return parts
		}
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Fatal(err)
		}
		parts = append(parts, multipartPart{
			Name:        part.FormName(),
			Filename:    part.FileName(),
			ContentType: part.Header.Get("Content-Type"),
			Data:        data,
		})
	}
}

func findPart(t *testing.T, parts []multipartPart, name string) multipartPart {
	t.Helper()
	for _, p := range parts {
		if p.Name == name {
			return p
		}
	}
	t.Fatalf("multipart part %q not found", name)
	return multipartPart{}
}

func multipartBody(t *testing.T, req recordedRequest) map[string]any {
	t.Helper()
	parts := parseMultipart(t, req.Header.Get("Content-Type"), req.Body)
	requestPart := findPart(t, parts, "request")
	var body map[string]any
	if err := json.Unmarshal(requestPart.Data, &body); err != nil {
		t.Fatal(err)
	}
	return body
}

// fixtureQuestions matches testdata/decision-request.json.
func fixtureQuestions() map[string]Question {
	return map[string]Question{
		"department": ChoiceQuestion{
			Instructions: ptr(JSONInput(map[string]any{"question": "Which team should handle this?"})),
			Criteria: map[string]*Input{
				"billing":   ptr(JSONInput(map[string]any{"includes": []any{"Charges", "Invoices", "Refunds"}})),
				"technical": ptr(JSONInput([]any{"Bugs", "Outages"})),
				"other":     ptr(NullInput()),
			},
		},
		"severity": ScoreQuestion{
			Instructions: ptr(JSONInput([]any{"How severe is the issue?"})),
			Criteria: []*Input{
				ptr(JSONInput(map[string]any{"meaning": "Cosmetic; functionality works"})),
				ptr(StringInput("Functionality impaired; workaround exists")),
				ptr(StringInput("Blocking; no workaround")),
			},
		},
		"requestsRefund": BooleanQuestion{
			Instructions: ptr(StringInput("Is the customer requesting a refund?")),
			Criteria: &BooleanCriteria{
				True:  ptr(JSONInput(map[string]any{"meaning": "Explicit request for money back"})),
				False: ptr(NullInput()),
			},
		},
	}
}

func fixtureOptions() DecisionCallOptions {
	return DecisionCallOptions{
		State:     []StatePart{JSONStatePart{Value: json.RawMessage(`{"message":"I was charged twice. Please refund the duplicate.","order":{"amount":49}}`)}},
		Questions: fixtureQuestions(),
	}
}

func newFixtureRecorder(t *testing.T) *recorder {
	t.Helper()
	return &recorder{respond: func(*http.Request) *http.Response {
		return jsonResponse(200, string(readFixture(t, "decision-response.json")))
	}}
}

func TestDoDecideSendsAllQuestionTypes(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions()); err != nil {
		t.Fatal(err)
	}
	if len(rc.requests) != 1 {
		t.Fatalf("requests = %d", len(rc.requests))
	}
	req := rc.requests[0]
	requireJSONEqual(t, readFixture(t, "decision-request.json"), req.Body)
	if got := req.Header.Get("Authorization"); got != "Bearer test-api-key" {
		t.Fatalf("authorization = %q", got)
	}
	if got := req.Header.Get("Content-Type"); got != "application/json" {
		t.Fatalf("content-type = %q", got)
	}
}

func TestDoDecidePreservesNativeResult(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	result, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions())
	if err != nil {
		t.Fatal(err)
	}
	wantAnswers := map[string]Answer{
		"department":     ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"technical": 0, "other": 0, "billing": 1}},
		"severity":       ScoreAnswer{Score: 0.97, Probabilities: map[string]float64{"0": 0.13, "1": 0.76, "2": 0.11}},
		"requestsRefund": BooleanAnswer{Probability: 0.99},
	}
	if !reflect.DeepEqual(result.Answers, wantAnswers) {
		t.Fatalf("answers = %#v", result.Answers)
	}
	if result.Rounding != (Rounding{ProbabilityDecimals: 2, ScoreDecimals: 2}) {
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
	requireJSONEqual(t, readFixture(t, "decision-response.json"), result.Response.Body)
}

func TestDoDecideCustomConfiguration(t *testing.T) {
	rc := &recorder{respond: func(*http.Request) *http.Response {
		resp := jsonResponse(200, string(readFixture(t, "decision-response.json")))
		resp.Header.Set("x-request-id", "test-id")
		return resp
	}}
	p := CreateTypeSafeAI(ProviderSettings{
		APIKey:  "custom-key",
		BaseURL: "https://custom.example/v1/",
		Fetch:   rc,
		Headers: http.Header{"custom": {"provider"}, "shared": {"provider"}},
	})
	opts := fixtureOptions()
	opts.Headers = http.Header{"shared": {"call"}}
	result, err := p.DecisionModel("jev-future").DoDecide(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(rc.requests) != 1 {
		t.Fatalf("requests = %d", len(rc.requests))
	}
	req := rc.requests[0]
	if req.URL != "https://custom.example/v1/systemone" {
		t.Fatalf("url = %q", req.URL)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer custom-key" {
		t.Fatalf("authorization = %q", got)
	}
	if got := req.Header.Get("custom"); got != "provider" {
		t.Fatalf("custom header = %q", got)
	}
	if got := req.Header.Get("shared"); got != "call" {
		t.Fatalf("shared header = %q, want per-call override", got)
	}
	if got := req.Header.Get("User-Agent"); !strings.Contains(got, "ai-sdk-go/typesafe/"+Version) {
		t.Fatalf("user-agent = %q", got)
	}
	var body map[string]any
	if err := json.Unmarshal(req.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "jev-future" {
		t.Fatalf("model = %v", body["model"])
	}
	if got := result.Response.Headers.Get("x-request-id"); got != "test-id" {
		t.Fatalf("response x-request-id = %q", got)
	}
}

func TestDoDecideUnknownProviderOptionWarns(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	opts := fixtureOptions()
	opts.ProviderOptions = map[string]map[string]any{"typesafe": {"temperature": 0}}
	result, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	want := []Warning{{Type: WarningTypeUnsupported, Feature: "providerOptions.typesafe.temperature"}}
	if !reflect.DeepEqual(result.Warnings, want) {
		t.Fatalf("warnings = %+v", result.Warnings)
	}
	requireJSONEqual(t, readFixture(t, "decision-request.json"), rc.requests[0].Body)
}

func TestDoDecideOtherProviderNamespaceIgnored(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	opts := fixtureOptions()
	opts.ProviderOptions = map[string]map[string]any{"anthropic": {"temperature": 0}}
	result, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Warnings) != 0 {
		t.Fatalf("warnings = %+v, want none", result.Warnings)
	}
	requireJSONEqual(t, readFixture(t, "decision-request.json"), rc.requests[0].Body)
}

func TestDoDecideNilQuestionRejected(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	opts := fixtureOptions()
	opts.Questions = map[string]Question{"broken": (*ChoiceQuestion)(nil)}
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts)
	if _, ok := errors.AsType[*InvalidArgumentError](err); !ok {
		t.Fatalf("err = %v (%T)", err, err)
	}
	if len(rc.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(rc.requests))
	}
}

func TestDoDecideLongContext(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
		set   bool
		want  string
	}{
		{"true", true, true, "true"},
		{"false", false, true, ""},
		{"absent", nil, false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rc := newFixtureRecorder(t)
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			opts := fixtureOptions()
			if tc.set {
				opts.ProviderOptions = map[string]map[string]any{"typesafe": {"longContext": tc.value}}
			}
			result, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts)
			if err != nil {
				t.Fatal(err)
			}
			if got := rc.requests[0].Header.Get("TypeSafe-Danger-Zone-Long-Context"); got != tc.want {
				t.Fatalf("long-context header = %q, want %q", got, tc.want)
			}
			requireJSONEqual(t, readFixture(t, "decision-request.json"), rc.requests[0].Body)
			if len(result.Warnings) != 0 {
				t.Fatalf("warnings = %+v", result.Warnings)
			}
		})
	}
}

func TestDoDecideLongContextInvalid(t *testing.T) {
	for _, value := range []any{"true", 1, nil} {
		rc := newFixtureRecorder(t)
		p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
		opts := fixtureOptions()
		opts.ProviderOptions = map[string]map[string]any{"typesafe": {"longContext": value}}
		_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts)
		if iaErr, ok := errors.AsType[*InvalidArgumentError](err); !ok {
			t.Fatalf("longContext %v: err = %v (%T)", value, err, err)
		} else if iaErr.Argument != "providerOptions.typesafe.longContext" {
			t.Fatalf("argument = %q", iaErr.Argument)
		}
		if len(rc.requests) != 0 {
			t.Fatalf("longContext %v: requests = %d, want 0", value, len(rc.requests))
		}
	}
}

func TestDoDecideLongContextWithImage(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	opts := DecisionCallOptions{
		State: []StatePart{FileStatePart{MediaType: "image/png", Data: []byte{1, 2, 3}}},
		Questions: map[string]Question{
			"cat": BooleanQuestion{Instructions: ptr(StringInput("Is this a cat?"))},
		},
		ProviderOptions: map[string]map[string]any{"typesafe": {"longContext": true, "temperature": 0}},
		Headers:         http.Header{"x-custom": {"custom"}},
	}
	result, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	h := rc.requests[0].Header
	if got := h.Get("TypeSafe-Danger-Zone-Images"); got != "true" {
		t.Fatalf("images header = %q", got)
	}
	if got := h.Get("TypeSafe-Danger-Zone-Long-Context"); got != "true" {
		t.Fatalf("long-context header = %q", got)
	}
	if got := h.Get("x-custom"); got != "custom" {
		t.Fatalf("x-custom = %q", got)
	}
	want := []Warning{{Type: WarningTypeUnsupported, Feature: "providerOptions.typesafe.temperature"}}
	if !reflect.DeepEqual(result.Warnings, want) {
		t.Fatalf("warnings = %+v", result.Warnings)
	}
}

func TestDoDecideQuestionLimits(t *testing.T) {
	choiceCriteria := func(n int) map[string]*Input {
		m := make(map[string]*Input, n)
		for i := range n {
			m[string(rune('a'+i%26))+string(rune('a'+i/26))] = nil
		}
		return m
	}
	scoreCriteria := func(n int) []*Input {
		s := make([]*Input, n)
		for i := range n {
			s[i] = ptr(NullInput())
		}
		return s
	}
	for _, tc := range []struct {
		name     string
		question Question
		wantErr  bool
	}{
		{"choice 255", ChoiceQuestion{Instructions: ptr(StringInput("Pick")), Criteria: choiceCriteria(255)}, false},
		{"choice 256", ChoiceQuestion{Instructions: ptr(StringInput("Pick")), Criteria: choiceCriteria(256)}, true},
		{"score 10", ScoreQuestion{Instructions: ptr(StringInput("Rate")), Criteria: scoreCriteria(10)}, false},
		{"score 11", ScoreQuestion{Instructions: ptr(StringInput("Rate")), Criteria: scoreCriteria(11)}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rc := newFixtureRecorder(t)
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			opts := DecisionCallOptions{
				State:     []StatePart{TextStatePart{Text: "test"}},
				Questions: map[string]Question{"question": tc.question},
			}
			_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts)
			if tc.wantErr {
				if iaErr, ok := errors.AsType[*InvalidArgumentError](err); !ok {
					t.Fatalf("err = %v (%T)", err, err)
				} else if iaErr.Argument != "questions.question.criteria" {
					t.Fatalf("argument = %q", iaErr.Argument)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			wantRequests := 1
			if tc.wantErr {
				wantRequests = 0
			}
			if len(rc.requests) != wantRequests {
				t.Fatalf("requests = %d, want %d", len(rc.requests), wantRequests)
			}
		})
	}
}

func TestDoDecideHTTPErrorStatusAndRetryability(t *testing.T) {
	for _, tc := range []struct {
		status    int
		retryable bool
	}{
		{401, false},
		{408, true},
		{409, true},
		{422, false},
		{429, true},
		{500, true},
		{529, true},
	} {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			rc := &recorder{respond: func(*http.Request) *http.Response {
				return jsonResponse(tc.status, `{"detail":"Provider error"}`)
			}}
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions())
			apiErr, ok := errors.AsType[*APICallError](err)
			if !ok {
				t.Fatalf("err = %v (%T)", err, err)
			}
			if apiErr.Status != tc.status {
				t.Fatalf("status = %d", apiErr.Status)
			}
			if apiErr.Retryable != tc.retryable {
				t.Fatalf("retryable = %v, want %v", apiErr.Retryable, tc.retryable)
			}
			if apiErr.Message != "Provider error" {
				t.Fatalf("message = %q", apiErr.Message)
			}
			if len(rc.requests) != 1 {
				t.Fatalf("requests = %d, want exactly one attempt", len(rc.requests))
			}
		})
	}
}

func TestDoDecideHTTPErrorMessagePrecedence(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"detail array JSON-encoded", `{"detail":[{"loc":["body","questions"],"msg":"Invalid question"}]}`, `[{"loc":["body","questions"],"msg":"Invalid question"}]`},
		{"error message object", `{"error":{"message":"Overloaded"}}`, "Overloaded"},
		{"error string", `{"error":"Overloaded"}`, "Overloaded"},
		{"message", `{"message":"Overloaded"}`, "Overloaded"},
		{"message beats error_type", `{"error_type":"max_tokens_exceeded","message":"Input exceeds the model context window"}`, "Input exceeds the model context window"},
		{"bare error_type", `{"error_type":"max_tokens_exceeded"}`, "max_tokens_exceeded"},
		{"error beats detail", `{"error":"e","detail":"d"}`, "e"},
		{"message beats error", `{"message":"m","error":"e"}`, "m"},
		{"unrecognized fallback", `{"unrecognised":"shape"}`, "TypeSafe request failed"},
		{"non-JSON fallback", `gateway timeout`, "TypeSafe request failed"},
		{"empty message wins over error_type", `{"message":"","error_type":"max_tokens_exceeded"}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rc := &recorder{respond: func(*http.Request) *http.Response {
				return jsonResponse(400, tc.body)
			}}
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions())
			apiErr, ok := errors.AsType[*APICallError](err)
			if !ok {
				t.Fatalf("err = %v (%T)", err, err)
			}
			if apiErr.Message != tc.want {
				t.Fatalf("message = %q, want %q", apiErr.Message, tc.want)
			}
			if apiErr.Status != 400 {
				t.Fatalf("status = %d", apiErr.Status)
			}
		})
	}
}

func TestDoDecideMalformedSuccess(t *testing.T) {
	for name, body := range map[string]string{
		"invalid JSON":        `not json`,
		"missing noul value":  `{"answers":{"test":{"type":"noul"}}}`,
		"unknown answer type": `{"answers":{"x":{"type":"weird","x":1}}}`,
		"missing answers":     `{"model":"jev-latest"}`,
	} {
		t.Run(name, func(t *testing.T) {
			rc := &recorder{respond: func(*http.Request) *http.Response {
				return jsonResponse(200, body)
			}}
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions())
			if _, ok := errors.AsType[*InvalidResponseDataError](err); !ok {
				t.Fatalf("err = %v (%T)", err, err)
			}
		})
	}
}

func TestDoDecideNullOptionalMetadata(t *testing.T) {
	rc := &recorder{respond: func(*http.Request) *http.Response {
		return jsonResponse(200, `{"model":null,"usage":null,"answers":{"topic":{"type":"choice","choice":"a","probabilities":{"a":1},"confidence":null}}}`)
	}}
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	result, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions())
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage != nil {
		t.Fatalf("usage = %+v, want nil", result.Usage)
	}
	if result.Response.ModelID != "jev-latest" {
		t.Fatalf("model = %q, want requested fallback", result.Response.ModelID)
	}
	confidence, ok := result.ProviderMetadata["typesafe"]["confidence"].(map[string]float64)
	if !ok || len(confidence) != 0 {
		t.Fatalf("confidence = %#v, want empty map", result.ProviderMetadata)
	}
}

func TestDoDecideContextCancellation(t *testing.T) {
	t.Run("cancelled before fetch", func(t *testing.T) {
		rc := newFixtureRecorder(t)
		p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		_, err := p.DecisionModel("jev-latest").DoDecide(ctx, fixtureOptions())
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v", err)
		}
		if len(rc.requests) != 0 {
			t.Fatalf("requests = %d", len(rc.requests))
		}
	})
	t.Run("cancelled during fetch", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		calls := 0
		p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: fetchFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			cancel()
			return nil, r.Context().Err()
		})})
		_, err := p.DecisionModel("jev-latest").DoDecide(ctx, fixtureOptions())
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v", err)
		}
		if calls != 1 {
			t.Fatalf("fetch calls = %d", calls)
		}
	})
}

func TestDoDecideStateTextAndJSONJoin(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	opts := fixtureOptions()
	opts.State = []StatePart{
		TextStatePart{Text: "Inspect."},
		JSONStatePart{Value: json.RawMessage(`[1,null]`)},
	}
	if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts); err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(rc.requests[0].Body, &body); err != nil {
		t.Fatal(err)
	}
	if body["state"] != "Inspect.\n[1,null]" {
		t.Fatalf("state = %v", body["state"])
	}
}

func TestDoDecideSingleJSONStatePassthrough(t *testing.T) {
	for _, value := range []string{
		`{"nested":{"amount":49},"history":["refund"]}`,
		`[1,null]`,
	} {
		rc := newFixtureRecorder(t)
		p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
		opts := fixtureOptions()
		opts.State = []StatePart{JSONStatePart{Value: json.RawMessage(value)}}
		if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts); err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(rc.requests[0].Body, &body); err != nil {
			t.Fatal(err)
		}
		var want any
		if err := json.Unmarshal([]byte(value), &want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(body["state"], want) {
			t.Fatalf("state = %v, want %v", body["state"], want)
		}
	}
}

func TestDoDecideBodyCaps(t *testing.T) {
	t.Run("response truncation", func(t *testing.T) {
		rc := &recorder{respond: func(*http.Request) *http.Response {
			return jsonResponse(200, `{"answers":{}}`)
		}}
		p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc, MaxResponseBodyBytes: 10})
		_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions())
		apiErr, ok := errors.AsType[*APICallError](err)
		if !ok || !apiErr.Truncated {
			t.Fatalf("err = %v (%T)", err, err)
		}
	})
	t.Run("error truncation", func(t *testing.T) {
		rc := &recorder{respond: func(*http.Request) *http.Response {
			return jsonResponse(500, `{"message":"`+strings.Repeat("x", 100)+`"}`)
		}}
		p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc, MaxErrorResponseBytes: 10})
		_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions())
		apiErr, ok := errors.AsType[*APICallError](err)
		if !ok || !apiErr.Truncated {
			t.Fatalf("err = %v (%T)", err, err)
		}
		if apiErr.Status != 500 {
			t.Fatalf("status = %d", apiErr.Status)
		}
	})
}
