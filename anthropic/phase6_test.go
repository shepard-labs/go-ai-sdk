package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
)

func TestPhase6BaseURLNormalization(t *testing.T) {
	tests := map[string]string{
		"":                                defaultBaseURL,
		"https://api.anthropic.com":       defaultBaseURL,
		"https://api.anthropic.com/":      defaultBaseURL,
		"https://api.anthropic.com/v1":    "https://api.anthropic.com/v1",
		"https://api.anthropic.com/path/": "https://api.anthropic.com/path/",
		"https://proxy.example/":          "https://proxy.example/",
	}
	for input, want := range tests {
		got := CreateAnthropic(ProviderSettings{BaseURL: input}).(*anthropicProvider).baseURL
		if got != want {
			t.Fatalf("base URL %q = %q, want %q", input, got, want)
		}
	}
}

func TestPhase6AdversarialModelIDs(t *testing.T) {
	tests := []struct {
		id            string
		known, claude bool
		caps          ModelCapabilities
	}{
		{"my-claude-sonnet-4-5", false, false, ModelCapabilities{MaxOutputTokens: 4096}},
		{"claude-sonnet-4-unknown", false, true, ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true}},
		{"claude-sonnet-4-5-20250929", true, true, ModelCapabilities{MaxOutputTokens: 64000, StructuredOutput: true}},
		{"claude-3-haiku-20240307", true, true, ModelCapabilities{MaxOutputTokens: 4096}},
		{"claude-sonnet-6", false, true, ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true}},
		{"legacy/claude-sonnet-4-5", false, false, ModelCapabilities{MaxOutputTokens: 4096}},
	}
	for _, tc := range tests {
		t.Run(tc.id, func(t *testing.T) {
			caps, known := modelCapabilities(tc.id)
			if known != tc.known || isClaudeModel(tc.id) != tc.claude || caps != tc.caps {
				t.Fatalf("caps = %#v, known = %v, claude = %v", caps, known, isClaudeModel(tc.id))
			}
		})
	}
}

type captureFetcher struct{ request *http.Request }

func (f *captureFetcher) Do(req *http.Request) (*http.Response, error) {
	f.request = req
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"id":"id","content":[],"usage":{}}`))}, nil
}

func TestPhase6EndpointJoinsPathAndPreservesQuery(t *testing.T) {
	fetch := &captureFetcher{}
	p := CreateAnthropic(ProviderSettings{BaseURL: "https://proxy.example/custom/v1/?tenant=a#configured", Fetch: fetch})
	_, err := p.Model("custom").DoGenerate(context.Background(), GenerateOptions{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if got := fetch.request.URL.Path; got != "/custom/v1/messages" {
		t.Fatalf("path = %q", got)
	}
	if got := fetch.request.URL.RawQuery; got != "tenant=a" {
		t.Fatalf("query = %q", got)
	}
	if strings.Contains(fetch.request.URL.RequestURI(), "#") {
		t.Fatalf("fragment transmitted in %q", fetch.request.URL.RequestURI())
	}
	if got := p.(*anthropicProvider).baseURL; got != "https://proxy.example/custom/v1/?tenant=a#configured" {
		t.Fatalf("base URL = %q", got)
	}
}

func TestPhase6InvalidBaseURLTypedError(t *testing.T) {
	for _, baseURL := range []string{"://bad", "relative/path"} {
		p := CreateAnthropic(ProviderSettings{BaseURL: baseURL})
		var target *InvalidBaseURLError
		if !errors.As(p.Err(), &target) {
			t.Fatalf("provider error for %q = %v", baseURL, p.Err())
		}
	}
}

func TestPhase6InvalidBaseURLPrecedesMultipleAuthMethods(t *testing.T) {
	p := CreateAnthropic(ProviderSettings{BaseURL: "://bad", APIKey: "key", AuthToken: "token"})
	var target *InvalidBaseURLError
	if !errors.As(p.Err(), &target) {
		t.Fatalf("provider error = %v, want *InvalidBaseURLError", p.Err())
	}
}

func TestPhase6SamplingExclusivityIsClaudeOnly(t *testing.T) {
	temperature, topP := 0.5, 0.8
	claude := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	req, warnings := claude.buildRequest(GenerateOptions{MaxTokens: 10, Temperature: &temperature, TopP: &topP}, false)
	if req.TopP != nil || len(warnings) == 0 {
		t.Fatalf("Claude topP/warnings = %#v/%#v", req.TopP, warnings)
	}
	custom := CreateAnthropic(ProviderSettings{}).Model("minimax-custom").(*anthropicLanguageModel)
	req, warnings = custom.buildRequest(GenerateOptions{MaxTokens: 10, Temperature: &temperature, TopP: &topP}, false)
	if req.Temperature == nil || req.TopP == nil || len(warnings) != 0 {
		t.Fatalf("custom sampling/warnings = %#v/%#v/%#v", req.Temperature, req.TopP, warnings)
	}
}

func TestPhase6ToolSearchBetaIsExplicitOnlyAndDeduped(t *testing.T) {
	p := CreateAnthropic(ProviderSettings{}).(*anthropicProvider)
	opts := ModelOptions{RequestTools: []Tool{{ID: "anthropic.tool_search_regex_20251119"}}}
	if got := p.headersForOptions(opts).Values("anthropic-beta"); len(got) != 0 {
		t.Fatalf("automatic tool search beta = %#v", got)
	}
	opts.AnthropicBeta = []string{"tool-search-2025-11-19", "tool-search-2025-11-19"}
	got := p.headersForOptions(opts).Values("anthropic-beta")
	if len(got) != 1 || got[0] != "tool-search-2025-11-19" {
		t.Fatalf("explicit tool search beta = %#v", got)
	}
}

func TestPhase6MaxTokensDefaultsThinkingAndCaps(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	tests := []struct {
		name     string
		max      int
		thinking *ThinkingConfig
		want     int
	}{
		{"zero", 0, nil, 4096},
		{"positive", 100, nil, 100},
		{"enabled_positive", 100, &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 1024}, 1124},
		{"adaptive", 0, &ThinkingConfig{Type: ThinkingTypeAdaptive}, 4096},
		{"disabled", 0, &ThinkingConfig{Type: ThinkingTypeDisabled}, 4096},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := model.buildRequest(GenerateOptions{MaxTokens: tc.max, Thinking: tc.thinking}, false)
			if req.MaxTokens != tc.want || req.MaxTokens == 0 {
				t.Fatalf("max_tokens = %d, want %d", req.MaxTokens, tc.want)
			}
		})
	}
	unknown := CreateAnthropic(ProviderSettings{}).Model("custom-model").(*anthropicLanguageModel)
	req, warnings := unknown.buildRequest(GenerateOptions{}, false)
	if req.MaxTokens != 4096 || len(warnings) != 1 || warnings[0].Type != "compatibility" {
		t.Fatalf("unknown default = %d, warnings %#v", req.MaxTokens, warnings)
	}
	unknownClaude := CreateAnthropic(ProviderSettings{}).Model("claude-future-model").(*anthropicLanguageModel)
	req, warnings = unknownClaude.buildRequest(GenerateOptions{}, false)
	if req.MaxTokens != 128000 || len(warnings) != 1 || warnings[0].Type != "compatibility" {
		t.Fatalf("unknown Claude default = %d, warnings %#v", req.MaxTokens, warnings)
	}
}

func TestPhase6ThinkingKnownModelCapValidation(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	tests := []struct {
		name   string
		max    int
		budget int
	}{
		{"budget_equals_cap", 1, 4096},
		{"budget_exceeds_cap", 1, 4097},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, warnings, _, _, err := model.buildRequestWithOptions(GenerateOptions{
				MaxTokens: tc.max,
				Thinking:  &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: tc.budget},
			}, false)
			var limitErr *ThinkingTokenLimitError
			if !errors.As(err, &limitErr) {
				t.Fatalf("error = %T %v, want *ThinkingTokenLimitError", err, err)
			}
			if limitErr.BudgetTokens != tc.budget || limitErr.MaxOutputTokens != 4096 {
				t.Fatalf("error = %#v", limitErr)
			}
			if len(warnings) != 0 {
				t.Fatalf("warnings = %#v", warnings)
			}
		})
	}

	req, warnings, _, _, err := model.buildRequestWithOptions(GenerateOptions{
		Thinking: &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 1024},
	}, false)
	if err != nil || req.MaxTokens != 4096 || len(warnings) != 0 {
		t.Fatalf("default overflow request = %#v, warnings = %#v, err = %v", req, warnings, err)
	}

	req, warnings, _, _, err = model.buildRequestWithOptions(GenerateOptions{
		MaxTokens: 3073,
		Thinking:  &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 1024},
	}, false)
	if err != nil || req.MaxTokens != 4096 || len(warnings) != 1 || warnings[0].Type != "unsupported-setting" {
		t.Fatalf("explicit overflow request = %#v, warnings = %#v, err = %v", req, warnings, err)
	}

	req, warnings, _, _, err = model.buildRequestWithOptions(GenerateOptions{
		MaxTokens: 1,
		Thinking:  &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 4095},
	}, false)
	if err != nil || req.MaxTokens != 4096 || len(warnings) != 0 {
		t.Fatalf("near-cap request = %#v, warnings = %#v, err = %v", req, warnings, err)
	}
}

func TestPhase6ThinkingBudgetMinimum(t *testing.T) {
	var attempts atomic.Int32
	model := CreateAnthropic(ProviderSettings{Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		attempts.Add(1)
		return nil, errors.New("unexpected transport")
	}}}).Model("custom-model").(*anthropicLanguageModel)

	for _, budget := range []int{-1, 1, minimumThinkingBudgetTokens - 1} {
		t.Run(fmt.Sprintf("invalid_%d", budget), func(t *testing.T) {
			_, _, _, _, err := model.buildRequestWithOptions(GenerateOptions{
				MaxTokens: 1,
				Thinking:  &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: budget},
			}, false)
			var constructionErr *RequestConstructionError
			if !errors.As(err, &constructionErr) {
				t.Fatalf("error = %T %v, want *RequestConstructionError", err, err)
			}
		})
	}

	for _, tc := range []struct {
		name, wantWarning string
		budget            int
	}{
		{name: "zero_defaults", budget: 0, wantWarning: "compatibility"},
		{name: "minimum_valid", budget: minimumThinkingBudgetTokens},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req, warnings, _, _, err := model.buildRequestWithOptions(GenerateOptions{
				MaxTokens: 1,
				Thinking:  &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: tc.budget},
			}, false)
			if err != nil || req.Thinking == nil || req.Thinking.BudgetTokens != minimumThinkingBudgetTokens || req.MaxTokens != minimumThinkingBudgetTokens+1 {
				t.Fatalf("request = %#v, warnings = %#v, err = %v", req, warnings, err)
			}
			if tc.wantWarning == "" && len(warnings) != 0 || tc.wantWarning != "" && (len(warnings) != 1 || warnings[0].Type != tc.wantWarning) {
				t.Fatalf("warnings = %#v", warnings)
			}
		})
	}

	_, err := model.DoGenerate(context.Background(), GenerateOptions{
		MaxTokens: 1,
		Thinking:  &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: minimumThinkingBudgetTokens - 1},
	})
	var constructionErr *RequestConstructionError
	if !errors.As(err, &constructionErr) || attempts.Load() != 0 {
		t.Fatalf("error = %T %v, transport attempts = %d", err, err, attempts.Load())
	}
}

func TestPhase6ThinkingCapFailurePreservesWarningsAndSkipsTransport(t *testing.T) {
	var attempts atomic.Int32
	temperature := 0.5
	model := CreateAnthropic(ProviderSettings{Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		attempts.Add(1)
		return nil, errors.New("unexpected transport")
	}}}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)

	_, warnings, _, _, err := model.buildRequestWithOptions(GenerateOptions{
		MaxTokens:   1,
		Temperature: &temperature,
		Thinking:    &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 4096},
	}, false)
	var limitErr *ThinkingTokenLimitError
	if !errors.As(err, &limitErr) || len(warnings) != 1 || warnings[0].Type != "unsupported-setting" {
		t.Fatalf("error = %T %v, warnings = %#v", err, err, warnings)
	}

	_, err = model.DoGenerate(context.Background(), GenerateOptions{
		MaxTokens: 1,
		Thinking:  &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 4096},
	})
	if !errors.As(err, &limitErr) || attempts.Load() != 0 {
		t.Fatalf("error = %T %v, transport attempts = %d", err, err, attempts.Load())
	}
}

func TestPhase6StopReasons(t *testing.T) {
	tests := map[string]FinishReason{
		"pause_turn":                    FinishReasonStop,
		"end_turn":                      FinishReasonStop,
		"stop_sequence":                 FinishReasonStop,
		"refusal":                       FinishReasonContentFilter,
		"tool_use":                      FinishReasonToolCalls,
		"max_tokens":                    FinishReasonLength,
		"model_context_window_exceeded": FinishReasonLength,
		"compaction":                    FinishReason("other"),
		"future":                        FinishReasonUnknown,
	}
	for raw, want := range tests {
		if got := FinishReasonFromStopReason(raw); got != want {
			t.Fatalf("stop reason %q = %q, want %q", raw, got, want)
		}
	}
}

func TestPhase6ResponseBodyLimitConfiguration(t *testing.T) {
	responseBody := `{"id":"id","content":[],"usage":{}}`
	for _, tc := range []struct {
		name       string
		limit      int64
		wantErr    bool
		wantLength int
	}{
		{"configured_limit", 8, true, 8},
		{"zero_uses_default_limit", 0, false, 0},
		{"negative_uses_default_limit", -1, false, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := CreateAnthropic(ProviderSettings{
				Fetch:                roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return jsonResponse(http.StatusOK, responseBody), nil }},
				MaxResponseBodyBytes: tc.limit,
			}).Model("claude-3-haiku-20240307")

			_, err := model.DoGenerate(context.Background(), GenerateOptions{})
			var apiErr *APICallError
			if errors.As(err, &apiErr) != tc.wantErr {
				t.Fatalf("error = %#v, want limited = %v", err, tc.wantErr)
			}
			if tc.wantErr && (!apiErr.Truncated || len(apiErr.Body) != tc.wantLength) {
				t.Fatalf("error = %#v", apiErr)
			}
			if !tc.wantErr && err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPhase6ErrorBodyLimitConfiguration(t *testing.T) {
	errorBody := strings.Repeat("x", 64)
	for _, tc := range []struct {
		name       string
		limit      int64
		wantLength int
	}{
		{"configured_limit", 8, 8},
		{"zero_uses_default_limit", 0, len(errorBody)},
		{"negative_uses_default_limit", -1, len(errorBody)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := CreateAnthropic(ProviderSettings{
				Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
					return jsonResponse(http.StatusInternalServerError, errorBody), nil
				}},
				MaxErrorResponseBytes: tc.limit,
				Retry:                 &RetryOptions{MaxRetries: 0},
			}).Model("claude-3-haiku-20240307")

			_, err := model.DoGenerate(context.Background(), GenerateOptions{})
			var apiErr *APICallError
			if !errors.As(err, &apiErr) || len(apiErr.Body) != tc.wantLength || apiErr.Truncated != (tc.limit > 0) {
				t.Fatalf("error = %#v", err)
			}
		})
	}
}
