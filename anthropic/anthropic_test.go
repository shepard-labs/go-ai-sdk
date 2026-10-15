package anthropic

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func TestCreateAnthropicProvider(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	p := CreateAnthropic(ProviderSettings{})
	if p == nil {
		t.Fatal("expected provider")
	}
	if p.Err() != nil {
		t.Fatalf("unexpected provider error: %v", p.Err())
	}
	if p.Name() != "anthropic" {
		t.Fatalf("unexpected provider name: %q", p.Name())
	}
	headers := p.(*anthropicProvider).headersForOptions(ModelOptions{})
	if got := headers.Get("anthropic-version"); got != anthropicVersion {
		t.Fatalf("anthropic-version = %q, want %q", got, anthropicVersion)
	}
	if got := headers.Get("User-Agent"); got != "ai-sdk-go/anthropic/"+Version {
		t.Fatalf("User-Agent = %q", got)
	}
}

func TestCreateAnthropicProviderWithAPIKey(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	p := CreateAnthropic(ProviderSettings{APIKey: "key"}).(*anthropicProvider)
	headers := p.headersForOptions(ModelOptions{})
	if got := headers.Get("x-api-key"); got != "key" {
		t.Fatalf("x-api-key = %q", got)
	}
	if got := headers.Get("Authorization"); got != "" {
		t.Fatalf("Authorization = %q", got)
	}
}

func TestCreateAnthropicProviderWithAuthToken(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	p := CreateAnthropic(ProviderSettings{AuthToken: "token"}).(*anthropicProvider)
	headers := p.headersForOptions(ModelOptions{})
	if got := headers.Get("Authorization"); got != "Bearer token" {
		t.Fatalf("Authorization = %q", got)
	}
	if got := headers.Get("x-api-key"); got != "" {
		t.Fatalf("x-api-key = %q", got)
	}
}

func TestCreateAnthropicProviderBothAuthMethods(t *testing.T) {
	p := CreateAnthropic(ProviderSettings{APIKey: "key", AuthToken: "token"})
	if !errors.Is(p.Err(), ErrMultipleAuthMethods) {
		t.Fatalf("Err() = %v, want %v", p.Err(), ErrMultipleAuthMethods)
	}
}

func TestCreateAnthropicProviderEnvFallback(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	p := CreateAnthropic(ProviderSettings{}).(*anthropicProvider)
	if got := p.headersForOptions(ModelOptions{}).Get("x-api-key"); got != "env-key" {
		t.Fatalf("x-api-key = %q", got)
	}

	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
	p = CreateAnthropic(ProviderSettings{}).(*anthropicProvider)
	if got := p.headersForOptions(ModelOptions{}).Get("Authorization"); got != "Bearer env-token" {
		t.Fatalf("Authorization = %q", got)
	}
}

func TestCreateAnthropicProviderCustomHeaders(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	headers := http.Header{"X-Test": []string{"value"}}
	p := CreateAnthropic(ProviderSettings{Headers: headers}).(*anthropicProvider)
	if got := p.headersForOptions(ModelOptions{}).Get("X-Test"); got != "value" {
		t.Fatalf("X-Test = %q", got)
	}
}

func TestAnthropicDefaultProvider(t *testing.T) {
	if Anthropic == nil {
		t.Fatal("expected default provider")
	}
}

func TestProviderModel(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307")
	if model == nil {
		t.Fatal("expected model")
	}
	if got := model.ModelID(); got != "claude-3-haiku-20240307" {
		t.Fatalf("ModelID() = %q", got)
	}
}

func TestProviderChat(t *testing.T) {
	if model := CreateAnthropic(ProviderSettings{}).Chat("claude-3-haiku-20240307"); model == nil {
		t.Fatal("expected chat model")
	}
}

func TestProviderMessages(t *testing.T) {
	if model := CreateAnthropic(ProviderSettings{}).Messages("claude-3-haiku-20240307"); model == nil {
		t.Fatal("expected messages model")
	}
}

func TestProviderTools(t *testing.T) {
	tools := CreateAnthropic(ProviderSettings{}).Tools()
	if tools.ToolNameMapping == nil {
		t.Fatal("expected tool name mapping")
	}
}

func TestModelCapabilitiesForID(t *testing.T) {
	tests := map[string]ModelCapabilities{
		"claude-sonnet-5-5":         {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true, RejectsThinkingDisabled: true, RejectsBudgetThinking: true, RejectsForcedToolUse: true, SupportsBetweenToolsThinking: true},
		"claude-opus-5-5":           {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true, RejectsThinkingDisabled: true, RejectsBudgetThinking: true, RejectsForcedToolUse: true},
		"claude-haiku-5-5":          {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true, RejectsBudgetThinking: true},
		"claude-fable-5-1":          {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true, RejectsThinkingDisabled: true, RejectsBudgetThinking: true, RejectsForcedToolUse: true},
		"claude-opus-4-8":           {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true},
		"claude-opus-4-7":           {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true},
		"claude-sonnet-4-6":         {MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true},
		"claude-opus-4-6":           {MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true},
		"claude-sonnet-4-5":         {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-opus-4-5":           {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-haiku-4-5":          {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-haiku-4-5-20251001": {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-opus-4-1":           {MaxOutputTokens: 32000, StructuredOutput: true},
		"claude-sonnet-4":           {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true},
		"claude-opus-4":             {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true},
		"claude-3-haiku-20240307":   {MaxOutputTokens: 4096},
	}
	for modelID, want := range tests {
		if got := ModelCapabilitiesForID(modelID); got != want {
			t.Fatalf("ModelCapabilitiesForID(%q) = %+v, want %+v", modelID, got, want)
		}
	}
}

func TestModelCapabilitiesUnknownModel(t *testing.T) {
	want := ModelCapabilities{MaxOutputTokens: 4096}
	if got := ModelCapabilitiesForID("unknown"); got != want {
		t.Fatalf("ModelCapabilitiesForID(unknown) = %+v, want %+v", got, want)
	}
}

func TestBetaHeadersAutoAdded(t *testing.T) {
	budget := 100
	p := CreateAnthropic(ProviderSettings{}).(*anthropicProvider)
	headers := p.headersForOptions(ModelOptions{
		StructuredOutputMode: StructuredOutputModeJSONTool,
		Container:            &Container{Skills: []Skill{{SkillID: "skill"}}},
		MCPServers:           []MCPServer{{Name: "mcp"}},
		ContextManagement:    &ContextManagement{Edits: []ContextManagementEdit{CompactEdit{Type: "compact"}}},
		TaskBudget:           &budget,
		Speed:                "fast",
	})
	want := []string{
		"structured-outputs-2025-11-13",
		"skills-2025-10-02",
		"files-api-2025-04-14",
		"mcp-client-2025-11-20",
		"context-management-2025-06-27",
		"compact-2026-01-12",
		"task-budgets-2026-03-13",
		"fast-mode-2026-02-01",
	}
	if got := headers.Values("anthropic-beta"); !reflect.DeepEqual(got, want) {
		t.Fatalf("anthropic-beta = %#v, want %#v", got, want)
	}
}

func TestBetaHeadersCustom(t *testing.T) {
	p := CreateAnthropic(ProviderSettings{}).(*anthropicProvider)
	headers := p.headersForOptions(ModelOptions{AnthropicBeta: []string{"custom-beta", "custom-beta"}})
	want := []string{"custom-beta"}
	if got := headers.Values("anthropic-beta"); !reflect.DeepEqual(got, want) {
		t.Fatalf("anthropic-beta = %#v, want %#v", got, want)
	}
}

func TestBaseURLNormalization(t *testing.T) {
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

func TestAdversarialModelIDs(t *testing.T) {
	unknownNewer := ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true, SupportsAdaptiveThinking: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true}
	tests := []struct {
		id            string
		known, claude bool
		caps          ModelCapabilities
	}{
		{"my-claude-sonnet-4-5", false, false, ModelCapabilities{MaxOutputTokens: 4096}},
		{"claude-sonnet-4-unknown", true, true, ModelCapabilities{MaxOutputTokens: 64000}},
		{"claude-sonnet-4-5-20250929", true, true, ModelCapabilities{MaxOutputTokens: 64000, StructuredOutput: true}},
		{"claude-3-haiku-20240307", true, true, ModelCapabilities{MaxOutputTokens: 4096}},
		{"claude-sonnet-6", false, true, unknownNewer},
		{"legacy/claude-sonnet-4-5", false, false, ModelCapabilities{MaxOutputTokens: 4096}},
		{"claude-sonnet-4", false, true, unknownNewer},
		{"claude-opus-4", false, true, unknownNewer},
		{"claude-sonnet-4-1", true, true, ModelCapabilities{MaxOutputTokens: 64000}},
		{"claude-opus-4@20250101", true, true, ModelCapabilities{MaxOutputTokens: 32000}},
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

func TestEndpointJoinsPathAndPreservesQuery(t *testing.T) {
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

func TestInvalidBaseURLTypedError(t *testing.T) {
	for _, baseURL := range []string{"://bad", "relative/path"} {
		p := CreateAnthropic(ProviderSettings{BaseURL: baseURL})
		var target *InvalidBaseURLError
		if !errors.As(p.Err(), &target) {
			t.Fatalf("provider error for %q = %v", baseURL, p.Err())
		}
	}
}

func TestInvalidBaseURLPrecedesMultipleAuthMethods(t *testing.T) {
	p := CreateAnthropic(ProviderSettings{BaseURL: "://bad", APIKey: "key", AuthToken: "token"})
	var target *InvalidBaseURLError
	if !errors.As(p.Err(), &target) {
		t.Fatalf("provider error = %v, want *InvalidBaseURLError", p.Err())
	}
}

func TestSamplingExclusivityIsClaudeOnly(t *testing.T) {
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

func TestToolSearchBetaIsExplicitOnlyAndDeduped(t *testing.T) {
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

func TestStopReasons(t *testing.T) {
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
