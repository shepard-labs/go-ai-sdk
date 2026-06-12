package anthropic

import (
	"errors"
	"net/http"
	"reflect"
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
		"claude-opus-4-8":           {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true},
		"claude-opus-4-7":           {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true},
		"claude-sonnet-4-6":         {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true},
		"claude-opus-4-6":           {MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true},
		"claude-sonnet-4-5":         {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-opus-4-5":           {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-haiku-4-5":          {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-haiku-4-5-20251001": {MaxOutputTokens: 64000, StructuredOutput: true},
		"claude-opus-4-1":           {MaxOutputTokens: 32000, StructuredOutput: true},
		"claude-sonnet-4":           {MaxOutputTokens: 64000},
		"claude-opus-4":             {MaxOutputTokens: 32000},
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
