package adapters

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"testing"
	"time"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/openai"
)

type fakeOpenAIModel struct {
	lastOptions openai.GenerateOptions
	result      *openai.GenerateResult
	err         error
}

func TestOpenAIAdapterNeutralReasoningMapsToProviderOptions(t *testing.T) {
	model := &fakeOpenAIModel{result: &openai.GenerateResult{FinishReason: openai.FinishReason{Unified: "stop"}}}
	result, err := NewOpenAIAdapter(model).Generate(context.Background(), GenerateOptions{
		ProviderOptions: llm.ProviderOptions{"openai": {"reasoningEffort": "low"}},
		Reasoning:       &llm.ReasoningOptions{Effort: llm.ReasoningHigh},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if got := model.lastOptions.ProviderOptions["openai"]["reasoningEffort"]; got != "high" {
		t.Fatalf("reasoningEffort = %#v, want high", got)
	}
	foundOverrideWarning := false
	for _, warning := range result.Warnings {
		if warning.Code == "reasoning_provider_option_overridden" {
			foundOverrideWarning = true
		}
	}
	if !foundOverrideWarning {
		t.Fatalf("warnings = %#v, want override warning", result.Warnings)
	}
}

func TestOpenAIAdapterNeutralReasoningBudgetUnsupported(t *testing.T) {
	budget := 100
	model := &fakeOpenAIModel{result: &openai.GenerateResult{FinishReason: openai.FinishReason{Unified: "stop"}}}
	_, err := NewOpenAIAdapter(model).Generate(context.Background(), GenerateOptions{Reasoning: &llm.ReasoningOptions{Effort: llm.ReasoningHigh, BudgetTokens: &budget}})
	var ufe *llm.UnsupportedFeatureError
	if !errors.As(err, &ufe) || ufe.Feature != "reasoning_budget" {
		t.Fatalf("error = %v, want UnsupportedFeatureError(reasoning_budget)", err)
	}
}

func TestOpenAIAdapterNeutralReasoningXHighWarnDowngrades(t *testing.T) {
	model := &fakeOpenAIModel{result: &openai.GenerateResult{FinishReason: openai.FinishReason{Unified: "stop"}}}
	result, err := NewOpenAIAdapter(model).Generate(context.Background(), GenerateOptions{
		Reasoning:                &llm.ReasoningOptions{Effort: llm.ReasoningXHigh},
		UnsupportedFeaturePolicy: llm.UnsupportedFeaturePolicyWarn,
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if got := model.lastOptions.ProviderOptions["openai"]["reasoningEffort"]; got != "high" {
		t.Fatalf("reasoningEffort = %#v, want high", got)
	}
	if len(result.Warnings) == 0 || result.Warnings[0].Code != "reasoning_effort_downgraded" {
		t.Fatalf("warnings = %#v, want downgrade warning", result.Warnings)
	}
}

func (f *fakeOpenAIModel) ModelID() string                          { return "fake" }
func (f *fakeOpenAIModel) Provider() string                         { return "fake" }
func (f *fakeOpenAIModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeOpenAIModel) DoGenerate(ctx context.Context, opts openai.GenerateOptions) (*openai.GenerateResult, error) {
	f.lastOptions = opts
	return f.result, f.err
}
func (f *fakeOpenAIModel) DoStream(ctx context.Context, opts openai.StreamOptions) (*openai.StreamResult, error) {
	return nil, nil
}

func TestOpenAIAdapterTranslatesAndReturns(t *testing.T) {
	model := &fakeOpenAIModel{result: &openai.GenerateResult{
		FinishReason: openai.FinishReason{Unified: "tool-calls"},
		Content: []openai.Content{
			openai.TextContent{Text: "hi"},
			openai.ToolCallContent{ToolCallContentEmbed: openai.ToolCallContentEmbed{ToolCallID: "id", ToolName: "t", Input: json.RawMessage(`{}`)}},
		},
		Usage: openai.Usage{InputTokens: openai.TokenCounts{Total: intPtr(3)}, OutputTokens: openai.OutputTokenCounts{Total: intPtr(4)}},
	}}
	result, err := NewOpenAIAdapter(model).Generate(context.Background(), GenerateOptions{
		System:   "sys",
		Messages: []Message{{Role: "user", Content: []Content{TextContent{Text: "q"}}}},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if _, ok := model.lastOptions.Messages[0].(openai.SystemMessage); !ok {
		t.Fatalf("message 0 = %#v, want SystemMessage", model.lastOptions.Messages[0])
	}
	if result.FinishReason.Unified != FinishReasonToolCalls {
		t.Fatalf("finish = %q", result.FinishReason)
	}
	if _, ok := result.Content[1].(ToolUseContent); !ok {
		t.Fatalf("content 1 = %#v, want ToolUseContent", result.Content[1])
	}
	if result.Usage.InputTokens != 3 || result.Usage.OutputTokens != 4 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}

func TestOpenAINewFieldsWiring(t *testing.T) {
	ts := time.Now()
	model := &fakeOpenAIModel{result: &openai.GenerateResult{
		FinishReason: openai.FinishReason{Unified: "stop"},
		Response:     openai.ResponseMetadata{ID: "resp_9", ModelID: "gpt-x", Timestamp: &ts, Headers: http.Header{"X-Id": []string{"z"}}},
	}}
	result, err := NewOpenAIAdapter(model).Generate(context.Background(), GenerateOptions{
		ToolChoice:      ToolChoice{Type: llm.ToolChoiceTool, ToolName: "fn"},
		ResponseFormat:  &ResponseFormat{Type: llm.ResponseFormatJSONObject},
		ProviderOptions: llm.ProviderOptions{"openai": {"k": "v"}},
		Headers:         map[string]string{"X-Test": "1"},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if model.lastOptions.ToolChoice == nil || model.lastOptions.ToolChoice.Type != "tool" || model.lastOptions.ToolChoice.ToolName != "fn" {
		t.Fatalf("tool choice = %#v", model.lastOptions.ToolChoice)
	}
	if model.lastOptions.ResponseFormat == nil || model.lastOptions.ResponseFormat.Type != "json" {
		t.Fatalf("response format = %#v", model.lastOptions.ResponseFormat)
	}
	if model.lastOptions.ProviderOptions["openai"]["k"] != "v" {
		t.Fatalf("provider options = %#v", model.lastOptions.ProviderOptions)
	}
	if model.lastOptions.Headers.Get("X-Test") != "1" {
		t.Fatalf("headers = %#v", model.lastOptions.Headers)
	}
	if result.Response.ID != "resp_9" || result.Response.ModelID != "gpt-x" || result.Response.Headers["X-Id"] != "z" {
		t.Fatalf("response metadata = %#v", result.Response)
	}
}
