package adapters

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/llm"
)

func TestREQADAPTER001_GoAISDKImportOnlyInAdapter(t *testing.T) {
	var offenders []string
	err := filepath.WalkDir("..", func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if strings.Contains(string(data), "github.com/shepard-labs/go-ai-sdk/anthropic") && !strings.HasPrefix(path, filepath.Join("..", "adapters")) {
			offenders = append(offenders, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk llm: %v", err)
	}
	if len(offenders) > 0 {
		t.Fatalf("anthropic imports outside llm/adapters: %v", offenders)
	}
}

type fakeAnthropicModel struct {
	lastOptions anthropic.GenerateOptions
	result      *anthropic.GenerateResult
	err         error
}

func (f *fakeAnthropicModel) ModelID() string                          { return "fake" }
func (f *fakeAnthropicModel) Provider() string                         { return "fake" }
func (f *fakeAnthropicModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeAnthropicModel) DoGenerate(ctx context.Context, opts anthropic.GenerateOptions) (*anthropic.GenerateResult, error) {
	f.lastOptions = opts
	return f.result, f.err
}
func (f *fakeAnthropicModel) DoStream(ctx context.Context, opts anthropic.StreamOptions) (*anthropic.StreamResult, error) {
	return nil, nil
}

func TestREQADAPTER002_APICallError429529Unwrapped(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, 529} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			want := &anthropic.APICallError{Status: status, Message: "retry"}
			client := NewAnthropicAdapter(&fakeAnthropicModel{err: want})
			_, err := client.Generate(context.Background(), GenerateOptions{})
			var got *anthropic.APICallError
			if !errors.As(err, &got) || got != want {
				t.Fatalf("error = %v, want original APICallError", err)
			}
		})
	}
}

func TestREQADAPTER003_UnknownFinishReasonMapsToError(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReason("new")}}
	result, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if result.FinishReason.Unified != FinishReasonError {
		t.Fatalf("finish = %q, want %q", result.FinishReason.Unified, FinishReasonError)
	}
}

func TestREQADAPTER003_UnknownRoleReturnsError(t *testing.T) {
	_, err := NewAnthropicAdapter(&fakeAnthropicModel{}).Generate(context.Background(), GenerateOptions{Messages: []Message{{Role: "system"}}})
	if err == nil {
		t.Fatal("Generate error = nil, want unknown role error")
	}
}

func TestREQADAPTER004_GeneratorFuncSatisfiesClient(t *testing.T) {
	var _ Client = GeneratorFunc(func(context.Context, GenerateOptions) (*GenerateResult, error) { return nil, nil })
}

func TestREQADAPTER_SystemMessageConversion(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	_, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{System: "system"})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	message, ok := model.lastOptions.Messages[0].(anthropic.SystemMessage)
	if !ok || message.Content != "system" {
		t.Fatalf("first message = %#v, want SystemMessage", model.lastOptions.Messages[0])
	}
}

func TestREQADAPTER_UserAndAssistantMessageConversion(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	_, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{Messages: []Message{
		{Role: "user", Content: []Content{TextContent{Text: "hello"}}},
		{Role: "assistant", Content: []Content{TextContent{Text: "hi"}}},
	}})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if _, ok := model.lastOptions.Messages[0].(anthropic.UserMessage); !ok {
		t.Fatalf("message 0 = %#v, want UserMessage", model.lastOptions.Messages[0])
	}
	if _, ok := model.lastOptions.Messages[1].(anthropic.AssistantMessage); !ok {
		t.Fatalf("message 1 = %#v, want AssistantMessage", model.lastOptions.Messages[1])
	}
}

func TestREQADAPTER_ToolUseAndToolResultConversion(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonToolCalls, Content: []anthropic.Content{
		anthropic.TextContent{Text: "use"},
		anthropic.ToolCallContent{ToolCallID: "call", ToolName: "tool", Input: []byte(`{"x":1}`)},
	}}}
	result, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{Messages: []Message{
		{Role: "assistant", Content: []Content{ToolUseContent{ID: "call", Name: "tool", Input: []byte(`{"x":1}`)}}},
		{Role: "user", Content: []Content{ToolResultContent{ToolUseID: "call", Text: "ok", IsError: true}}},
	}})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	assistant := model.lastOptions.Messages[0].(anthropic.AssistantMessage)
	if tool, ok := assistant.Content[0].(anthropic.ToolCallContent); !ok || tool.ToolCallID != "call" || tool.ToolName != "tool" {
		t.Fatalf("assistant content = %#v, want ToolCallContent", assistant.Content[0])
	}
	user := model.lastOptions.Messages[1].(anthropic.UserMessage)
	if tool, ok := user.Content[0].(anthropic.ToolResultContent); !ok || tool.ToolCallID != "call" || !tool.IsError {
		t.Fatalf("user content = %#v, want ToolResultContent", user.Content[0])
	}
	if len(result.Content) != 2 || result.FinishReason.Unified != FinishReasonToolCalls {
		t.Fatalf("result = %#v", result)
	}
}

func TestREQADAPTER_ToolSchemaConversion(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	_, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{Tools: []Tool{{Name: "tool", Description: "desc", InputSchema: []byte(`{"type":"object"}`)}}})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	schema, ok := model.lastOptions.Tools[0].InputSchema.(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("schema = %#v, want decoded map", model.lastOptions.Tools[0].InputSchema)
	}
}

func TestAnthropicToolChoiceMapping(t *testing.T) {
	cases := []struct {
		name     string
		choice   ToolChoice
		wantType string
		wantName string
	}{
		{"auto", ToolChoice{Type: llm.ToolChoiceAuto}, "auto", ""},
		{"required", ToolChoice{Type: llm.ToolChoiceRequired}, "any", ""},
		{"tool", ToolChoice{Type: llm.ToolChoiceTool, ToolName: "lookup"}, "tool", "lookup"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
			_, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{ToolChoice: tc.choice})
			if err != nil {
				t.Fatalf("Generate error = %v", err)
			}
			if model.lastOptions.ToolChoice == nil {
				t.Fatalf("ToolChoice = nil, want %q", tc.wantType)
			}
			if model.lastOptions.ToolChoice.Type != tc.wantType || model.lastOptions.ToolChoice.Name != tc.wantName {
				t.Fatalf("ToolChoice = %#v, want type %q name %q", model.lastOptions.ToolChoice, tc.wantType, tc.wantName)
			}
		})
	}
}

func TestAnthropicToolChoiceNoneUnsupported(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	_, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{ToolChoice: ToolChoice{Type: llm.ToolChoiceNone}})
	var ufe *llm.UnsupportedFeatureError
	if !errors.As(err, &ufe) || ufe.Feature != "tool_choice_none" {
		t.Fatalf("error = %v, want UnsupportedFeatureError(tool_choice_none)", err)
	}
}

func TestAnthropicResponseFormatUnsupportedError(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	_, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{ResponseFormat: &ResponseFormat{Type: llm.ResponseFormatJSONObject}})
	var ufe *llm.UnsupportedFeatureError
	if !errors.As(err, &ufe) || ufe.Feature != "response_format" {
		t.Fatalf("error = %v, want UnsupportedFeatureError(response_format)", err)
	}
}

func TestAnthropicUnsupportedFeatureWarnPolicy(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	result, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{
		ResponseFormat:           &ResponseFormat{Type: llm.ResponseFormatJSONObject},
		UnsupportedFeaturePolicy: llm.UnsupportedFeaturePolicyWarn,
	})
	if err != nil {
		t.Fatalf("Generate error = %v, want nil under warn policy", err)
	}
	if len(result.Warnings) == 0 || result.Warnings[0].Provider != "anthropic" {
		t.Fatalf("warnings = %#v, want anthropic unsupported warning", result.Warnings)
	}
}

func TestAnthropicNeutralReasoningMapsToThinking(t *testing.T) {
	budget := 1234
	cases := []struct {
		name       string
		reasoning  *llm.ReasoningOptions
		wantType   anthropic.ThinkingType
		wantBudget int
	}{
		{"none", &llm.ReasoningOptions{Effort: llm.ReasoningNone}, anthropic.ThinkingTypeDisabled, 0},
		{"high", &llm.ReasoningOptions{Effort: llm.ReasoningHigh}, anthropic.ThinkingTypeEnabled, 8192},
		{"exact_budget", &llm.ReasoningOptions{BudgetTokens: &budget}, anthropic.ThinkingTypeEnabled, budget},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
			result, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{
				Reasoning:                tc.reasoning,
				UnsupportedFeaturePolicy: llm.UnsupportedFeaturePolicyWarn,
			})
			if err != nil {
				t.Fatalf("Generate error = %v", err)
			}
			thinking := anthropicThinkingFromOptions(model.lastOptions)
			if thinking == nil && !anthropicGenerateOptionsHasThinking() {
				if len(result.Warnings) == 0 || result.Warnings[0].Provider != "anthropic" || result.Warnings[0].Code != "unsupported_feature" {
					t.Fatalf("warnings = %#v, want unsupported Anthropic reasoning warning", result.Warnings)
				}
				return
			}
			if thinking == nil {
				t.Fatal("Thinking = nil")
			}
			if thinking.Type != tc.wantType || thinking.BudgetTokens != tc.wantBudget {
				t.Fatalf("Thinking = %#v, want type %q budget %d", thinking, tc.wantType, tc.wantBudget)
			}
		})
	}
}

func anthropicGenerateOptionsHasThinking() bool {
	_, ok := reflect.TypeOf(anthropic.GenerateOptions{}).FieldByName("Thinking")
	return ok
}

func anthropicThinkingFromOptions(opts anthropic.GenerateOptions) *anthropic.ThinkingConfig {
	field := reflect.ValueOf(opts).FieldByName("Thinking")
	if !field.IsValid() || field.IsNil() {
		return nil
	}
	thinking, _ := field.Interface().(*anthropic.ThinkingConfig)
	return thinking
}

func TestAnthropicNeutralReasoningValidation(t *testing.T) {
	budget := 1
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	_, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{Reasoning: &llm.ReasoningOptions{Effort: llm.ReasoningNone, BudgetTokens: &budget}})
	var ufe *llm.UnsupportedFeatureError
	if !errors.As(err, &ufe) || ufe.Feature != "reasoning_budget" {
		t.Fatalf("error = %v, want UnsupportedFeatureError(reasoning_budget)", err)
	}
}

func TestAnthropicResponseMetadata(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{
		FinishReason:    anthropic.FinishReasonStop,
		MessageMetadata: anthropic.MessageMetadata{"id": "msg_123", "model": "claude-x"},
	}}
	result, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if result.Response.ID != "msg_123" || result.Response.ModelID != "claude-x" {
		t.Fatalf("response metadata = %#v", result.Response)
	}
}

func TestAnthropicWarningsPropagate(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{
		FinishReason: anthropic.FinishReasonStop,
		Warnings:     []anthropic.Warning{{Type: "x", Message: "heads up"}},
	}}
	result, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if len(result.Warnings) != 1 || result.Warnings[0].Message != "heads up" || result.Warnings[0].Provider != "anthropic" {
		t.Fatalf("warnings = %#v", result.Warnings)
	}
}

func TestREQADAPTER_UsageConversion(t *testing.T) {
	model := &fakeAnthropicModel{result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop, Usage: anthropic.Usage{InputTokens: anthropic.TokenUsage{Total: 7}, OutputTokens: anthropic.TokenUsage{Total: 11}}}}
	result, err := NewAnthropicAdapter(model).Generate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if result.Usage.InputTokens != 7 || result.Usage.OutputTokens != 11 {
		t.Fatalf("usage = %#v, want 7/11", result.Usage)
	}
}
