package adapters

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"testing"
	"time"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

type fakeOpenAICompatibleModel struct {
	modelID       string
	generateCalls int
	lastOptions   openaicompatible.GenerateOptions
	result        *openaicompatible.GenerateResult
	stream        *openaicompatible.StreamResult
	err           error
}

func (f *fakeOpenAICompatibleModel) ModelID() string {
	if f.modelID != "" {
		return f.modelID
	}
	return "fake"
}
func (f *fakeOpenAICompatibleModel) Provider() string                         { return "fake" }
func (f *fakeOpenAICompatibleModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeOpenAICompatibleModel) DoGenerate(ctx context.Context, opts openaicompatible.GenerateOptions) (*openaicompatible.GenerateResult, error) {
	f.generateCalls++
	f.lastOptions = opts
	return f.result, f.err
}
func (f *fakeOpenAICompatibleModel) DoStream(ctx context.Context, opts openaicompatible.StreamOptions) (*openaicompatible.StreamResult, error) {
	return f.stream, f.err
}

func intPtr(v int) *int { return &v }

func TestOpenAICompatibleSystemAndMaxTokens(t *testing.T) {
	model := &fakeOpenAICompatibleModel{result: &openaicompatible.GenerateResult{FinishReason: openaicompatible.FinishReason{Unified: "stop"}}}
	_, err := NewOpenAICompatibleAdapter(model).Generate(context.Background(), GenerateOptions{System: "sys", MaxTokens: 64})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	sys, ok := model.lastOptions.Messages[0].(openaicompatible.SystemMessage)
	if !ok || sys.Content != "sys" {
		t.Fatalf("message 0 = %#v, want SystemMessage", model.lastOptions.Messages[0])
	}
	if model.lastOptions.MaxOutputTokens == nil || *model.lastOptions.MaxOutputTokens != 64 {
		t.Fatalf("max output tokens = %v, want 64", model.lastOptions.MaxOutputTokens)
	}
}

func TestOpenAICompatibleToolResultBecomesToolMessage(t *testing.T) {
	model := &fakeOpenAICompatibleModel{result: &openaicompatible.GenerateResult{FinishReason: openaicompatible.FinishReason{Unified: "stop"}}}
	_, err := NewOpenAICompatibleAdapter(model).Generate(context.Background(), GenerateOptions{Messages: []Message{
		{Role: "assistant", Content: []Content{ToolUseContent{ID: "call", Name: "tool", Input: json.RawMessage(`{"x":1}`)}}},
		{Role: "user", Content: []Content{
			TextContent{Text: "here"},
			ToolResultContent{ToolUseID: "call", Text: "boom", IsError: true},
		}},
	}})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	assistant, ok := model.lastOptions.Messages[0].(openaicompatible.AssistantMessage)
	if !ok {
		t.Fatalf("message 0 = %#v, want AssistantMessage", model.lastOptions.Messages[0])
	}
	if call, ok := assistant.Content[0].(openaicompatible.ToolCallContent); !ok || call.ToolCallID != "call" {
		t.Fatalf("assistant content = %#v, want ToolCallContent", assistant.Content[0])
	}
	if _, ok := model.lastOptions.Messages[1].(openaicompatible.UserMessage); !ok {
		t.Fatalf("message 1 = %#v, want UserMessage", model.lastOptions.Messages[1])
	}
	toolMsg, ok := model.lastOptions.Messages[2].(openaicompatible.ToolMessage)
	if !ok {
		t.Fatalf("message 2 = %#v, want ToolMessage", model.lastOptions.Messages[2])
	}
	result, ok := toolMsg.Content[0].(openaicompatible.ToolResultContent)
	if !ok || result.ToolCallID != "call" || result.Output.Type != "error-text" || result.Output.Value != "boom" {
		t.Fatalf("tool result = %#v", toolMsg.Content[0])
	}
}

func TestOpenAICompatibleResultAndUsage(t *testing.T) {
	model := &fakeOpenAICompatibleModel{result: &openaicompatible.GenerateResult{
		FinishReason: openaicompatible.FinishReason{Unified: "tool-calls"},
		Content: []openaicompatible.Content{
			openaicompatible.TextContent{Text: "hi"},
			openaicompatible.ToolCallContent{ToolCallID: "id", ToolName: "t", Input: json.RawMessage(`{}`)},
		},
		Usage: openaicompatible.Usage{InputTokens: openaicompatible.TokenCounts{Total: intPtr(5)}, OutputTokens: openaicompatible.OutputTokenCounts{Total: intPtr(9)}},
	}}
	result, err := NewOpenAICompatibleAdapter(model).Generate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if result.FinishReason.Unified != FinishReasonToolCalls {
		t.Fatalf("finish = %q", result.FinishReason)
	}
	if len(result.Content) != 2 {
		t.Fatalf("content = %#v", result.Content)
	}
	if _, ok := result.Content[1].(ToolUseContent); !ok {
		t.Fatalf("content 1 = %#v, want ToolUseContent", result.Content[1])
	}
	if result.Usage.InputTokens != 5 || result.Usage.OutputTokens != 9 {
		t.Fatalf("usage = %#v, want 5/9", result.Usage)
	}
}

func TestOpenAICompatibleToolSchemaConversion(t *testing.T) {
	model := &fakeOpenAICompatibleModel{result: &openaicompatible.GenerateResult{FinishReason: openaicompatible.FinishReason{Unified: "stop"}}}
	_, err := NewOpenAICompatibleAdapter(model).Generate(context.Background(), GenerateOptions{Tools: []Tool{{Name: "t", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`)}}})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	tool := model.lastOptions.Tools[0]
	if tool.Name != "t" || tool.Type != "function" {
		t.Fatalf("tool = %#v", tool)
	}
	schema, ok := tool.InputSchema.(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("schema = %#v", tool.InputSchema)
	}
}

func TestOpenAICompatibleUnknownRoleErrors(t *testing.T) {
	_, err := NewOpenAICompatibleAdapter(&fakeOpenAICompatibleModel{}).Generate(context.Background(), GenerateOptions{Messages: []Message{{Role: "system"}}})
	if err == nil {
		t.Fatal("Generate error = nil, want unknown role error")
	}
}

func TestOpenAICompatibleNewFieldsWiring(t *testing.T) {
	ts := time.Now()
	model := &fakeOpenAICompatibleModel{result: &openaicompatible.GenerateResult{
		FinishReason: openaicompatible.FinishReason{Unified: "stop"},
		Warnings:     []openaicompatible.Warning{{Type: "x", Message: "warn"}},
		Response:     openaicompatible.ResponseMetadata{ID: "resp_1", ModelID: "m1", Timestamp: &ts, Headers: http.Header{"X-Req": []string{"abc"}}},
	}}
	schema := json.RawMessage(`{"type":"object"}`)
	result, err := NewOpenAICompatibleAdapter(model).Generate(context.Background(), GenerateOptions{
		ToolChoice:      ToolChoice{Type: llm.ToolChoiceRequired},
		ResponseFormat:  &ResponseFormat{Type: llm.ResponseFormatJSONSchema, Name: "out", JSONSchema: schema},
		ProviderOptions: llm.ProviderOptions{"openaicompatible": {"k": "v"}},
		Headers:         map[string]string{"X-Test": "1"},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if model.lastOptions.ToolChoice == nil || model.lastOptions.ToolChoice.Type != "required" {
		t.Fatalf("tool choice = %#v", model.lastOptions.ToolChoice)
	}
	if model.lastOptions.ResponseFormat == nil || model.lastOptions.ResponseFormat.Type != "json" || model.lastOptions.ResponseFormat.Name != "out" {
		t.Fatalf("response format = %#v", model.lastOptions.ResponseFormat)
	}
	if model.lastOptions.ResponseFormat.Schema == nil {
		t.Fatalf("response format schema not forwarded")
	}
	if model.lastOptions.ProviderOptions["openaicompatible"]["k"] != "v" {
		t.Fatalf("provider options = %#v", model.lastOptions.ProviderOptions)
	}
	if model.lastOptions.Headers.Get("X-Test") != "1" {
		t.Fatalf("headers = %#v", model.lastOptions.Headers)
	}
	if result.Response.ID != "resp_1" || result.Response.ModelID != "m1" || result.Response.Headers["X-Req"] != "abc" {
		t.Fatalf("response metadata = %#v", result.Response)
	}
	if len(result.Warnings) == 0 || result.Warnings[len(result.Warnings)-1].Message != "warn" {
		t.Fatalf("warnings = %#v", result.Warnings)
	}
}

func TestNewOpenAICompatibleClientRequiresBaseURL(t *testing.T) {
	_, err := NewOpenAICompatibleClient(OpenAICompatibleSettings{Name: "x"}, "model")
	if err == nil {
		t.Fatal("expected error for missing BaseURL")
	}
}
