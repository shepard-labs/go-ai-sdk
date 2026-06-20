package adapters

import (
	"context"
	"encoding/json"
	"regexp"
	"testing"

	. "github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

type fakeOpenAICompatibleModel struct {
	lastOptions openaicompatible.GenerateOptions
	result      *openaicompatible.GenerateResult
	err         error
}

func (f *fakeOpenAICompatibleModel) ModelID() string                          { return "fake" }
func (f *fakeOpenAICompatibleModel) Provider() string                         { return "fake" }
func (f *fakeOpenAICompatibleModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeOpenAICompatibleModel) DoGenerate(ctx context.Context, opts openaicompatible.GenerateOptions) (*openaicompatible.GenerateResult, error) {
	f.lastOptions = opts
	return f.result, f.err
}
func (f *fakeOpenAICompatibleModel) DoStream(ctx context.Context, opts openaicompatible.StreamOptions) (*openaicompatible.StreamResult, error) {
	return nil, nil
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
	if result.FinishReason != FinishReasonToolCalls {
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

func TestNewOpenAICompatibleClientRequiresBaseURL(t *testing.T) {
	_, err := NewOpenAICompatibleClient(OpenAICompatibleSettings{Name: "x"}, "model")
	if err == nil {
		t.Fatal("expected error for missing BaseURL")
	}
}
