package adapters

import (
	"context"
	"encoding/json"
	"regexp"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/cohere"
	"github.com/shepard-labs/go-ai-sdk/google"
	. "github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

// ---- Google ----

type fakeGoogleModel struct {
	lastOptions google.GenerateOptions
	result      *google.GenerateResult
}

func (f *fakeGoogleModel) ModelID() string                          { return "fake" }
func (f *fakeGoogleModel) Provider() string                         { return "fake" }
func (f *fakeGoogleModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeGoogleModel) DoGenerate(ctx context.Context, opts google.GenerateOptions) (*google.GenerateResult, error) {
	f.lastOptions = opts
	return f.result, nil
}
func (f *fakeGoogleModel) DoStream(ctx context.Context, opts google.StreamOptions) (*google.StreamResult, error) {
	return nil, nil
}

func TestGoogleAdapterTranslation(t *testing.T) {
	model := &fakeGoogleModel{result: &google.GenerateResult{
		FinishReason: google.FinishReason{Unified: "tool-calls"},
		Content: []google.Content{
			google.ToolCallContent{ToolCallID: "id", ToolName: "t", Input: json.RawMessage(`{}`)},
		},
		Usage: google.Usage{InputTokens: google.TokenCounts{Total: intPtr(2)}, OutputTokens: google.OutputTokenCounts{Total: intPtr(3)}},
	}}
	result, err := NewGoogleAdapter(model).Generate(context.Background(), GenerateOptions{
		System:    "sys",
		MaxTokens: 10,
		Messages: []Message{
			{Role: "user", Content: []Content{TextContent{Text: "hi"}, ToolResultContent{ToolUseID: "id", Text: "ok"}}},
		},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if _, ok := model.lastOptions.Messages[0].(google.SystemMessage); !ok {
		t.Fatalf("message 0 = %#v, want SystemMessage", model.lastOptions.Messages[0])
	}
	if _, ok := model.lastOptions.Messages[1].(google.UserMessage); !ok {
		t.Fatalf("message 1 = %#v, want UserMessage", model.lastOptions.Messages[1])
	}
	if _, ok := model.lastOptions.Messages[2].(google.ToolMessage); !ok {
		t.Fatalf("message 2 = %#v, want ToolMessage", model.lastOptions.Messages[2])
	}
	if model.lastOptions.MaxOutputTokens == nil || *model.lastOptions.MaxOutputTokens != 10 {
		t.Fatalf("max output tokens = %v", model.lastOptions.MaxOutputTokens)
	}
	if result.FinishReason != FinishReasonToolCalls || len(result.Content) != 1 {
		t.Fatalf("result = %#v", result)
	}
	if result.Usage.InputTokens != 2 || result.Usage.OutputTokens != 3 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}

// ---- Cohere ----

type fakeCohereModel struct {
	lastOptions cohere.GenerateOptions
	result      *cohere.GenerateResult
}

func (f *fakeCohereModel) ModelID() string                          { return "fake" }
func (f *fakeCohereModel) Provider() string                         { return "fake" }
func (f *fakeCohereModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeCohereModel) DoGenerate(ctx context.Context, opts cohere.GenerateOptions) (*cohere.GenerateResult, error) {
	f.lastOptions = opts
	return f.result, nil
}
func (f *fakeCohereModel) DoStream(ctx context.Context, opts cohere.StreamOptions) (*cohere.StreamResult, error) {
	return nil, nil
}

func TestCohereAdapterTranslation(t *testing.T) {
	model := &fakeCohereModel{result: &cohere.GenerateResult{
		FinishReason: cohere.FinishReason{Unified: "stop"},
		Content:      []cohere.Content{cohere.TextContent{Text: "hello"}},
		Usage:        cohere.Usage{InputTokens: cohere.TokenCounts{Total: intPtr(1)}, OutputTokens: cohere.OutputTokenCounts{Total: intPtr(8)}},
	}}
	result, err := NewCohereAdapter(model).Generate(context.Background(), GenerateOptions{
		Messages: []Message{{Role: "assistant", Content: []Content{ToolUseContent{ID: "c", Name: "t", Input: json.RawMessage(`{"a":1}`)}}}},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	assistant := model.lastOptions.Messages[0].(cohere.AssistantMessage)
	if call, ok := assistant.Content[0].(cohere.ToolCallContent); !ok || call.ToolName != "t" {
		t.Fatalf("assistant content = %#v", assistant.Content[0])
	}
	if result.FinishReason != FinishReasonStop {
		t.Fatalf("finish = %q", result.FinishReason)
	}
	if result.Usage.InputTokens != 1 || result.Usage.OutputTokens != 8 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}

// ---- OpenRouter ----

type fakeOpenRouterModel struct {
	lastOptions openrouter.GenerateOptions
	result      *openrouter.GenerateResult
}

func (f *fakeOpenRouterModel) ModelID() string                          { return "fake" }
func (f *fakeOpenRouterModel) Provider() string                         { return "fake" }
func (f *fakeOpenRouterModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeOpenRouterModel) DoGenerate(ctx context.Context, opts openrouter.GenerateOptions) (*openrouter.GenerateResult, error) {
	f.lastOptions = opts
	return f.result, nil
}
func (f *fakeOpenRouterModel) DoStream(ctx context.Context, opts openrouter.StreamOptions) (*openrouter.StreamResult, error) {
	return nil, nil
}

func TestOpenRouterAdapterTranslation(t *testing.T) {
	model := &fakeOpenRouterModel{result: &openrouter.GenerateResult{
		FinishReason: openrouter.FinishReasonToolCalls,
		Content: []openrouter.Content{
			openrouter.ToolCallContent{ToolCallID: "id", ToolName: "t", Input: map[string]any{"x": float64(1)}},
		},
		Usage: openrouter.Usage{InputTokens: 6, OutputTokens: 7},
	}}
	result, err := NewOpenRouterAdapter(model).Generate(context.Background(), GenerateOptions{
		MaxTokens: 20,
		Messages: []Message{
			{Role: "assistant", Content: []Content{ToolUseContent{ID: "id", Name: "t", Input: json.RawMessage(`{"x":1}`)}}},
			{Role: "user", Content: []Content{ToolResultContent{ToolUseID: "id", Text: "done"}}},
		},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if model.lastOptions.MaxTokens == nil || *model.lastOptions.MaxTokens != 20 {
		t.Fatalf("max tokens = %v", model.lastOptions.MaxTokens)
	}
	assistant := model.lastOptions.Messages[0].(openrouter.AssistantMessage)
	call := assistant.Content[0].(openrouter.ToolCallContent)
	m, ok := call.Input.(map[string]any)
	if !ok || m["x"] != float64(1) {
		t.Fatalf("tool call input = %#v, want decoded object", call.Input)
	}
	toolMsg := model.lastOptions.Messages[1].(openrouter.ToolMessage)
	res := toolMsg.Content[0].(openrouter.ToolResultContent)
	if res.ToolCallID != "id" || res.Output != "done" {
		t.Fatalf("tool result = %#v", res)
	}
	if result.FinishReason != FinishReasonToolCalls {
		t.Fatalf("finish = %q", result.FinishReason)
	}
	use := result.Content[0].(ToolUseContent)
	var decoded map[string]any
	if err := json.Unmarshal(use.Input, &decoded); err != nil || decoded["x"] != float64(1) {
		t.Fatalf("result tool input = %s", use.Input)
	}
	if result.Usage.InputTokens != 6 || result.Usage.OutputTokens != 7 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}
