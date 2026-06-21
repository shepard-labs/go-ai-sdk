package adapters

import (
	"context"
	"errors"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/google"
	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/openai"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

// --- CollectStream unit tests (manual channels) ---

func TestCollectStreamText(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamTextStart{}
	ch <- llm.StreamTextDelta{Text: "hello"}
	ch <- llm.StreamTextDelta{Text: " world"}
	ch <- llm.StreamTextEnd{}
	ch <- llm.StreamFinish{FinishReason: llm.FinishReason{Unified: llm.FinishReasonStop}}
	close(ch)

	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(result.Content))
	}
	tc, ok := result.Content[0].(llm.TextContent)
	if !ok || tc.Text != "hello world" {
		t.Fatalf("content 0 = %#v, want TextContent{hello world}", result.Content[0])
	}
	if result.FinishReason.Unified != llm.FinishReasonStop {
		t.Fatalf("finish reason = %v, want stop", result.FinishReason.Unified)
	}
}

func TestCollectStreamToolCall(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamToolCallStart{ID: "t1", Name: "search"}
	ch <- llm.StreamToolInputDelta{ID: "t1", JSON: `{"`}
	ch <- llm.StreamToolInputDelta{ID: "t1", JSON: `x":1}`}
	ch <- llm.StreamToolInputEnd{ID: "t1"}
	ch <- llm.StreamFinish{FinishReason: llm.FinishReason{Unified: llm.FinishReasonToolCalls}}
	close(ch)

	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(result.Content))
	}
	tu, ok := result.Content[0].(llm.ToolUseContent)
	if !ok {
		t.Fatalf("content 0 = %#v, want ToolUseContent", result.Content[0])
	}
	if tu.ID != "t1" || tu.Name != "search" || string(tu.Input) != `{"x":1}` {
		t.Fatalf("tool use = %#v, want id=t1 name=search input={\"x\":1}", tu)
	}
}

func TestCollectStreamMetadata(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamMetadata{Response: llm.ResponseMetadata{ID: "resp-1", ModelID: "model-x"}}
	ch <- llm.StreamTextStart{}
	ch <- llm.StreamTextDelta{Text: "ok"}
	ch <- llm.StreamTextEnd{}
	ch <- llm.StreamFinish{FinishReason: llm.FinishReason{Unified: llm.FinishReasonStop}}
	close(ch)

	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if result.Response.ID != "resp-1" || result.Response.ModelID != "model-x" {
		t.Fatalf("response metadata = %#v, want id=resp-1 model=model-x", result.Response)
	}
}

func TestCollectStreamWarning(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamWarning{Warning: llm.Warning{Code: "unsupported", Message: "x ignored", Provider: "fake"}}
	ch <- llm.StreamFinish{FinishReason: llm.FinishReason{Unified: llm.FinishReasonStop}}
	close(ch)

	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if len(result.Warnings) != 1 || result.Warnings[0].Code != "unsupported" {
		t.Fatalf("warnings = %#v, want one unsupported warning", result.Warnings)
	}
}

func TestCollectStreamError(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamTextStart{}
	ch <- llm.StreamTextDelta{Text: "partial"}
	ch <- llm.StreamError{Err: errors.New("boom")}
	close(ch)

	result, err := llm.CollectStream(context.Background(), ch)
	if err == nil || err.Error() != "boom" {
		t.Fatalf("err = %v, want boom", err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d, want 1 (partial)", len(result.Content))
	}
	tc, ok := result.Content[0].(llm.TextContent)
	if !ok || tc.Text != "partial" {
		t.Fatalf("content 0 = %#v, want TextContent{partial}", result.Content[0])
	}
}

func TestCollectStreamProviderMetadataInFinish(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamTextStart{}
	ch <- llm.StreamTextDelta{Text: "ok"}
	ch <- llm.StreamTextEnd{}
	ch <- llm.StreamFinish{
		FinishReason:     llm.FinishReason{Unified: llm.FinishReasonStop},
		ProviderMetadata: llm.ProviderMetadata{"fake": {"k": "v"}},
	}
	close(ch)

	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if result.ProviderMetadata["fake"]["k"] != "v" {
		t.Fatalf("provider metadata = %#v, want fake.k=v", result.ProviderMetadata)
	}
}

func TestCollectStreamClosedWithoutFinish(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamTextStart{}
	ch <- llm.StreamTextDelta{Text: "partial"}
	close(ch)

	result, err := llm.CollectStream(context.Background(), ch)
	if err == nil {
		t.Fatalf("err = nil, want stream-closed-without-finish error")
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d, want 1 (partial)", len(result.Content))
	}
}

func TestCollectStreamContextCancel(t *testing.T) {
	ch := make(chan llm.StreamPart, 10)
	ch <- llm.StreamTextStart{}
	ch <- llm.StreamTextDelta{Text: "partial"}
	close(ch)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := llm.CollectStream(ctx, ch)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if result == nil {
		t.Fatalf("result = nil, want partial result")
	}
}

// --- Adapter-level conformance: provider stream parts → adapter → CollectStream ---

func TestAnthropicAdapterCollectStream(t *testing.T) {
	model := &streamableFakeAnthropicModel{parts: []anthropic.StreamPart{
		anthropic.StreamResponseMetadata{ID: "msg-1", ModelID: "claude"},
		anthropic.StreamTextStart{},
		anthropic.StreamTextDelta{Text: "hello"},
		anthropic.StreamTextDelta{Text: " world"},
		anthropic.StreamTextEnd{},
		anthropic.StreamFinish{
			FinishReason:     anthropic.FinishReasonStop,
			ProviderMetadata: anthropic.ProviderMetadata{"k": "v"},
		},
	}}
	ch, err := NewAnthropicAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if got := collectedText(result); got != "hello world" {
		t.Fatalf("text = %q, want hello world", got)
	}
	if result.Response.ID != "msg-1" {
		t.Fatalf("response id = %q, want msg-1", result.Response.ID)
	}
	if result.ProviderMetadata["anthropic"]["k"] != "v" {
		t.Fatalf("provider metadata = %#v, want anthropic.k=v", result.ProviderMetadata)
	}
}

func TestAnthropicAdapterCollectStreamToolCall(t *testing.T) {
	model := &streamableFakeAnthropicModel{parts: []anthropic.StreamPart{
		anthropic.StreamToolInputStart{ID: "t1", ToolName: "search"},
		anthropic.StreamToolInputDelta{ID: "t1", Delta: anthropic.InputJSONDelta{PartialJSON: `{"`}},
		anthropic.StreamToolInputDelta{ID: "t1", Delta: anthropic.InputJSONDelta{PartialJSON: `x":1}`}},
		anthropic.StreamToolInputEnd{ID: "t1"},
		anthropic.StreamFinish{FinishReason: anthropic.FinishReasonToolCalls},
	}}
	ch, err := NewAnthropicAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	assertToolUse(t, result, "t1", "search", `{"x":1}`)
}

func TestOpenAIAdapterCollectStream(t *testing.T) {
	model := &streamableFakeOpenAIModel{parts: []openai.StreamPart{
		openai.StreamStart{Warnings: []openai.Warning{{Type: "unsupported", Message: "x ignored"}}},
		openai.StreamResponseMetadata{ID: "resp-1", ModelID: "gpt"},
		openai.StreamTextStart{},
		openai.StreamTextDelta{Text: "hello"},
		openai.StreamTextDelta{Text: " world"},
		openai.StreamTextEnd{},
		openai.StreamFinish{FinishReason: openai.FinishReason{Unified: "stop"}, ProviderMetadata: openai.ProviderMetadata{"k": "v"}},
	}}
	ch, err := NewOpenAIAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if got := collectedText(result); got != "hello world" {
		t.Fatalf("text = %q, want hello world", got)
	}
	if result.Response.ID != "resp-1" {
		t.Fatalf("response id = %q, want resp-1", result.Response.ID)
	}
	if len(result.Warnings) != 1 || result.Warnings[0].Message != "x ignored" {
		t.Fatalf("warnings = %#v, want one warning", result.Warnings)
	}
	if result.ProviderMetadata["openai"]["k"] != "v" {
		t.Fatalf("provider metadata = %#v, want openai.k=v", result.ProviderMetadata)
	}
}

func TestOpenAIAdapterCollectStreamError(t *testing.T) {
	model := &streamableFakeOpenAIModel{parts: []openai.StreamPart{
		openai.StreamTextStart{},
		openai.StreamTextDelta{Text: "partial"},
		openai.StreamError{Err: errors.New("boom")},
	}}
	ch, err := NewOpenAIAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	result, err := llm.CollectStream(context.Background(), ch)
	if err == nil || err.Error() != "boom" {
		t.Fatalf("err = %v, want boom", err)
	}
	if got := collectedText(result); got != "partial" {
		t.Fatalf("text = %q, want partial", got)
	}
}

func TestOpenAICompatibleAdapterCollectStream(t *testing.T) {
	model := &streamableFakeOpenAICompatibleModel{parts: []openaicompatible.StreamPart{
		openaicompatible.StreamStart{Warnings: []openaicompatible.Warning{{Type: "unsupported", Message: "x ignored"}}},
		openaicompatible.StreamResponseMetadata{ID: "resp-2", ModelID: "local"},
		openaicompatible.StreamTextStart{ID: "t"},
		openaicompatible.StreamTextDelta{ID: "t", Text: "hello"},
		openaicompatible.StreamTextEnd{ID: "t"},
		openaicompatible.StreamFinish{FinishReason: openaicompatible.FinishReason{Unified: "stop"}, ProviderMetadata: openaicompatible.ProviderMetadata{"k": "v"}},
	}}
	ch, err := NewOpenAICompatibleAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if got := collectedText(result); got != "hello" {
		t.Fatalf("text = %q, want hello", got)
	}
	if result.Response.ID != "resp-2" {
		t.Fatalf("response id = %q, want resp-2", result.Response.ID)
	}
	if len(result.Warnings) != 1 {
		t.Fatalf("warnings = %#v, want one warning", result.Warnings)
	}
	if result.ProviderMetadata["openaicompatible"]["k"] != "v" {
		t.Fatalf("provider metadata = %#v, want openaicompatible.k=v", result.ProviderMetadata)
	}
}

func TestGoogleAdapterCollectStream(t *testing.T) {
	model := &streamableFakeGoogleModel{parts: []google.StreamPart{
		google.StreamStart{Warnings: []google.Warning{{Type: "unsupported", Message: "x ignored"}}},
		google.StreamResponseMetadata{ID: "resp-3", ModelID: "gemini"},
		google.StreamTextStart{},
		google.StreamTextDelta{Text: "hello"},
		google.StreamTextEnd{},
		google.StreamFinish{FinishReason: google.FinishReason{Unified: "stop"}, ProviderMetadata: google.ProviderMetadata{"k": "v"}},
	}}
	ch, err := NewGoogleAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if got := collectedText(result); got != "hello" {
		t.Fatalf("text = %q, want hello", got)
	}
	if result.Response.ID != "resp-3" {
		t.Fatalf("response id = %q, want resp-3", result.Response.ID)
	}
	if len(result.Warnings) != 1 {
		t.Fatalf("warnings = %#v, want one warning", result.Warnings)
	}
	if result.ProviderMetadata["google"]["k"] != "v" {
		t.Fatalf("provider metadata = %#v, want google.k=v", result.ProviderMetadata)
	}
}

func TestOpenRouterAdapterCollectStream(t *testing.T) {
	model := &streamableFakeOpenRouterModel{parts: []openrouter.StreamPart{
		openrouter.StreamResponseMetadata{ID: "resp-4", ModelID: "router"},
		openrouter.StreamTextStart{ID: "t"},
		openrouter.StreamTextDelta{ID: "t", Delta: "hello"},
		openrouter.StreamTextEnd{ID: "t"},
		openrouter.StreamFinish{FinishReason: openrouter.FinishReasonStop, ProviderMetadata: openrouter.ProviderMetadata{"k": "v"}},
	}}
	ch, err := NewOpenRouterAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	result, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}
	if got := collectedText(result); got != "hello" {
		t.Fatalf("text = %q, want hello", got)
	}
	if result.Response.ID != "resp-4" {
		t.Fatalf("response id = %q, want resp-4", result.Response.ID)
	}
	if result.ProviderMetadata["openrouter"]["k"] != "v" {
		t.Fatalf("provider metadata = %#v, want openrouter.k=v", result.ProviderMetadata)
	}
}

// --- test helpers ---

func collectedText(result *GenerateResult) string {
	for _, c := range result.Content {
		if tc, ok := c.(TextContent); ok {
			return tc.Text
		}
	}
	return ""
}

func assertToolUse(t *testing.T, result *GenerateResult, id, name, input string) {
	t.Helper()
	for _, c := range result.Content {
		if tu, ok := c.(ToolUseContent); ok {
			if tu.ID != id || tu.Name != name || string(tu.Input) != input {
				t.Fatalf("tool use = %#v, want id=%s name=%s input=%s", tu, id, name, input)
			}
			return
		}
	}
	t.Fatalf("no ToolUseContent in result %#v", result.Content)
}
