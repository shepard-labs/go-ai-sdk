package adapters

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"regexp"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/google"
	. "github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/openai"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

// streamableFakeAnthropicModel emits a preconfigured sequence of stream parts.
type streamableFakeAnthropicModel struct {
	parts []anthropic.StreamPart
}

func (f *streamableFakeAnthropicModel) ModelID() string                          { return "fake" }
func (f *streamableFakeAnthropicModel) Provider() string                         { return "fake" }
func (f *streamableFakeAnthropicModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *streamableFakeAnthropicModel) DoGenerate(ctx context.Context, opts anthropic.GenerateOptions) (*anthropic.GenerateResult, error) {
	return nil, nil
}
func (f *streamableFakeAnthropicModel) DoStream(ctx context.Context, opts anthropic.StreamOptions) (*anthropic.StreamResult, error) {
	ch := make(chan anthropic.StreamPart, len(f.parts))
	for _, p := range f.parts {
		ch <- p
	}
	close(ch)
	return &anthropic.StreamResult{Stream: ch, Parts: ch}, nil
}

func TestAnthropicAdapterStreamMapping(t *testing.T) {
	model := &streamableFakeAnthropicModel{parts: []anthropic.StreamPart{
		anthropic.StreamTextStart{},
		anthropic.StreamTextDelta{Text: "Hel"},
		anthropic.StreamTextDelta{Text: "lo"},
		anthropic.StreamTextEnd{},
		anthropic.StreamReasoningStart{ID: "r1"},
		anthropic.StreamReasoningDelta{Delta: anthropic.ThinkingDelta{Thinking: "think"}},
		anthropic.StreamReasoningEnd{ID: "r1"},
		anthropic.StreamToolInputStart{ID: "t1", ToolName: "search"},
		anthropic.StreamToolInputDelta{ID: "t1", Delta: anthropic.InputJSONDelta{PartialJSON: `{"q":"`}},
		anthropic.StreamToolInputDelta{ID: "t1", Delta: anthropic.InputJSONDelta{PartialJSON: `go"}`}},
		anthropic.StreamToolInputEnd{ID: "t1"},
		anthropic.StreamFinish{FinishReason: anthropic.FinishReasonStop, Usage: anthropic.Usage{InputTokens: anthropic.TokenUsage{Total: 7}, OutputTokens: anthropic.TokenUsage{Total: 11}}},
	}}
	ch, err := NewAnthropicAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	want := []StreamPart{
		StreamTextStart{},
		StreamTextDelta{Text: "Hel"},
		StreamTextDelta{Text: "lo"},
		StreamTextEnd{},
		StreamReasoningStart{},
		StreamReasoningDelta{Text: "think"},
		StreamReasoningEnd{},
		StreamToolCallStart{ID: "t1", Name: "search"},
		StreamToolInputDelta{ID: "t1", JSON: `{"q":"`},
		StreamToolInputDelta{ID: "t1", JSON: `go"}`},
		StreamToolInputEnd{ID: "t1"},
		StreamFinish{FinishReason: FinishReasonStop, Usage: Usage{InputTokens: 7, OutputTokens: 11}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream parts =\n%#v\nwant\n%#v", got, want)
	}
}

func TestAnthropicAdapterStreamErrorOrdering(t *testing.T) {
	model := &streamableFakeAnthropicModel{parts: []anthropic.StreamPart{
		anthropic.StreamTextDelta{Text: "partial"},
		anthropic.StreamError{Err: errors.New("boom")},
	}}
	ch, err := NewAnthropicAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	if len(got) != 2 {
		t.Fatalf("parts = %d, want 2", len(got))
	}
	if _, ok := got[0].(StreamTextDelta); !ok {
		t.Fatalf("part 0 = %#v, want StreamTextDelta", got[0])
	}
	se, ok := got[1].(StreamError)
	if !ok || se.Err.Error() != "boom" {
		t.Fatalf("part 1 = %#v, want StreamError{boom}", got[1])
	}
}

// streamableFakeOpenAIModel emits a preconfigured sequence of stream parts.
type streamableFakeOpenAIModel struct {
	parts []openai.StreamPart
}

func (f *streamableFakeOpenAIModel) ModelID() string                          { return "fake" }
func (f *streamableFakeOpenAIModel) Provider() string                         { return "fake" }
func (f *streamableFakeOpenAIModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *streamableFakeOpenAIModel) DoGenerate(ctx context.Context, opts openai.GenerateOptions) (*openai.GenerateResult, error) {
	return nil, nil
}
func (f *streamableFakeOpenAIModel) DoStream(ctx context.Context, opts openai.StreamOptions) (*openai.StreamResult, error) {
	ch := make(chan openai.StreamPart, len(f.parts))
	for _, p := range f.parts {
		ch <- p
	}
	close(ch)
	return &openai.StreamResult{Stream: ch, Parts: ch}, nil
}

func TestOpenAIAdapterStreamMapping(t *testing.T) {
	model := &streamableFakeOpenAIModel{parts: []openai.StreamPart{
		openai.StreamTextStart{},
		openai.StreamTextDelta{Text: "hi"},
		openai.StreamTextEnd{},
		openai.StreamReasoningStart{},
		openai.StreamReasoningDelta{Text: "thinking"},
		openai.StreamReasoningEnd{},
		openai.StreamToolInputStart{ID: "c1", ToolName: "run"},
		openai.StreamToolInputDelta{ID: "c1", Delta: `{"x":1}`},
		openai.StreamToolInputEnd{ID: "c1"},
		openai.StreamFinish{FinishReason: openai.FinishReason{Unified: "tool-calls"}, Usage: openai.Usage{InputTokens: openai.TokenCounts{Total: intPtr(3)}, OutputTokens: openai.OutputTokenCounts{Total: intPtr(4)}}},
	}}
	ch, err := NewOpenAIAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	want := []StreamPart{
		StreamTextStart{},
		StreamTextDelta{Text: "hi"},
		StreamTextEnd{},
		StreamReasoningStart{},
		StreamReasoningDelta{Text: "thinking"},
		StreamReasoningEnd{},
		StreamToolCallStart{ID: "c1", Name: "run"},
		StreamToolInputDelta{ID: "c1", JSON: `{"x":1}`},
		StreamToolInputEnd{ID: "c1"},
		StreamFinish{FinishReason: FinishReasonToolCalls, Usage: Usage{InputTokens: 3, OutputTokens: 4}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream parts =\n%#v\nwant\n%#v", got, want)
	}
}

func TestOpenAIAdapterStreamErrorConverted(t *testing.T) {
	model := &streamableFakeOpenAIModel{parts: []openai.StreamPart{
		openai.StreamError{Err: io.ErrUnexpectedEOF},
	}}
	ch, err := NewOpenAIAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	if len(got) != 1 {
		t.Fatalf("parts = %d, want 1", len(got))
	}
	if _, ok := got[0].(StreamError); !ok {
		t.Fatalf("part 0 = %#v, want StreamError", got[0])
	}
}

// streamableFakeGoogleModel emits a preconfigured sequence of stream parts.
type streamableFakeGoogleModel struct {
	parts []google.StreamPart
}

func (f *streamableFakeGoogleModel) ModelID() string                          { return "fake" }
func (f *streamableFakeGoogleModel) Provider() string                         { return "fake" }
func (f *streamableFakeGoogleModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *streamableFakeGoogleModel) DoGenerate(ctx context.Context, opts google.GenerateOptions) (*google.GenerateResult, error) {
	return nil, nil
}
func (f *streamableFakeGoogleModel) DoStream(ctx context.Context, opts google.StreamOptions) (*google.StreamResult, error) {
	ch := make(chan google.StreamPart, len(f.parts))
	for _, p := range f.parts {
		ch <- p
	}
	close(ch)
	return &google.StreamResult{Stream: ch, Parts: ch}, nil
}

func TestGoogleAdapterStreamMapping(t *testing.T) {
	model := &streamableFakeGoogleModel{parts: []google.StreamPart{
		google.StreamTextStart{},
		google.StreamTextDelta{Text: "hi"},
		google.StreamTextEnd{},
		google.StreamReasoningStart{},
		google.StreamReasoningDelta{Text: "th"},
		google.StreamReasoningEnd{},
		google.StreamToolInputStart{ID: "g1", ToolName: "ls"},
		google.StreamToolInputDelta{ID: "g1", Delta: `{"path":"/"`},
		google.StreamToolInputEnd{ID: "g1"},
		google.StreamFinish{FinishReason: google.FinishReason{Unified: "stop"}, Usage: google.Usage{InputTokens: google.TokenCounts{Total: intPtr(8)}, OutputTokens: google.OutputTokenCounts{Total: intPtr(9)}}},
	}}
	ch, err := NewGoogleAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	want := []StreamPart{
		StreamTextStart{},
		StreamTextDelta{Text: "hi"},
		StreamTextEnd{},
		StreamReasoningStart{},
		StreamReasoningDelta{Text: "th"},
		StreamReasoningEnd{},
		StreamToolCallStart{ID: "g1", Name: "ls"},
		StreamToolInputDelta{ID: "g1", JSON: `{"path":"/"`},
		StreamToolInputEnd{ID: "g1"},
		StreamFinish{FinishReason: FinishReasonStop, Usage: Usage{InputTokens: 8, OutputTokens: 9}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream parts =\n%#v\nwant\n%#v", got, want)
	}
}

func TestStreamReturnsErrStreamNotImplementedForCohere(t *testing.T) {
	_, err := NewCohereAdapter(&fakeCohereModel{}).Stream(context.Background(), GenerateOptions{})
	if !errors.Is(err, ErrStreamNotImplemented) {
		t.Fatalf("err = %v, want ErrStreamNotImplemented", err)
	}
}

func TestOpenRouterAdapterStreamMapping(t *testing.T) {
	model := &streamableFakeOpenRouterModel{parts: []openrouter.StreamPart{
		openrouter.StreamTextStart{ID: "t"},
		openrouter.StreamTextDelta{ID: "t", Delta: "hi"},
		openrouter.StreamTextEnd{ID: "t"},
		openrouter.StreamReasoningStart{ID: "r"},
		openrouter.StreamReasoningDelta{ID: "r", Delta: "thinking"},
		openrouter.StreamReasoningEnd{ID: "r"},
		openrouter.StreamToolInputStart{ID: "c", ToolName: "run"},
		openrouter.StreamToolInputDelta{ID: "c", Delta: `{"x":1}`},
		openrouter.StreamToolInputEnd{ID: "c"},
		openrouter.StreamFinish{FinishReason: openrouter.FinishReasonToolCalls, Usage: openrouter.Usage{InputTokens: 3, OutputTokens: 4}},
	}}
	ch, err := NewOpenRouterAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	want := []StreamPart{
		StreamTextStart{},
		StreamTextDelta{Text: "hi"},
		StreamTextEnd{},
		StreamReasoningStart{},
		StreamReasoningDelta{Text: "thinking"},
		StreamReasoningEnd{},
		StreamToolCallStart{ID: "c", Name: "run"},
		StreamToolInputDelta{ID: "c", JSON: `{"x":1}`},
		StreamToolInputEnd{ID: "c"},
		StreamFinish{FinishReason: FinishReasonToolCalls, Usage: Usage{InputTokens: 3, OutputTokens: 4}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream parts =\n%#v\nwant\n%#v", got, want)
	}
}

func TestOpenAICompatibleAdapterStreamMapping(t *testing.T) {
	model := &streamableFakeOpenAICompatibleModel{parts: []openaicompatible.StreamPart{
		openaicompatible.StreamTextStart{ID: "t"},
		openaicompatible.StreamTextDelta{ID: "t", Text: "hi"},
		openaicompatible.StreamTextEnd{ID: "t"},
		openaicompatible.StreamReasoningStart{ID: "r"},
		openaicompatible.StreamReasoningDelta{ID: "r", Text: "thinking"},
		openaicompatible.StreamReasoningEnd{ID: "r"},
		openaicompatible.StreamToolInputStart{ID: "c", ToolName: "run"},
		openaicompatible.StreamToolInputDelta{ID: "c", Delta: `{"x":1}`},
		openaicompatible.StreamToolInputEnd{ID: "c"},
		openaicompatible.StreamFinish{FinishReason: openaicompatible.FinishReason{Unified: "stop"}, Usage: openaicompatible.Usage{InputTokens: openaicompatible.TokenCounts{Total: intPtr(8)}, OutputTokens: openaicompatible.OutputTokenCounts{Total: intPtr(9)}}},
	}}
	ch, err := NewOpenAICompatibleAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	want := []StreamPart{
		StreamTextStart{},
		StreamTextDelta{Text: "hi"},
		StreamTextEnd{},
		StreamReasoningStart{},
		StreamReasoningDelta{Text: "thinking"},
		StreamReasoningEnd{},
		StreamToolCallStart{ID: "c", Name: "run"},
		StreamToolInputDelta{ID: "c", JSON: `{"x":1}`},
		StreamToolInputEnd{ID: "c"},
		StreamFinish{FinishReason: FinishReasonStop, Usage: Usage{InputTokens: 8, OutputTokens: 9}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream parts =\n%#v\nwant\n%#v", got, want)
	}
}

func TestStreamContextCancelEmitsStreamError(t *testing.T) {
	// A stream that never emits; cancelling the Stream ctx should cause the
	// adapter goroutine to emit a StreamError{ctx.Err()} and close.
	parts := make(chan anthropic.StreamPart)
	defer close(parts)
	model := &blockingAnthropicModel{parts: parts}
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := NewAnthropicAdapter(model).Stream(ctx, GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	cancel()
	got := drainParts(ch)
	if len(got) != 1 {
		t.Fatalf("parts = %d, want 1 (StreamError)", len(got))
	}
	se, ok := got[0].(StreamError)
	if !ok {
		t.Fatalf("part 0 = %#v, want StreamError", got[0])
	}
	if !errors.Is(se.Err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", se.Err)
	}
}

type blockingAnthropicModel struct {
	parts chan anthropic.StreamPart
}

func (f *blockingAnthropicModel) ModelID() string                          { return "fake" }
func (f *blockingAnthropicModel) Provider() string                         { return "fake" }
func (f *blockingAnthropicModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *blockingAnthropicModel) DoGenerate(ctx context.Context, opts anthropic.GenerateOptions) (*anthropic.GenerateResult, error) {
	return nil, nil
}
func (f *blockingAnthropicModel) DoStream(ctx context.Context, opts anthropic.StreamOptions) (*anthropic.StreamResult, error) {
	return &anthropic.StreamResult{Stream: f.parts, Parts: f.parts}, nil
}

func drainParts(ch <-chan StreamPart) []StreamPart {
	var got []StreamPart
	for p := range ch {
		got = append(got, p)
	}
	return got
}

// Ensure JSON envelope of a tool input delta is preserved verbatim.
func TestStreamToolInputDeltaJSONPreserved(t *testing.T) {
	model := &streamableFakeOpenAIModel{parts: []openai.StreamPart{
		openai.StreamToolInputDelta{ID: "x", Delta: `{"a":"b\"c"}`},
		openai.StreamFinish{FinishReason: openai.FinishReason{Unified: "stop"}},
	}}
	ch, err := NewOpenAIAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	parts := drainParts(ch)
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2", len(parts))
	}
	d, ok := parts[0].(StreamToolInputDelta)
	if !ok || d.JSON != `{"a":"b\"c"}` {
		t.Fatalf("delta = %#v, want JSON preserved", parts[0])
	}
}

// Smoke: a tool_use content round-trips through the assistant mapping with
// reasoning content present, ensuring the new ReasoningContent tag is handled
// by the OpenAI-compatible translator (regression guard for §1.2).
func TestReasoningContentRoundTripsThroughOpenAICompatible(t *testing.T) {
	model := &fakeOpenAICompatibleModel{result: &openaicompatible.GenerateResult{
		FinishReason: openaicompatible.FinishReason{Unified: "stop"},
		Content: []openaicompatible.Content{
			openaicompatible.TextContent{Text: "answer"},
			openaicompatible.ReasoningContent{Text: "because"},
		},
		Usage: openaicompatible.Usage{InputTokens: openaicompatible.TokenCounts{Total: intPtr(1)}, OutputTokens: openaicompatible.OutputTokenCounts{Total: intPtr(2)}},
	}}
	result, err := NewOpenAICompatibleAdapter(model).Generate(context.Background(), GenerateOptions{Messages: []Message{
		{Role: "assistant", Content: []Content{ReasoningContent{Text: "prior thought"}}},
	}})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if len(result.Content) != 2 {
		t.Fatalf("content len = %d, want 2", len(result.Content))
	}
	if rc, ok := result.Content[1].(ReasoningContent); !ok || rc.Text != "because" {
		t.Fatalf("content 1 = %#v, want ReasoningContent{because}", result.Content[1])
	}
	// Verify the input assistant reasoning was translated to the provider form.
	assistant, ok := model.lastOptions.Messages[0].(openaicompatible.AssistantMessage)
	if !ok {
		t.Fatalf("message 0 = %#v, want AssistantMessage", model.lastOptions.Messages[0])
	}
	if rc, ok := assistant.Content[0].(openaicompatible.ReasoningContent); !ok || rc.Text != "prior thought" {
		t.Fatalf("assistant content 0 = %#v, want ReasoningContent", assistant.Content[0])
	}
}

// Ensure JSON of a parsed tool input is valid (regression guard).
func TestStreamToolInputEndInputIsValidJSON(t *testing.T) {
	model := &streamableFakeGoogleModel{parts: []google.StreamPart{
		google.StreamToolCall{ToolCall: google.ToolCallContent{ToolCallID: "g", ToolName: "t", Input: json.RawMessage(`{"k":"v"}`)}},
		google.StreamFinish{FinishReason: google.FinishReason{Unified: "stop"}},
	}}
	ch, err := NewGoogleAdapter(model).Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	parts := drainParts(ch)
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2", len(parts))
	}
	end, ok := parts[0].(StreamToolInputEnd)
	if !ok || string(end.Input) != `{"k":"v"}` {
		t.Fatalf("end = %#v, want Input preserved", parts[0])
	}
}

// streamableFakeOpenRouterModel emits a preconfigured sequence of stream parts.
type streamableFakeOpenRouterModel struct {
	parts []openrouter.StreamPart
}

func (f *streamableFakeOpenRouterModel) ModelID() string                          { return "fake" }
func (f *streamableFakeOpenRouterModel) Provider() string                         { return "fake" }
func (f *streamableFakeOpenRouterModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *streamableFakeOpenRouterModel) DoGenerate(ctx context.Context, opts openrouter.GenerateOptions) (*openrouter.GenerateResult, error) {
	return nil, nil
}
func (f *streamableFakeOpenRouterModel) DoStream(ctx context.Context, opts openrouter.StreamOptions) (*openrouter.StreamResult, error) {
	ch := make(chan openrouter.StreamPart, len(f.parts))
	for _, p := range f.parts {
		ch <- p
	}
	close(ch)
	return &openrouter.StreamResult{Stream: ch, Parts: ch}, nil
}

// streamableFakeOpenAICompatibleModel emits a preconfigured sequence of stream parts.
type streamableFakeOpenAICompatibleModel struct {
	parts []openaicompatible.StreamPart
}

func (f *streamableFakeOpenAICompatibleModel) ModelID() string                          { return "fake" }
func (f *streamableFakeOpenAICompatibleModel) Provider() string                         { return "fake" }
func (f *streamableFakeOpenAICompatibleModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *streamableFakeOpenAICompatibleModel) DoGenerate(ctx context.Context, opts openaicompatible.GenerateOptions) (*openaicompatible.GenerateResult, error) {
	return nil, nil
}
func (f *streamableFakeOpenAICompatibleModel) DoStream(ctx context.Context, opts openaicompatible.StreamOptions) (*openaicompatible.StreamResult, error) {
	ch := make(chan openaicompatible.StreamPart, len(f.parts))
	for _, p := range f.parts {
		ch <- p
	}
	close(ch)
	return &openaicompatible.StreamResult{Stream: ch, Parts: ch}, nil
}
