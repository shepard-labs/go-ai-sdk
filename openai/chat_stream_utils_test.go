package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

// helper: drain a channel into a slice
func drainParts(ch <-chan StreamPart) []StreamPart {
	var out []StreamPart
	for p := range ch {
		out = append(out, p)
	}
	return out
}

// helper: create a buffered parts channel and a chat model for unit tests
func newTestStreamParts() (chan StreamPart, *openaiChatLanguageModel) {
	ch := make(chan StreamPart, 64)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	return ch, m
}

// --- chatFinishReasonFromString ---

func TestChatFinishReasonFromStringStop(t *testing.T) {
	if got := chatFinishReasonFromString("stop"); got.Unified != "stop" {
		t.Errorf("stop: %q", got.Unified)
	}
}

func TestChatFinishReasonFromStringLength(t *testing.T) {
	if got := chatFinishReasonFromString("length"); got.Unified != "length" {
		t.Errorf("length: %q", got.Unified)
	}
}

func TestChatFinishReasonFromStringToolCalls(t *testing.T) {
	if got := chatFinishReasonFromString("tool_calls"); got.Unified != "tool-calls" {
		t.Errorf("tool_calls: %q", got.Unified)
	}
}

func TestChatFinishReasonFromStringFunctionCall(t *testing.T) {
	if got := chatFinishReasonFromString("function_call"); got.Unified != "tool-calls" {
		t.Errorf("function_call: %q", got.Unified)
	}
}

func TestChatFinishReasonFromStringContentFilter(t *testing.T) {
	if got := chatFinishReasonFromString("content_filter"); got.Unified != "content-filter" {
		t.Errorf("content_filter: %q", got.Unified)
	}
}

func TestChatFinishReasonFromStringUnknown(t *testing.T) {
	if got := chatFinishReasonFromString("some_other"); got.Unified != "other" {
		t.Errorf("unknown: %q", got.Unified)
	}
}

// --- resolveToolCallIndex ---

func TestResolveToolCallIndexWithPointer(t *testing.T) {
	state := newChatStreamState()
	idx := 5
	got := resolveToolCallIndex(state, &idx)
	if got != 5 {
		t.Errorf("expected 5, got %d", got)
	}
	if state.toolCallCount != 0 {
		t.Errorf("toolCallCount should not increment when index is provided: %d", state.toolCallCount)
	}
}

func TestResolveToolCallIndexNilAutoIncrement(t *testing.T) {
	state := newChatStreamState()
	if got := resolveToolCallIndex(state, nil); got != 0 {
		t.Errorf("first: %d", got)
	}
	if got := resolveToolCallIndex(state, nil); got != 1 {
		t.Errorf("second: %d", got)
	}
	if state.toolCallCount != 2 {
		t.Errorf("toolCallCount: %d", state.toolCallCount)
	}
}

// --- mergeProviderMetadata ---

func TestMergeProviderMetadataBothNil(t *testing.T) {
	if got := mergeProviderMetadata(nil, nil); got != nil {
		t.Errorf("both nil: %v", got)
	}
}

func TestMergeProviderMetadataFirstNil(t *testing.T) {
	b := ProviderMetadata{"openai": map[string]any{"x": 1}}
	got := mergeProviderMetadata(nil, b)
	if len(got) != 1 {
		t.Errorf("first nil: %v", got)
	}
}

func TestMergeProviderMetadataSecondNil(t *testing.T) {
	a := ProviderMetadata{"openai": map[string]any{"x": 1}}
	got := mergeProviderMetadata(a, nil)
	if len(got) != 1 {
		t.Errorf("second nil: %v", got)
	}
}

func TestMergeProviderMetadataBothSet(t *testing.T) {
	a := ProviderMetadata{"openai": map[string]any{"x": 1}}
	b := ProviderMetadata{"anthropic": map[string]any{"y": 2}, "openai": map[string]any{"z": 3}}
	got := mergeProviderMetadata(a, b)
	if got["openai"].(map[string]any)["z"] != 3 {
		t.Errorf("b overrides a: %v", got)
	}
	if got["anthropic"] == nil {
		t.Errorf("anthropic missing: %v", got)
	}
}

// --- cloneRawMessage ---

func TestCloneRawMessageNil(t *testing.T) {
	if got := cloneRawMessage(nil); got != nil {
		t.Errorf("nil: %v", got)
	}
}

func TestCloneRawMessageNonNil(t *testing.T) {
	orig := json.RawMessage(`{"a":1}`)
	got := cloneRawMessage(orig)
	if string(got) != `{"a":1}` {
		t.Errorf("clone: %s", got)
	}
	orig[1] = 'X'
	if got[1] == 'X' {
		t.Errorf("clone shares underlying array")
	}
}

// --- buildChatProviderMetadataFromUsage ---

func TestBuildChatProviderMetadataFromUsageEmpty(t *testing.T) {
	if got := buildChatProviderMetadataFromUsage(nil); got != nil {
		t.Errorf("nil: %v", got)
	}
}

func TestBuildChatProviderMetadataFromUsageNoDets(t *testing.T) {
	raw := json.RawMessage(`{"prompt_tokens":1,"completion_tokens":1}`)
	if got := buildChatProviderMetadataFromUsage(raw); got != nil {
		t.Errorf("no details: %v", got)
	}
}

func TestBuildChatProviderMetadataFromUsageWithDetails(t *testing.T) {
	raw := json.RawMessage(`{"completion_tokens_details":{"accepted_prediction_tokens":3,"rejected_prediction_tokens":2}}`)
	got := buildChatProviderMetadataFromUsage(raw)
	if got == nil {
		t.Fatal("expected metadata, got nil")
	}
	om := got["openai"].(map[string]any)
	if om["acceptedPredictionTokens"] != 3 {
		t.Errorf("accepted: %v", om["acceptedPredictionTokens"])
	}
	if om["rejectedPredictionTokens"] != 2 {
		t.Errorf("rejected: %v", om["rejectedPredictionTokens"])
	}
}

func TestBuildChatProviderMetadataFromUsagePartialDetails(t *testing.T) {
	// Only accepted — rejected should be absent, not zero
	raw := json.RawMessage(`{"completion_tokens_details":{"accepted_prediction_tokens":5}}`)
	got := buildChatProviderMetadataFromUsage(raw)
	if got == nil {
		t.Fatal("expected metadata, got nil")
	}
	om := got["openai"].(map[string]any)
	if om["acceptedPredictionTokens"] != 5 {
		t.Errorf("accepted: %v", om["acceptedPredictionTokens"])
	}
	if _, has := om["rejectedPredictionTokens"]; has {
		t.Errorf("should not have rejectedPredictionTokens")
	}
}

// --- processChatStreamDelta ---

// TestProcessChatStreamDeltaTextOnly verifies basic text-only delta.
func TestProcessChatStreamDeltaTextOnly(t *testing.T) {
	ch, m := newTestStreamParts()
	state := newChatStreamState()
	m.processChatStreamDelta(ch, state, chatStreamDelta{Content: "hello"})
	close(ch)
	parts := drainParts(ch)

	types := make([]string, len(parts))
	for i, p := range parts {
		switch p.(type) {
		case StreamTextStart:
			types[i] = "start"
		case StreamTextDelta:
			types[i] = "delta"
		default:
			types[i] = "other"
		}
	}
	if len(parts) != 2 || types[0] != "start" || types[1] != "delta" {
		t.Errorf("unexpected parts: %v", types)
	}
}

// TestProcessChatStreamDeltaReasoningThenContent verifies reasoning
// start, delta, then end + text start on content.
func TestProcessChatStreamDeltaReasoningThenContent(t *testing.T) {
	ch, m := newTestStreamParts()
	state := newChatStreamState()
	m.processChatStreamDelta(ch, state, chatStreamDelta{ReasoningContent: "think"})
	m.processChatStreamDelta(ch, state, chatStreamDelta{Content: "answer"})
	close(ch)
	parts := drainParts(ch)

	var seen []string
	for _, p := range parts {
		switch p.(type) {
		case StreamReasoningStart:
			seen = append(seen, "rstart")
		case StreamReasoningDelta:
			seen = append(seen, "rdelta")
		case StreamReasoningEnd:
			seen = append(seen, "rend")
		case StreamTextStart:
			seen = append(seen, "tstart")
		case StreamTextDelta:
			seen = append(seen, "tdelta")
		}
	}
	expected := []string{"rstart", "rdelta", "rend", "tstart", "tdelta"}
	if strings.Join(seen, ",") != strings.Join(expected, ",") {
		t.Errorf("sequence: %v, want %v", seen, expected)
	}
}

// TestProcessChatStreamDeltaToolCallComplete verifies a complete tool
// call in one delta emits: StreamToolInputStart, StreamToolInputDelta,
// StreamToolInputEnd, StreamToolCall.
func TestProcessChatStreamDeltaToolCallComplete(t *testing.T) {
	ch, m := newTestStreamParts()
	state := newChatStreamState()
	id := "c1"
	name := "search"
	args := `{"q":"x"}`
	m.processChatStreamDelta(ch, state, chatStreamDelta{
		ToolCalls: []chatStreamToolCall{{
			ID:    &id,
			Index: nil,
			Function: chatStreamFunction{
				Name:      &name,
				Arguments: &args,
			},
		}},
	})
	close(ch)
	parts := drainParts(ch)

	var types []string
	for _, p := range parts {
		switch p.(type) {
		case StreamToolInputStart:
			types = append(types, "start")
		case StreamToolInputDelta:
			types = append(types, "delta")
		case StreamToolInputEnd:
			types = append(types, "end")
		case StreamToolCall:
			types = append(types, "call")
		}
	}
	if len(types) < 3 {
		t.Errorf("expected at least start/end/call, got %v", types)
	}
}

// TestProcessChatStreamDeltaAnnotation verifies url_citation annotations
// produce a SourceContent part.
func TestProcessChatStreamDeltaAnnotation(t *testing.T) {
	ch, m := newTestStreamParts()
	state := newChatStreamState()
	m.processChatStreamDelta(ch, state, chatStreamDelta{
		Annotations: []map[string]any{
			{
				"type":        "url_citation",
				"url_citation": map[string]any{"uuid": "u1", "url": "https://a.com", "title": "A"},
			},
		},
	})
	close(ch)
	parts := drainParts(ch)
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	sc, ok := parts[0].(SourceContent)
	if !ok {
		t.Fatalf("expected SourceContent, got %T", parts[0])
	}
	if sc.URL != "https://a.com" {
		t.Errorf("url: %q", sc.URL)
	}
}

// TestProcessChatStreamDeltaAnnotationNonURLCitation verifies that
// non-url_citation annotations are silently skipped.
func TestProcessChatStreamDeltaAnnotationNonURLCitation(t *testing.T) {
	ch, m := newTestStreamParts()
	state := newChatStreamState()
	m.processChatStreamDelta(ch, state, chatStreamDelta{
		Annotations: []map[string]any{
			{"type": "footnote"},
		},
	})
	close(ch)
	parts := drainParts(ch)
	if len(parts) != 0 {
		t.Errorf("unexpected parts for non-url_citation: %v", parts)
	}
}

// TestFlushChatStreamStateEmpty verifies no panic when state is empty.
func TestFlushChatStreamStateEmpty(t *testing.T) {
	ch := make(chan StreamPart, 8)
	state := newChatStreamState()
	state.flushChatStreamState(ch)
	close(ch)
	if len(drainParts(ch)) != 0 {
		t.Errorf("expected no parts from empty state")
	}
}

// TestFlushChatStreamStateReasoningActive verifies that active reasoning
// gets a StreamReasoningEnd when flushed.
func TestFlushChatStreamStateReasoningActive(t *testing.T) {
	ch := make(chan StreamPart, 8)
	state := newChatStreamState()
	state.reasoningActive = true
	state.flushChatStreamState(ch)
	close(ch)
	parts := drainParts(ch)
	if len(parts) != 1 {
		t.Fatalf("expected 1 part (StreamReasoningEnd), got %d", len(parts))
	}
	if _, ok := parts[0].(StreamReasoningEnd); !ok {
		t.Errorf("expected StreamReasoningEnd, got %T", parts[0])
	}
}

// TestFlushChatStreamStateTextStarted verifies that started text
// gets a StreamTextEnd when flushed.
func TestFlushChatStreamStateTextStarted(t *testing.T) {
	ch := make(chan StreamPart, 8)
	state := newChatStreamState()
	state.textStarted = true
	state.flushChatStreamState(ch)
	close(ch)
	parts := drainParts(ch)
	if len(parts) != 1 {
		t.Fatalf("expected 1 part (StreamTextEnd), got %d", len(parts))
	}
	if _, ok := parts[0].(StreamTextEnd); !ok {
		t.Errorf("expected StreamTextEnd, got %T", parts[0])
	}
}

// TestFlushChatStreamStateUnfinishedTool verifies that an unfinished
// tool call is closed out by flushChatStreamState.
func TestFlushChatStreamStateUnfinishedTool(t *testing.T) {
	ch := make(chan StreamPart, 16)
	state := newChatStreamState()
	state.toolCalls[0] = &chatStreamToolState{ID: "c1", Name: "f", Started: true, Finished: false}
	state.flushChatStreamState(ch)
	close(ch)
	parts := drainParts(ch)
	var types []string
	for _, p := range parts {
		switch p.(type) {
		case StreamToolInputEnd:
			types = append(types, "end")
		case StreamToolCall:
			types = append(types, "call")
		}
	}
	if len(types) < 2 || types[0] != "end" || types[1] != "call" {
		t.Errorf("unfinished tool flush parts: %v", types)
	}
}

// TestFlushChatStreamStateUnfinishedToolInvalidJSON verifies that
// invalid buffered args fall back to "{}".
func TestFlushChatStreamStateUnfinishedToolInvalidJSON(t *testing.T) {
	ch := make(chan StreamPart, 16)
	state := newChatStreamState()
	tool := &chatStreamToolState{ID: "c1", Name: "f", Started: true, Finished: false}
	tool.Args.WriteString("not-json")
	state.toolCalls[0] = tool
	state.flushChatStreamState(ch)
	close(ch)
	parts := drainParts(ch)
	var call *StreamToolCall
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			call = &tc
		}
	}
	if call == nil {
		t.Fatal("expected StreamToolCall")
	}
	if !json.Valid(call.Input) {
		t.Errorf("invalid JSON emitted: %s", call.Input)
	}
	if string(call.Input) != "{}" {
		t.Errorf("expected {}, got %s", call.Input)
	}
}
