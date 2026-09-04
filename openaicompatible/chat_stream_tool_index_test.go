package openaicompatible

import (
	"context"
	"net/http"
	"testing"
)

// TestChatStreamTruncatedArgsProduceNoToolCall verifies that a truncated
// '{"a":1' accumulation split across chunks must never execute as a tool
// call. Flush closes the input with a StreamError and an error finish
// instead.
func TestChatStreamTruncatedArgsProduceNoToolCall(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{"a":`)).
		add(toolCallChunks(0, "call-1", "weather", `1`)).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})

	parts := collectStream(result.Stream)
	for _, p := range parts {
		if _, ok := p.(StreamToolCall); ok {
			t.Fatal("truncated input must not produce StreamToolCall")
		}
	}
	var streamErr *StreamError
	var finish *StreamFinish
	for i := range parts {
		switch p := parts[i].(type) {
		case StreamError:
			streamErr = &p
		case StreamFinish:
			finish = &p
		}
	}
	if streamErr == nil {
		t.Fatal("no StreamError for truncated tool input")
	}
	if _, ok := streamErr.Err.(InvalidResponseDataError); !ok {
		t.Fatalf("error = %T (%v), want InvalidResponseDataError", streamErr.Err, streamErr.Err)
	}
	if finish == nil || finish.FinishReason.Unified != "error" {
		t.Fatalf("finish = %#v, want error", finish)
	}
}

// TestChatStreamGarbageSuffixProducesNoToolCall verifies that a parsable
// prefix '{"a":1}' followed by '<garbage>' must not execute the prefix as
// a tool call while silently dropping the garbage.
func TestChatStreamGarbageSuffixProducesNoToolCall(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{"a":1}`)).
		add(toolCallChunks(0, "call-1", "weather", `<garbage>`)).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})

	parts := collectStream(result.Stream)
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			t.Fatalf("garbage-suffixed input must not produce StreamToolCall, got input %q", string(tc.Input))
		}
	}
	var streamErr *StreamError
	for i := range parts {
		if e, ok := parts[i].(StreamError); ok {
			streamErr = &e
		}
	}
	if streamErr == nil {
		t.Fatal("no StreamError for garbage-suffixed tool input")
	}
	if _, ok := streamErr.Err.(InvalidResponseDataError); !ok {
		t.Fatalf("error = %T (%v), want InvalidResponseDataError", streamErr.Err, streamErr.Err)
	}
}

// TestChatStreamReusedIndexSplitsIntoNewSlot verifies that when an
// explicit index is reused with a new tool call id (provider fallback),
// the second call splits into a fresh slot instead of merging into the
// first call's arguments. Both calls complete independently.
func TestChatStreamReusedIndexSplitsIntoNewSlot(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{"a":1}`)).
		add(toolCallChunks(0, "call-2", "billing", `{"b":2}`)).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})

	parts := collectStream(result.Stream)
	byID := map[string]string{}
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			byID[tc.ToolCallID] = string(tc.Input)
		}
	}
	if len(byID) != 2 {
		t.Fatalf("expected 2 split tool calls, got %v", byID)
	}
	if byID["call-1"] != `{"a":1}` {
		t.Errorf("call-1 input = %q, want %q", byID["call-1"], `{"a":1}`)
	}
	if byID["call-2"] != `{"b":2}` {
		t.Errorf("call-2 input = %q, want %q", byID["call-2"], `{"b":2}`)
	}
}

// TestChatStreamMissingIndexSticksToLatestSlot verifies that a fragment
// with no index (Anthropic-style fallback) sticks to the latest open
// slot, so args-only continuations accumulate onto the current tool
// instead of opening a new slot (or erroring).
func TestChatStreamMissingIndexSticksToLatestSlot(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(-1, "call-1", "weather", `{"a":`)).
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"tool_calls":[{"function":{"arguments":"1}"}}]}}]}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})

	parts := collectStream(result.Stream)
	var toolCalls []StreamToolCall
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			toolCalls = append(toolCalls, tc)
		}
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 sticky tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].ToolCallID != "call-1" || string(toolCalls[0].Input) != `{"a":1}` {
		t.Fatalf("sticky tool call = %+v, want call-1 with %q", toolCalls[0], `{"a":1}`)
	}
}
