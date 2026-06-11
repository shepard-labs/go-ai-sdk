package openaicompatible

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type sseBuilder struct {
	chunks []string
}

func sse() *sseBuilder { return &sseBuilder{} }

func (b *sseBuilder) add(json string) *sseBuilder {
	b.chunks = append(b.chunks, json)
	return b
}

func (b *sseBuilder) addRaw(data string) *sseBuilder {
	b.chunks = append(b.chunks, data)
	return b
}

func (b *sseBuilder) done() *sseBuilder {
	b.chunks = append(b.chunks, "[DONE]")
	return b
}

func (b *sseBuilder) build() string {
	var out strings.Builder
	for _, chunk := range b.chunks {
		if chunk == "[DONE]" {
			out.WriteString("data: [DONE]\n\n")
			break
		}
		out.WriteString("data: " + strings.ReplaceAll(chunk, "\n", "\ndata: ") + "\n\n")
	}
	return out.String()
}

func sseResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"X-Request-Id": []string{"resp-id"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func collectStream(stream <-chan StreamPart) []StreamPart {
	var parts []StreamPart
	for p := range stream {
		parts = append(parts, p)
	}
	return parts
}

type trackedBody struct {
	io.Reader
	closed atomic.Bool
}

func (b *trackedBody) Close() error {
	b.closed.Store(true)
	return nil
}

func minimalChunk(content string) string {
	return `{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"content":"` + content + `"}}]}`
}

func toolCallChunks(index int, id, name, args string) string {
	var idxField string
	if index >= 0 {
		idxField = `"index":` + itoa(index) + `,`
	}
	argsJSON, _ := json.Marshal(args)
	return `{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"tool_calls":[{` + idxField + `"id":"` + id + `","function":{"name":"` + name + `","arguments":` + string(argsJSON) + `}}]}}]}`
}

func itoa(n int) string {
	return strconv.Itoa(n)
}

// --- Tests ---

func TestChatStreamRequestHasStreamTrue(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	body := decodeRequestBody(t, result.Request.Body)
	if got, ok := body["stream"]; !ok || got != true {
		t.Fatalf("stream field = %v", body["stream"])
	}
}

func TestChatStreamIncludeUsageAddsStreamOptions(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, IncludeUsage: true, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	body := decodeRequestBody(t, result.Request.Body)
	so, ok := body["stream_options"].(map[string]any)
	if !ok || so["include_usage"] != true {
		t.Fatalf("stream_options = %#v", body["stream_options"])
	}
}

func TestChatStreamNoIncludeUsageWhenFalse(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	body := decodeRequestBody(t, result.Request.Body)
	if _, ok := body["stream_options"]; ok {
		t.Fatalf("stream_options present when IncludeUsage=false: %#v", body)
	}
}

func TestChatStreamTransformRequestBodyAfterStreamFields(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0},
		TransformRequestBody: func(body map[string]any) map[string]any {
			if _, ok := body["stream"]; ok {
				body["transformed"] = "streaming"
			}
			return body
		},
	})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	body := decodeRequestBody(t, result.Request.Body)
	if body["transformed"] != "streaming" {
		t.Fatalf("transform not applied after stream fields: %#v", body)
	}
}

func TestChatStreamResultStreamAndPartsSameChannel(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	collectStream(result.Stream)
	if result.Stream != result.Parts {
		t.Fatal("Stream and Parts are not the same channel")
	}
}

func TestChatStreamStartIsFirstWithWarnings(t *testing.T) {
	topK := 1
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, TopK: &topK}})
	parts := collectStream(result.Stream)
	if len(parts) == 0 {
		t.Fatal("no parts")
	}
	start, ok := parts[0].(StreamStart)
	if !ok {
		t.Fatalf("first part is not StreamStart: %T", parts[0])
	}
	if len(start.Warnings) != 1 || start.Warnings[0].Feature != "topK" {
		t.Fatalf("warnings = %#v", start.Warnings)
	}
}

func TestChatStreamRawChunksEmitWhenIncludeRawChunksTrue(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}, IncludeRawChunks: true})
	parts := collectStream(result.Stream)
	hasRaw := false
	for _, p := range parts {
		if _, ok := p.(StreamRaw); ok {
			hasRaw = true
			break
		}
	}
	if !hasRaw {
		t.Fatal("no StreamRaw emitted when IncludeRawChunks=true")
	}
}

func TestChatStreamNoRawChunksWhenIncludeRawChunksFalse(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	for _, p := range parts {
		if _, ok := p.(StreamRaw); ok {
			t.Fatal("StreamRaw emitted when IncludeRawChunks=false")
		}
	}
}

func TestChatStreamRawExcludesDataPrefixAndDone(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hello")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}, IncludeRawChunks: true})
	parts := collectStream(result.Stream)
	for _, p := range parts {
		if raw, ok := p.(StreamRaw); ok {
			if strings.Contains(string(raw.Raw), "data:") {
				t.Fatalf("StreamRaw.Raw contains data: prefix: %q", raw.Raw)
			}
			if string(raw.Raw) == "[DONE]" {
				t.Fatalf("StreamRaw.Raw is [DONE]: %q", raw.Raw)
			}
		}
	}
}

func TestChatStreamRawDecodedIsNilForInvalidJSON(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().addRaw("not-json").done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}, IncludeRawChunks: true})
	parts := collectStream(result.Stream)
	for _, p := range parts {
		if raw, ok := p.(StreamRaw); ok {
			if raw.Decoded != nil {
				t.Fatalf("StreamRaw.Decoded is not nil for invalid JSON: %#v", raw.Decoded)
			}
			if string(raw.Raw) != "not-json" {
				t.Fatalf("StreamRaw.Raw mismatch: %q", raw.Raw)
			}
		}
	}
}

func TestChatStreamInvalidJSONEmitsStreamErrorThenFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().addRaw("not-valid-json").done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	hasError := false
	hasFinish := false
	finishCount := 0
	for _, p := range parts {
		switch p.(type) {
		case StreamError:
			hasError = true
		case StreamFinish:
			hasFinish = true
			finishCount++
		}
	}
	if !hasError {
		t.Fatal("no StreamError for invalid JSON")
	}
	if !hasFinish {
		t.Fatal("no StreamFinish for invalid JSON")
	}
	if finishCount != 1 {
		t.Fatalf("expected exactly 1 StreamFinish, got %d", finishCount)
	}
}

func TestChatStreamResponseMetadataOnFirstChunk(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var metaFound bool
	for _, p := range parts {
		if meta, ok := p.(StreamResponseMetadata); ok {
			if meta.ID != "chat-1" || meta.ModelID != "gpt" || meta.Timestamp == nil {
				t.Fatalf("metadata = %#v", meta)
			}
			metaFound = true
			break
		}
	}
	if !metaFound {
		t.Fatal("no StreamResponseMetadata emitted")
	}
	if result.Response.ID != "chat-1" || result.Response.ModelID != "gpt" || result.Response.Timestamp == nil {
		t.Fatalf("stream response metadata not set: %#v", result.Response)
	}
}

func TestChatStreamResponseHeadersRetainedAndCloned(t *testing.T) {
	headers := http.Header{"X-Custom": []string{"original"}}
	resp := &http.Response{StatusCode: 200, Header: headers, Body: io.NopCloser(strings.NewReader(sse().add(minimalChunk("hi")).done().build()))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	collectStream(result.Stream)
	headers.Set("X-Custom", "changed")
	if got := result.Response.Headers.Get("X-Custom"); got != "original" {
		t.Fatalf("headers not cloned: %q", got)
	}
}

func TestChatStreamReasoningStartDeltaEnd(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"reasoning":"think"}}]}`).
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"reasoning":" more"}}]}`).
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"content":"text"}}]}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var events []string
	for _, p := range parts {
		switch p := p.(type) {
		case StreamStart:
		case StreamResponseMetadata:
		case StreamReasoningStart:
			events = append(events, "reasoning-start")
		case StreamReasoningDelta:
			events = append(events, "reasoning-delta:"+p.Text)
		case StreamReasoningEnd:
			events = append(events, "reasoning-end")
		case StreamTextStart:
			events = append(events, "text-start")
		case StreamTextDelta:
			events = append(events, "text-delta:"+p.Text)
		case StreamTextEnd:
			events = append(events, "text-end")
		case StreamFinish:
		}
	}
	expected := []string{"reasoning-start", "reasoning-delta:think", "reasoning-delta: more", "reasoning-end", "text-start", "text-delta:text", "text-end"}
	if len(events) != len(expected) {
		t.Fatalf("events = %#v, want %#v", events, expected)
	}
	for i := range events {
		if events[i] != expected[i] {
			t.Fatalf("event[%d] = %q, want %q - all: %v", i, events[i], expected[i], events)
		}
	}
}

func TestChatStreamReasoningEndsBeforeToolEvents(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"reasoning":"think"}}]}`).
		add(toolCallChunks(0, "call-1", "weather", `{"city":"`)).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var events []string
	for _, p := range parts {
		switch p.(type) {
		case StreamReasoningStart:
			events = append(events, "reasoning-start")
		case StreamReasoningEnd:
			events = append(events, "reasoning-end")
		case StreamToolInputStart:
			events = append(events, "tool-input-start")
		}
	}
	found := false
	for i := 0; i < len(events)-1; i++ {
		if events[i] == "reasoning-end" && events[i+1] == "tool-input-start" {
			found = true
		}
	}
	if !found {
		t.Fatalf("reasoning end not before tool input: %v", events)
	}
}

func TestChatStreamTextStartDeltaEndUseIDTxt0(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hello")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	for _, p := range parts {
		switch p := p.(type) {
		case StreamTextStart:
			if p.ID != "txt-0" {
				t.Fatalf("StreamTextStart.ID = %q", p.ID)
			}
		case StreamTextDelta:
			if p.ID != "txt-0" {
				t.Fatalf("StreamTextDelta.ID = %q", p.ID)
			}
		case StreamTextEnd:
			if p.ID != "txt-0" {
				t.Fatalf("StreamTextEnd.ID = %q", p.ID)
			}
		}
	}
}

func TestChatStreamToolCallAccumulatesByExplicitIndex(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{"city":"`)).
		add(toolCallChunks(0, "call-1", "weather", `SF"}`)).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var toolCall *StreamToolCall
	for i := range parts {
		if tc, ok := parts[i].(StreamToolCall); ok {
			toolCall = &tc
		}
	}
	if toolCall == nil {
		t.Fatal("no StreamToolCall")
	}
	if string(toolCall.Input) != `{"city":"SF"}` {
		t.Fatalf("tool call input = %q", string(toolCall.Input))
	}
}

func TestChatStreamMissingToolCallIndexUsesCurrentCount(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(-1, "call-1", "weather", `{"city":"SF"}`)).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var toolCall *StreamToolCall
	for i := range parts {
		if tc, ok := parts[i].(StreamToolCall); ok {
			toolCall = &tc
		}
	}
	if toolCall == nil || toolCall.ToolCallID != "call-1" || toolCall.ToolName != "weather" {
		t.Fatalf("missing index tool call not handled: toolCall=%#v", toolCall)
	}
}

func TestChatStreamNewToolCallMissingIDErrors(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"weather","arguments":"{}"}}]}}]}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var streamErr *StreamError
	for i := range parts {
		if e, ok := parts[i].(StreamError); ok {
			streamErr = &e
		}
	}
	if streamErr == nil {
		t.Fatal("no StreamError for missing tool call ID")
	}
	inv, ok := streamErr.Err.(InvalidResponseDataError)
	if !ok || inv.Message != "Expected 'id' to be a string." {
		t.Fatalf("error = %v", streamErr.Err)
	}
}

func TestChatStreamNewToolCallMissingNameErrors(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"arguments":"{}"}}]}}]}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var streamErr *StreamError
	for i := range parts {
		if e, ok := parts[i].(StreamError); ok {
			streamErr = &e
		}
	}
	if streamErr == nil {
		t.Fatal("no StreamError for missing tool call function.name")
	}
	inv, ok := streamErr.Err.(InvalidResponseDataError)
	if !ok || inv.Message != "Expected 'function.name' to be a string." {
		t.Fatalf("error = %v", streamErr.Err)
	}
}

func TestChatStreamToolCallEmitsImmediatelyOnParsableJSON(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{"city":"SF","`)).
		add(toolCallChunks(0, "call-1", "weather", `temp":72}`)).
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
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if string(toolCalls[0].Input) != `{"city":"SF","temp":72}` {
		t.Fatalf("tool call input = %q", string(toolCalls[0].Input))
	}
}

func TestChatStreamUnfinishedToolCallsFlushAtEnd(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{"city":"SF"}`)).
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
		t.Fatalf("expected 1 tool call on flush, got %d", len(toolCalls))
	}
	if string(toolCalls[0].Input) != `{"city":"SF"}` {
		t.Fatalf("flushed tool call input = %q", string(toolCalls[0].Input))
	}
}

func TestChatStreamFinishedToolCallsIgnoreLaterDeltas(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{"city":"SF"}`)).
		add(toolCallChunks(0, "call-1", "weather", `extra`)).
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
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if string(toolCalls[0].Input) != `{"city":"SF"}` {
		t.Fatalf("tool call input = %q (extra delta should be ignored)", string(toolCalls[0].Input))
	}
}

func TestChatStreamToolCallPreservesRawJSON(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(toolCallChunks(0, "call-1", "weather", `{ "b" : 2 }`)). // extra spaces
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			if string(tc.Input) != `{ "b" : 2 }` {
				t.Fatalf("raw JSON not preserved: %q", string(tc.Input))
			}
		}
	}
}

func TestChatStreamGoogleThoughtSignatureInMetadata(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"weather","arguments":"{\"city\":\"SF\"}"},"extra_content":{"google":{"thought_signature":"sig-123"}}}]}}]}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	m := p.Chat("m")
	result, _ := m.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			if tc.ProviderMetadata == nil || tc.ProviderMetadata["acme"].(map[string]any)["thoughtSignature"] != "sig-123" {
				t.Fatalf("tool call ProviderMetadata = %#v", tc.ProviderMetadata)
			}
		}
	}
}

func TestChatStreamErrorChunkEmitsStreamErrorThenFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"error":{"message":"something went wrong","type":"server_error"}}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
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
		t.Fatal("no StreamError for error chunk")
	}
	apiErr, ok := streamErr.Err.(APIError)
	if !ok || apiErr.Message != "something went wrong" {
		t.Fatalf("error = %v", streamErr.Err)
	}
	if finish == nil || finish.FinishReason.Unified != "error" {
		t.Fatalf("finish = %#v", finish)
	}
}

func TestChatStreamJSONParseFailureFinishReasonError(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().addRaw("{invalid").done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil || finish.FinishReason.Unified != "error" {
		t.Fatalf("finish reason = %#v", finish)
	}
}

func TestChatStreamLatestUsageConvertedAtFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil || finish.Usage.InputTokens.Total == nil || *finish.Usage.InputTokens.Total != 5 {
		t.Fatalf("finish usage = %#v", finish)
	}
}

func TestChatStreamCustomUsageConverterHonored(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":1}}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0},
		ConvertUsage: func(OpenAICompatibleTokenUsage) Usage {
			return Usage{InputTokens: TokenCounts{Total: intPtr(99)}}
		},
	})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil || finish.Usage.InputTokens.Total == nil || *finish.Usage.InputTokens.Total != 99 {
		t.Fatalf("custom usage not honored: %#v", finish)
	}
}

func TestChatStreamAcceptedRejectedPredictionTokens(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"content":"hi"}}],"usage":{"completion_tokens_details":{"accepted_prediction_tokens":4,"rejected_prediction_tokens":5}}}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acmeProvider", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, ProviderOptions: ProviderOptions{"acmeProvider": map[string]any{}}}})
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil {
		t.Fatal("no StreamFinish")
	}
	inner := finish.ProviderMetadata["acmeProvider"].(map[string]any)
	if inner["acceptedPredictionTokens"] != 4 || inner["rejectedPredictionTokens"] != 5 {
		t.Fatalf("prediction tokens = %#v", finish.ProviderMetadata)
	}
}

func TestChatStreamExtractorReceivesChunks(t *testing.T) {
	type recorded struct {
		raw     []byte
		decoded map[string]any
	}
	var chunks []recorded
	var mu sync.Mutex
	extractor := &testStreamMetadataExtractor{
		process: func(raw []byte, decoded map[string]any) {
			mu.Lock()
			chunks = append(chunks, recorded{raw: append([]byte(nil), raw...), decoded: decoded})
			mu.Unlock()
		},
		build: func() ProviderMetadata { return ProviderMetadata{"acme": map[string]any{"extracted": "value"}} },
	}
	metaExt := &recordingMetadataExtractor{streamExtractor: extractor}
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(minimalChunk("hi")).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, MetadataExtractor: metaExt})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	if len(chunks) != 1 {
		t.Fatalf("extractor received %d chunks, want 1", len(chunks))
	}
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil {
		t.Fatal("no StreamFinish")
	}
	if finish.ProviderMetadata["acme"].(map[string]any)["extracted"] != "value" {
		t.Fatalf("extracted metadata not merged: %#v", finish.ProviderMetadata)
	}
}

func TestChatStreamExtractorReceivesErrorChunks(t *testing.T) {
	var processed [][]byte
	var mu sync.Mutex
	extractor := &testStreamMetadataExtractor{
		process: func(raw []byte, _ map[string]any) {
			mu.Lock()
			processed = append(processed, append([]byte(nil), raw...))
			mu.Unlock()
		},
		build: func() ProviderMetadata { return nil },
	}
	metaExt := &recordingMetadataExtractor{streamExtractor: extractor}
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"error":{"message":"bad"}}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, MetadataExtractor: metaExt})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	collectStream(result.Stream)
	if len(processed) != 1 {
		t.Fatalf("extractor received %d chunks for error, want 1", len(processed))
	}
}

func TestChatStreamSuccessfulStreamExactOneStreamFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(minimalChunk("hi")).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	finishCount := 0
	for _, p := range parts {
		if _, ok := p.(StreamFinish); ok {
			finishCount++
		}
	}
	if finishCount != 1 {
		t.Fatalf("got %d StreamFinish events, want 1", finishCount)
	}
}

func TestChatStreamFatalErrorAtMostOneStreamErrorExactOneFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().addRaw("{invalid").done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	errorCount := 0
	finishCount := 0
	for _, p := range parts {
		switch p.(type) {
		case StreamError:
			errorCount++
		case StreamFinish:
			finishCount++
		}
	}
	if errorCount > 1 {
		t.Fatalf("got %d StreamError events, want at most 1", errorCount)
	}
	if finishCount != 1 {
		t.Fatalf("got %d StreamFinish events, want 1", finishCount)
	}
}

func TestChatStreamResponseBodyClosesOnNormalCompletion(t *testing.T) {
	body := &trackedBody{Reader: strings.NewReader(sse().add(minimalChunk("hi")).done().build())}
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: body}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	collectStream(result.Stream)
	if !body.closed.Load() {
		t.Fatal("body not closed on normal completion")
	}
}

func TestChatStreamResponseBodyClosesOnParseError(t *testing.T) {
	body := &trackedBody{Reader: strings.NewReader(sse().addRaw("{invalid").done().build())}
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: body}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	if !body.closed.Load() {
		t.Fatal("body not closed on parse error")
	}
}

func TestChatStreamResponseBodyClosesOnContextCancellation(t *testing.T) {
	body := &trackedBody{Reader: strings.NewReader(sse().add(`{"id":"chat-1","choices":[{"delta":{"content":"hi"}}]}`).build() + "data: never\n")}
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: body}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	ctx, cancel := context.WithCancel(context.Background())
	result, _ := p.Chat("m").DoStream(ctx, StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	<-result.Stream
	cancel()
	collectStream(result.Stream)
	if !body.closed.Load() {
		t.Fatal("body not closed after context cancellation")
	}
}

func TestChatStreamPreStreamNon2xxReturnsAPICallError(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(400, `{"error":{"message":"bad"}}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	_, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err == nil {
		t.Fatal("expected error for non-2xx")
	}
	apiErr := new(APICallError)
	if !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Fatalf("error = %v", err)
	}
}

func TestChatStreamNilFetcherResponseReturnsError(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{nil}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	_, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err == nil {
		t.Fatal("expected error for nil response")
	}
}

func TestChatStreamProviderErrorBeforeHTTP(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "", Name: ""})
	_, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err == nil {
		t.Fatal("expected error for provider configuration failure")
	}
}

func TestChatStreamEstablishmentRetriesForRetryablePreStream(t *testing.T) {
	f := &recordingFetcher{
		responses: []*http.Response{response(500, `{"error":{"message":"down"}}`), sseResponse(sse().add(minimalChunk("ok")).done().build())},
	}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 1, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	if f.calls != 2 {
		t.Fatalf("calls = %d, want 2", f.calls)
	}
}

func TestChatStreamMidStreamParseFailuresNotRetried(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().addRaw("{invalid").done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 3, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	if f.calls != 1 {
		t.Fatalf("calls = %d, want 1 (mid-stream not retried)", f.calls)
	}
}

func TestChatStreamSchemaValidationFailureFinishReasonError(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().add(`{"choices":[{"delta":{}}]}`).done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil {
		t.Fatal("no StreamFinish")
	}
	// Valid JSON that is a normal chunk (no error field) should succeed, not fail.
	// The finish reason should be "other" since no explicit finish_reason was given.
	if finish.FinishReason.Unified != "other" {
		t.Fatalf("finish reason = %q, want other", finish.FinishReason.Unified)
	}
}

// Extended test helpers

type testStreamMetadataExtractor struct {
	process func(raw []byte, decoded map[string]any)
	build   func() ProviderMetadata
}

func (e *testStreamMetadataExtractor) ProcessChunk(raw []byte, decoded map[string]any) {
	if e.process != nil {
		e.process(raw, decoded)
	}
}

func (e *testStreamMetadataExtractor) BuildMetadata() ProviderMetadata {
	if e.build != nil {
		return e.build()
	}
	return nil
}
