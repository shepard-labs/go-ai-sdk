package openaicompatible

import (
	"context"
	"net/http"
	"testing"
)

func TestChatGenerateParsesArrayContentThinkingAndText(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":[{"type":"thinking","thinking":"deep thought"},{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.test/i.png"}},{"type":"text","text":" world"}],"reasoning_content":""},"finish_reason":"stop"}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if len(result.Content) != 2 {
		t.Fatalf("content = %#v, want 2 parts", result.Content)
	}
	text, ok := result.Content[0].(TextContent)
	if !ok || text.Text != "hello world" {
		t.Fatalf("text = %#v, want %q", result.Content[0], "hello world")
	}
	reasoning, ok := result.Content[1].(ReasoningContent)
	if !ok || reasoning.Text != "deep thought" {
		t.Fatalf("reasoning = %#v, want %q", result.Content[1], "deep thought")
	}
}

func TestChatGenerateArrayContentMergesWithReasoningFields(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":[{"type":"thinking","thinking":"more"}],"reasoning_content":"base "},"finish_reason":"stop"}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content = %#v, want 1 reasoning part", result.Content)
	}
	reasoning, ok := result.Content[0].(ReasoningContent)
	if !ok || reasoning.Text != "base more" {
		t.Fatalf("reasoning = %#v, want %q", result.Content[0], "base more")
	}
}

func TestChatStreamParsesArrayContentThinkingAndText(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"content":[{"type":"thinking","thinking":"hmm"}]}}]}`).
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{"content":[{"type":"text","text":"hi"},{"type":"unknown","data":"x"}]}}]}`).
		add(`{"id":"chat-1","created":10,"model":"gpt","choices":[{"delta":{},"finish_reason":"stop"}]}`).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var events []string
	var finish *StreamFinish
	for _, p := range parts {
		switch p := p.(type) {
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
			finish = &p
		case StreamError:
			t.Fatalf("unexpected StreamError: %v", p.Err)
		}
	}
	expected := []string{"reasoning-start", "reasoning-delta:hmm", "reasoning-end", "text-start", "text-delta:hi", "text-end"}
	if len(events) != len(expected) {
		t.Fatalf("events = %#v, want %#v", events, expected)
	}
	for i := range events {
		if events[i] != expected[i] {
			t.Fatalf("event[%d] = %q, want %q - all: %v", i, events[i], expected[i], events)
		}
	}
	if finish == nil || finish.FinishReason.Unified != "stop" {
		t.Fatalf("finish = %#v, want stop", finish)
	}
}
