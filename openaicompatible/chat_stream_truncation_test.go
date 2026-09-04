package openaicompatible

import (
	"context"
	"net/http"
	"testing"
)

func TestChatStreamTruncatedWithoutFinishReasonErrors(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{sseResponse(sse().
		add(minimalChunk("hel")).
		add(minimalChunk("lo")).
		done().build())}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, _ := p.Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}}})
	parts := collectStream(result.Stream)
	var streamErr *StreamError
	var finish *StreamFinish
	var text string
	for i := range parts {
		switch p := parts[i].(type) {
		case StreamTextDelta:
			text += p.Text
		case StreamError:
			streamErr = &p
		case StreamFinish:
			finish = &p
		}
	}
	// Partial content may still be delivered, but the stream must not end
	// as a successful finish when no finish reason was received.
	if text != "hello" {
		t.Fatalf("text = %q, want %q", text, "hello")
	}
	if streamErr == nil {
		t.Fatal("no StreamError for truncated stream")
	}
	if _, ok := streamErr.Err.(InvalidResponseDataError); !ok {
		t.Fatalf("error = %T (%v), want InvalidResponseDataError", streamErr.Err, streamErr.Err)
	}
	if finish == nil || finish.FinishReason.Unified != "error" {
		t.Fatalf("finish = %#v, want error", finish)
	}
}
