package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// Verifies that the completion prompt uses the <user>: / <assistant>:
// format per spec.
func TestCompletionPromptUsesUserAssistantFormat(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "Hello"}}},
			AssistantMessage{Content: []AssistantContent{TextContent{Text: "Hi"}}},
			UserMessage{Content: []UserContent{TextContent{Text: "Bye"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	prompt, _ := body["prompt"].(string)
	if !strings.Contains(prompt, "<user>:\nHello\n\n") {
		t.Errorf("prompt missing user greeting: %q", prompt)
	}
	if !strings.Contains(prompt, "<assistant>:\nHi\n\n") {
		t.Errorf("prompt missing assistant reply: %q", prompt)
	}
	if !strings.HasSuffix(prompt, "<assistant>:\n") {
		t.Errorf("prompt should end with <assistant>: prefix: %q", prompt)
	}
}

// Verifies that SystemMessage in a completion prompt throws
// InvalidPromptError per spec ("No system message support").
func TestCompletionRejectsSystemMessage(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{
			SystemMessage{Content: "be terse"},
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// Verifies that tool calls in an assistant message throw
// UnsupportedFunctionalityError per spec.
func TestCompletionRejectsToolCallsInAssistantMessage(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "call the tool"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{ToolCallContentEmbed: ToolCallContentEmbed{ToolCallID: "c1", ToolName: "f", Input: []byte(`{}`)}},
			}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// Verifies that ToolMessage throws UnsupportedFunctionalityError.
func TestCompletionRejectsToolMessage(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "x"}}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{},
			}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// Verifies that the auto-generated stop sequence "\n<user>:" is emitted
// when no user stop sequences are provided.
func TestCompletionAutoStopSequenceWhenNoUserStops(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	stop, _ := body["stop"].([]any)
	if len(stop) != 1 || stop[0] != "\n<user>:" {
		t.Errorf("stop: %v, want [\"\\n<user>:\"]", body["stop"])
	}
}
