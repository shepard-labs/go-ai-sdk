package openai

import (
	"context"
	"net/http"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

func runCompletionWith(t *testing.T, opts GenerateOptions) (*GenerateResult, *recordingFetcher) {
	t.Helper()
	respBody := `{"id":"r","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"hi","finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), opts)
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	return result, f
}

// TestCompletionWarnsOnTopK verifies the completion model emits a warning
// when TopK is set (per spec, the completions API doesn't support topK).
func TestCompletionWarnsOnTopK(t *testing.T) {
	k := 5
	res, _ := runCompletionWith(t, GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		TopK:     &k,
	})
	found := false
	for _, w := range res.Warnings {
		if w.Feature == "topK" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected topK warning, got: %+v", res.Warnings)
	}
}

// TestCompletionWarnsOnTools verifies the completion model emits a warning
// when tools are provided (per spec, the completions API doesn't support tools).
func TestCompletionWarnsOnTools(t *testing.T) {
	res, _ := runCompletionWith(t, GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []openaicompatible.Tool{
			{Type: "function", Name: "x", InputSchema: map[string]any{"type": "object"}},
		},
	})
	found := false
	for _, w := range res.Warnings {
		if w.Feature == "tools" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected tools warning, got: %+v", res.Warnings)
	}
}

// TestCompletionWarnsOnToolChoice verifies the completion model emits a warning
// when tool_choice is provided.
func TestCompletionWarnsOnToolChoice(t *testing.T) {
	tc := openaicompatible.ToolChoice{Type: "auto"}
	res, _ := runCompletionWith(t, GenerateOptions{
		Messages:   []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ToolChoice: &tc,
	})
	found := false
	for _, w := range res.Warnings {
		if w.Feature == "toolChoice" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected toolChoice warning, got: %+v", res.Warnings)
	}
}

// TestCompletionWarnsOnJsonResponseFormat verifies the completion model emits
// a warning when ResponseFormat is JSON.
func TestCompletionWarnsOnJsonResponseFormat(t *testing.T) {
	res, _ := runCompletionWith(t, GenerateOptions{
		Messages:       []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ResponseFormat: &ResponseFormat{Type: "json"},
	})
	found := false
	for _, w := range res.Warnings {
		if w.Feature == "responseFormat" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected responseFormat warning, got: %+v", res.Warnings)
	}
}

// TestCompletionNoWarningsOnCleanRequest verifies a clean request has no warnings.
func TestCompletionNoWarningsOnCleanRequest(t *testing.T) {
	res, _ := runCompletionWith(t, GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if len(res.Warnings) > 0 {
		t.Errorf("expected no warnings, got: %+v", res.Warnings)
	}
}
