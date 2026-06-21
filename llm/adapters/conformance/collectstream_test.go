package conformance

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// TestCollectStreamAssemblesResult feeds a hand-crafted StreamPart channel
// through llm.CollectStream and verifies the assembled GenerateResult matches
// the expected shape. It is provider-agnostic.
func TestCollectStreamAssemblesResult(t *testing.T) {
	parts := []llm.StreamPart{
		llm.StreamMetadata{Response: llm.ResponseMetadata{ID: "resp_1", ModelID: "m-x"}},
		llm.StreamTextStart{},
		llm.StreamTextDelta{Text: "Hello, "},
		llm.StreamTextDelta{Text: "world"},
		llm.StreamTextEnd{},
		llm.StreamReasoningStart{},
		llm.StreamReasoningDelta{Text: "let me think"},
		llm.StreamReasoningEnd{},
		llm.StreamToolCallStart{ID: "call_1", Name: "lookup"},
		llm.StreamToolInputDelta{ID: "call_1", JSON: `{"q":`},
		llm.StreamToolInputDelta{ID: "call_1", JSON: `"go"}`},
		llm.StreamToolInputEnd{ID: "call_1"},
		llm.StreamWarning{Warning: llm.Warning{Code: "x", Message: "heads up", Provider: "fake"}},
		llm.StreamFinish{
			FinishReason: llm.FinishReason{Unified: llm.FinishReasonToolCalls, Raw: "tool_calls"},
			Usage:        llm.Usage{InputTokens: 4, OutputTokens: 6, ReasoningTokens: 2},
		},
	}

	ch := make(chan llm.StreamPart)
	go func() {
		defer close(ch)
		for _, p := range parts {
			ch <- p
		}
	}()

	res, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream error = %v", err)
	}

	text, ok := firstContent[llm.TextContent](res.Content)
	if !ok || text.Text != "Hello, world" {
		t.Fatalf("text content = %#v, want %q", res.Content, "Hello, world")
	}
	reasoning, ok := firstContent[llm.ReasoningContent](res.Content)
	if !ok || reasoning.Text != "let me think" {
		t.Fatalf("reasoning content = %#v, want %q", res.Content, "let me think")
	}
	use, ok := firstContent[llm.ToolUseContent](res.Content)
	if !ok || use.ID != "call_1" || use.Name != "lookup" {
		t.Fatalf("tool content = %#v, want ToolUseContent{call_1,lookup}", res.Content)
	}
	var decoded map[string]any
	if err := json.Unmarshal(use.Input, &decoded); err != nil || decoded["q"] != "go" {
		t.Fatalf("tool input = %s, want {\"q\":\"go\"}", use.Input)
	}

	if res.FinishReason.Unified != llm.FinishReasonToolCalls {
		t.Fatalf("finish reason = %q, want tool-calls", res.FinishReason.Unified)
	}
	if res.Usage.InputTokens != 4 || res.Usage.OutputTokens != 6 || res.Usage.ReasoningTokens != 2 {
		t.Fatalf("usage = %#v, want 4/6/2", res.Usage)
	}
	if res.Response.ID != "resp_1" || res.Response.ModelID != "m-x" {
		t.Fatalf("response metadata = %#v", res.Response)
	}
	if len(res.Warnings) != 1 || res.Warnings[0].Message != "heads up" {
		t.Fatalf("warnings = %#v, want one 'heads up'", res.Warnings)
	}
}
