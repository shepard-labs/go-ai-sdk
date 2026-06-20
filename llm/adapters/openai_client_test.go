package adapters

import (
	"context"
	"encoding/json"
	"regexp"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

type fakeOpenAIModel struct {
	lastOptions openai.GenerateOptions
	result      *openai.GenerateResult
	err         error
}

func (f *fakeOpenAIModel) ModelID() string                          { return "fake" }
func (f *fakeOpenAIModel) Provider() string                         { return "fake" }
func (f *fakeOpenAIModel) SupportURLs() map[string][]*regexp.Regexp { return nil }
func (f *fakeOpenAIModel) DoGenerate(ctx context.Context, opts openai.GenerateOptions) (*openai.GenerateResult, error) {
	f.lastOptions = opts
	return f.result, f.err
}
func (f *fakeOpenAIModel) DoStream(ctx context.Context, opts openai.StreamOptions) (*openai.StreamResult, error) {
	return nil, nil
}

func TestOpenAIAdapterTranslatesAndReturns(t *testing.T) {
	model := &fakeOpenAIModel{result: &openai.GenerateResult{
		FinishReason: openai.FinishReason{Unified: "tool-calls"},
		Content: []openai.Content{
			openai.TextContent{Text: "hi"},
			openai.ToolCallContent{ToolCallContentEmbed: openai.ToolCallContentEmbed{ToolCallID: "id", ToolName: "t", Input: json.RawMessage(`{}`)}},
		},
		Usage: openai.Usage{InputTokens: openai.TokenCounts{Total: intPtr(3)}, OutputTokens: openai.OutputTokenCounts{Total: intPtr(4)}},
	}}
	result, err := NewOpenAIAdapter(model).Generate(context.Background(), GenerateOptions{
		System:   "sys",
		Messages: []Message{{Role: "user", Content: []Content{TextContent{Text: "q"}}}},
	})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if _, ok := model.lastOptions.Messages[0].(openai.SystemMessage); !ok {
		t.Fatalf("message 0 = %#v, want SystemMessage", model.lastOptions.Messages[0])
	}
	if result.FinishReason != FinishReasonToolCalls {
		t.Fatalf("finish = %q", result.FinishReason)
	}
	if _, ok := result.Content[1].(ToolUseContent); !ok {
		t.Fatalf("content 1 = %#v, want ToolUseContent", result.Content[1])
	}
	if result.Usage.InputTokens != 3 || result.Usage.OutputTokens != 4 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}
