package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
)

type emptyToolDispatcher struct{}

func (emptyToolDispatcher) Dispatch(context.Context, string, json.RawMessage) (json.RawMessage, error) {
	return nil, fmt.Errorf("unknown tool")
}

func TestREQTOOL002_AgentLoopSurfacesIsError(t *testing.T) {
	client := &mockClient{results: []*GenerateResult{
		{FinishReason: FinishReasonToolCalls, Content: []Content{ToolUseContent{ID: "1", Name: "missing"}}},
		{FinishReason: FinishReasonStop},
	}}
	messages, _, err := AgentLoop(context.Background(), client, GenerateOptions{}, emptyToolDispatcher{}, "", 3)
	if err != nil {
		t.Fatalf("AgentLoop error = %v", err)
	}
	result := messages[1].Content[0].(ToolResultContent)
	if !result.IsError {
		t.Fatalf("tool result = %#v, want IsError", result)
	}
}

func TestREQLLM002_CancellationPropagatesFromClient(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := loopClientFunc(func(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
		return nil, ctx.Err()
	})
	_, _, err := AgentLoop(ctx, client, GenerateOptions{}, emptyToolDispatcher{}, "", 1)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}
