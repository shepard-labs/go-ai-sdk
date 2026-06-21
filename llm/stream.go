package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
)

// CollectStream drains the stream channel and assembles a GenerateResult.
//
// It collects text content, reasoning content, tool calls, warnings,
// request/response metadata, finish reason, usage, and provider metadata.
//
// CollectStream handles termination as follows:
//   - StreamFinish freezes the result (finish reason, usage, provider metadata)
//     and CollectStream returns once the channel closes.
//   - StreamError returns the partial result built so far together with the error.
//   - A channel close without a StreamFinish or StreamError returns the partial
//     result and an error.
//   - Context cancellation drains any buffered parts until the channel closes,
//     then returns the partial result and ctx.Err().
func CollectStream(ctx context.Context, parts <-chan StreamPart) (*GenerateResult, error) {
	result := &GenerateResult{}

	var textBuf strings.Builder
	var reasoningBuf strings.Builder
	var inText bool
	var inReasoning bool

	type pendingToolCall struct {
		id   string
		name string
		json strings.Builder
	}
	toolCalls := map[string]*pendingToolCall{}
	var toolOrder []string

	finishToolCalls := func() {
		for _, id := range toolOrder {
			tc := toolCalls[id]
			var input json.RawMessage
			if s := tc.json.String(); s != "" {
				input = json.RawMessage(s)
			}
			result.Content = append(result.Content, ToolUseContent{ID: tc.id, Name: tc.name, Input: input})
		}
		toolOrder = nil
	}

	finishOpenBlocks := func() {
		if inText {
			result.Content = append(result.Content, TextContent{Text: textBuf.String()})
			textBuf.Reset()
			inText = false
		}
		if inReasoning {
			result.Content = append(result.Content, ReasoningContent{Text: reasoningBuf.String()})
			reasoningBuf.Reset()
			inReasoning = false
		}
		finishToolCalls()
	}

	var sawFinish bool

	for {
		// Check cancellation first: when ctx is already cancelled and parts has
		// buffered data, a bare select would pick randomly between the two ready
		// cases. Prioritizing ctx makes cancellation deterministic.
		if err := ctx.Err(); err != nil {
			// Drain buffered parts until the channel closes, then return the
			// partial result and the context error.
			for range parts {
			}
			finishOpenBlocks()
			return result, err
		}
		select {
		case <-ctx.Done():
			// Drain buffered parts until the channel closes, then return the
			// partial result and the context error.
			for range parts {
			}
			finishOpenBlocks()
			return result, ctx.Err()
		case part, ok := <-parts:
			if !ok {
				finishOpenBlocks()
				if !sawFinish {
					return result, errors.New("llm: stream closed without finish")
				}
				return result, nil
			}

			switch p := part.(type) {
			case StreamTextStart:
				inText = true
			case StreamTextDelta:
				inText = true
				textBuf.WriteString(p.Text)
			case StreamTextEnd:
				if inText {
					result.Content = append(result.Content, TextContent{Text: textBuf.String()})
					textBuf.Reset()
					inText = false
				}
			case StreamReasoningStart:
				inReasoning = true
			case StreamReasoningDelta:
				inReasoning = true
				reasoningBuf.WriteString(p.Text)
			case StreamReasoningEnd:
				if inReasoning {
					result.Content = append(result.Content, ReasoningContent{Text: reasoningBuf.String()})
					reasoningBuf.Reset()
					inReasoning = false
				}
			case StreamToolCallStart:
				if _, exists := toolCalls[p.ID]; !exists {
					toolCalls[p.ID] = &pendingToolCall{id: p.ID, name: p.Name}
					toolOrder = append(toolOrder, p.ID)
				}
			case StreamToolInputDelta:
				if tc, exists := toolCalls[p.ID]; exists {
					tc.json.WriteString(p.JSON)
				}
			case StreamToolInputEnd:
				if tc, exists := toolCalls[p.ID]; exists {
					var input json.RawMessage
					if p.Input != nil {
						input = p.Input
					} else if s := tc.json.String(); s != "" {
						input = json.RawMessage(s)
					}
					result.Content = append(result.Content, ToolUseContent{ID: tc.id, Name: tc.name, Input: input})
					delete(toolCalls, p.ID)
					for i, id := range toolOrder {
						if id == p.ID {
							toolOrder = append(toolOrder[:i], toolOrder[i+1:]...)
							break
						}
					}
				}
			case StreamWarning:
				result.Warnings = append(result.Warnings, p.Warning)
			case StreamMetadata:
				result.Request = p.Request
				result.Response = p.Response
			case StreamFinish:
				sawFinish = true
				result.FinishReason = p.FinishReason
				result.Usage = p.Usage
				result.ProviderMetadata = p.ProviderMetadata
			case StreamError:
				finishOpenBlocks()
				return result, p.Err
			}
		}
	}
}
