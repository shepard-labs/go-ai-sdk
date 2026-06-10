package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"strings"
)

type pendingToolCall struct {
	id, name string
	args     strings.Builder
	finished bool
}

func (m *cohereChatLanguageModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildChatRequest(opts.GenerateOptions)
	if err != nil {
		return nil, err
	}
	config.Body["stream"] = true
	bodyBytes, err := json.Marshal(config.Body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeStream(ctx, endpointChat, bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	parts := make(chan StreamPart)
	sresp := &StreamResponse{Headers: cloneHeader(resp.Headers)}
	go m.runCohereStream(ctx, resp, parts, sresp, config.Warnings, opts)
	return &StreamResult{Stream: parts, Parts: parts, Request: RequestMetadata{Body: append([]byte(nil), bodyBytes...)}, Response: sresp}, nil
}

func (m *cohereChatLanguageModel) runCohereStream(ctx context.Context, resp *httpStreamResponse, parts chan<- StreamPart, sresp *StreamResponse, warnings []Warning, opts StreamOptions) {
	defer close(parts)
	defer resp.Body.Close()
	parts <- StreamStart{Warnings: warnings}
	finish := FinishReason{Unified: "other", Raw: ""}
	var usage *CohereUsageTokens
	reasoning := false
	var pending *pendingToolCall
	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}
		rawCopy := append([]byte(nil), raw...)
		if opts.IncludeRawChunks {
			var decoded map[string]any
			_ = json.Unmarshal(rawCopy, &decoded)
			parts <- StreamRaw{Raw: rawCopy, Decoded: decoded}
		}
		var typ cohereStreamType
		if err := json.Unmarshal(rawCopy, &typ); err != nil || typ.Type == "" {
			finish = FinishReason{Unified: "error", Raw: ""}
			parts <- StreamError{Err: InvalidResponseDataError{Message: errString(err), Data: string(rawCopy)}}
			return true
		}
		switch typ.Type {
		case "content-start":
			var ch cohereStreamContentChunk
			if !decodeChunk(rawCopy, &ch, parts, &finish) {
				return true
			}
			id := strconv.Itoa(ch.Index)
			if ch.Delta.Message.Content.Type == "thinking" {
				reasoning = true
				parts <- StreamReasoningStart{ID: id}
			} else {
				parts <- StreamTextStart{ID: id}
			}
		case "content-delta":
			var ch cohereStreamContentChunk
			if !decodeChunk(rawCopy, &ch, parts, &finish) {
				return true
			}
			id := strconv.Itoa(ch.Index)
			if ch.Delta.Message.Content.Thinking != nil {
				parts <- StreamReasoningDelta{ID: id, Text: *ch.Delta.Message.Content.Thinking}
			} else if ch.Delta.Message.Content.Text != nil {
				parts <- StreamTextDelta{ID: id, Text: *ch.Delta.Message.Content.Text}
			}
		case "content-end":
			var ch struct {
				Index int `json:"index"`
			}
			if !decodeChunk(rawCopy, &ch, parts, &finish) {
				return true
			}
			id := strconv.Itoa(ch.Index)
			if reasoning {
				parts <- StreamReasoningEnd{ID: id}
				reasoning = false
			} else {
				parts <- StreamTextEnd{ID: id}
			}
		case "tool-call-start":
			var ch cohereStreamToolCallChunk
			if !decodeChunk(rawCopy, &ch, parts, &finish) {
				return true
			}
			tc := ch.Delta.Message.ToolCalls
			pending = &pendingToolCall{id: tc.ID, name: tc.Function.Name}
			pending.args.WriteString(tc.Function.Arguments)
			parts <- StreamToolInputStart{ID: tc.ID, ToolName: tc.Function.Name}
			if tc.Function.Arguments != "" {
				parts <- StreamToolInputDelta{ID: tc.ID, Delta: tc.Function.Arguments}
			}
		case "tool-call-delta":
			var ch cohereStreamToolCallChunk
			if !decodeChunk(rawCopy, &ch, parts, &finish) {
				return true
			}
			if pending != nil && !pending.finished {
				delta := ch.Delta.Message.ToolCalls.Function.Arguments
				pending.args.WriteString(delta)
				parts <- StreamToolInputDelta{ID: pending.id, Delta: delta}
			}
		case "tool-call-end":
			if pending != nil && !pending.finished {
				input := normalizedJSONInput(pending.args.String())
				parts <- StreamToolInputEnd{ID: pending.id}
				parts <- StreamToolCall{ToolCallContent: ToolCallContent{ToolCallID: pending.id, ToolName: pending.name, Input: json.RawMessage(input)}}
				pending.finished = true
				pending = nil
			}
		case "message-start":
			var ch cohereStreamMessageStart
			if !decodeChunk(rawCopy, &ch, parts, &finish) {
				return true
			}
			sresp.ID = ch.ID
			parts <- StreamResponseMetadata{ID: ch.ID}
		case "message-end":
			var ch cohereStreamMessageEnd
			if !decodeChunk(rawCopy, &ch, parts, &finish) {
				return true
			}
			finish = FinishReason{Unified: mapCohereFinishReason(ch.Delta.FinishReason), Raw: ch.Delta.FinishReason}
			usage = ch.Delta.Usage.Tokens
		}
		return true
	})
	parts <- StreamFinish{FinishReason: finish, Usage: convertCohereUsage(usage)}
}

func decodeChunk(raw []byte, v any, parts chan<- StreamPart, finish *FinishReason) bool {
	if err := json.Unmarshal(raw, v); err != nil {
		*finish = FinishReason{Unified: "error", Raw: ""}
		parts <- StreamError{Err: InvalidResponseDataError{Message: err.Error(), Data: string(raw)}}
		return false
	}
	return true
}
func errString(err error) string {
	if err == nil {
		return "missing chunk type"
	}
	return err.Error()
}
func normalizedJSONInput(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		s = "{}"
	}
	if !json.Valid([]byte(s)) {
		return "{}"
	}
	var b bytes.Buffer
	if err := json.Compact(&b, []byte(s)); err != nil {
		return "{}"
	}
	return b.String()
}
