package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
)

// runChatStream drives an OpenAI chat-completion SSE stream, translating
// deltas into StreamPart events. It is run on its own goroutine by DoStream.
func (m *openaiChatLanguageModel) runChatStream(
	ctx context.Context,
	resp *httpStreamResponse,
	parts chan<- StreamPart,
	sresp *StreamResponse,
	warnings []Warning,
	opts StreamOptions,
) {
	defer close(parts)
	defer resp.Body.Close()
	parts <- StreamStart{Warnings: warnings}
	headers := resp.Headers.Clone()
	sresp.Headers = headers

	state := newChatStreamState()
	state.providerOptions = cloneProviderOptions(opts.ProviderOptions)
	metadataKey := providerMetadataKey("openai.chat", state.providerOptions)
	var latestRawUsage json.RawMessage
	var usageShape chatUsageShape
	metadataSent := false

	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}
		rawCopy := append([]byte(nil), raw...)
		if opts.IncludeRawChunks {
			parts <- StreamRaw{Raw: rawCopy}
		}
		var chunk chatStreamChunk
		if err := json.Unmarshal(rawCopy, &chunk); err != nil {
			parts <- StreamError{Err: InvalidResponseDataError{Message: err.Error(), Data: string(rawCopy)}}
			state.finishReason = errorFinishReason()
			state.fatal = true
			return false
		}
		if chunk.Error != nil {
			// Per spec: a chunk-level error (returned with HTTP 200)
			// should surface as an APICallError with a derived status
			// code from the error type. The HTTP status itself is 200
			// (the stream started successfully), so the spec's status
			// derivation rule is what consumers should see.
			status, _ := openAIErrorTypeStatusCode(chunk.Error.Type)
			apiErr := &APICallError{
				Message: chunk.Error.Message,
				Type:    chunk.Error.Type,
				Param:   chunk.Error.Param,
				Code:    chunk.Error.Code,
				Status:  status,
			}
			parts <- StreamError{Err: apiErr}
			state.finishReason = errorFinishReason()
			state.fatal = true
			return false
		}
		if id := chunk.ID; id != "" {
			sresp.ID = id
		}
		if model := chunk.Model; model != "" {
			sresp.ModelID = model
		}
		if !metadataSent && (sresp.ID != "" || sresp.ModelID != "") {
			metadataSent = true
			if chunk.Created != nil {
				ts := time.Unix(*chunk.Created, 0)
				sresp.Timestamp = &ts
				parts <- StreamResponseMetadata{ID: sresp.ID, ModelID: sresp.ModelID, Timestamp: &ts}
			} else {
				parts <- StreamResponseMetadata{ID: sresp.ID, ModelID: sresp.ModelID}
			}
		}
		if len(chunk.Usage) > 0 && string(chunk.Usage) != "null" {
			latestRawUsage = cloneRawMessage(chunk.Usage)
			_ = json.Unmarshal(chunk.Usage, &usageShape)
		}
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				state.finishReason = chatFinishReasonFromString(*choice.FinishReason)
			}
			m.processChatStreamDelta(parts, state, choice.Delta)
		}
		return true
	})
	if state.fatal {
		return
	}
	state.flushChatStreamState(parts)
	finalUsage := buildChatUsage(latestRawUsage, usageShape)
	// Compose provider metadata from prediction-token details.
	usageMeta := buildChatProviderMetadataFromUsage(latestRawUsage)
	otherMeta := buildChatProviderMetadata("openai.chat", metadataKey, state.providerOptions, usageShape)
	finalMeta := mergeProviderMetadata(usageMeta, otherMeta)
	parts <- StreamFinish{FinishReason: state.finishReason, Usage: finalUsage, ProviderMetadata: finalMeta}
	sresp.ProviderMetadata = finalMeta
}

// chatStreamChunk mirrors the OpenAI chat-completion SSE chunk.
type chatStreamChunk struct {
	ID      string                `json:"id"`
	Created *int64                `json:"created"`
	Model   string                `json:"model"`
	Choices []chatStreamChoice    `json:"choices"`
	Usage   json.RawMessage       `json:"usage"`
	Error   *chatStreamErrorShape `json:"error"`
}

type chatStreamChoice struct {
	Delta        chatStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type chatStreamDelta struct {
	Content          string               `json:"content"`
	ReasoningContent string               `json:"reasoning_content"`
	Reasoning        string               `json:"reasoning"`
	ToolCalls        []chatStreamToolCall `json:"tool_calls"`
	Annotations      []map[string]any     `json:"annotations"`
}

type chatStreamToolCall struct {
	ID           *string            `json:"id"`
	Index        *int               `json:"index"`
	Function     chatStreamFunction `json:"function"`
	ExtraContent struct {
		Google struct {
			ThoughtSignature *string `json:"thought_signature"`
		} `json:"google"`
	} `json:"extra_content"`
}

type chatStreamFunction struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

type chatStreamErrorShape struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   any    `json:"param"`
	Code    any    `json:"code"`
}

type chatStreamToolState struct {
	ID         string
	Name       string
	Args       strings.Builder
	Started    bool
	Finished   bool
	ThoughtSig *string
}

type chatStreamState struct {
	reasoningActive bool
	textStarted     bool
	finishReason    FinishReason
	toolCalls       map[int]*chatStreamToolState
	pendingArgs     map[int]string
	toolCallCount   int
	providerOptions ProviderOptions
	fatal           bool
}

func newChatStreamState() *chatStreamState {
	return &chatStreamState{
		toolCalls:   map[int]*chatStreamToolState{},
		pendingArgs: map[int]string{},
	}
}

// chatUsageShape mirrors the OpenAI chat-completion usage.
type chatUsageShape struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func errorFinishReason() FinishReason {
	return FinishReason{Unified: "other", Raw: "error"}
}

func chatFinishReasonFromString(reason string) FinishReason {
	switch reason {
	case "stop":
		return FinishReason{Unified: "stop", Raw: reason}
	case "length":
		return FinishReason{Unified: "length", Raw: reason}
	case "tool_calls":
		return FinishReason{Unified: "tool-calls", Raw: reason}
	case "function_call":
		return FinishReason{Unified: "tool-calls", Raw: reason}
	case "content_filter":
		return FinishReason{Unified: "content-filter", Raw: reason}
	default:
		return FinishReason{Unified: "other", Raw: reason}
	}
}

func (m *openaiChatLanguageModel) processChatStreamDelta(parts chan<- StreamPart, state *chatStreamState, delta chatStreamDelta) {
	reasoning := delta.ReasoningContent
	if reasoning == "" {
		reasoning = delta.Reasoning
	}
	for _, a := range delta.Annotations {
		typ, _ := a["type"].(string)
		if typ != "url_citation" {
			continue
		}
		inner, _ := a["url_citation"].(map[string]any)
		if inner == nil {
			inner = a
		}
		sc := SourceContent{
			SourceType: typ,
			ID:         stringValue(inner["uuid"]),
			URL:        stringValue(inner["url"]),
			Title:      stringValue(inner["title"]),
		}
		parts <- sc
	}
	if reasoning != "" {
		if !state.reasoningActive {
			state.reasoningActive = true
			parts <- StreamReasoningStart{ID: "reasoning-0"}
		}
		parts <- StreamReasoningDelta{ID: "reasoning-0", Text: reasoning}
	}
	if delta.Content != "" {
		if state.reasoningActive {
			parts <- StreamReasoningEnd{ID: "reasoning-0"}
			state.reasoningActive = false
		}
		if !state.textStarted {
			state.textStarted = true
			parts <- StreamTextStart{ID: "txt-0"}
		}
		parts <- StreamTextDelta{ID: "txt-0", Text: delta.Content}
	}
	if len(delta.ToolCalls) > 0 {
		if state.reasoningActive {
			parts <- StreamReasoningEnd{ID: "reasoning-0"}
			state.reasoningActive = false
		}
	}
	for _, tc := range delta.ToolCalls {
		index := resolveToolCallIndex(state, tc.Index)
		tool, exists := state.toolCalls[index]
		if !exists {
			if tc.ID == nil || *tc.ID == "" {
				// If arguments arrive before the tool id, buffer them
				// until the next chunk supplies the id.
				if tc.Function.Arguments != nil {
					state.pendingArgs[index] += *tc.Function.Arguments
				}
				continue
			}
			if tc.Function.Name == nil || *tc.Function.Name == "" {
				// Binding decision #14: function.name may arrive in a
				// later chunk than function.arguments. Buffer the
				// arguments until the name arrives.
				toolID := *tc.ID
				pending := state.pendingArgs[index]
				if tc.Function.Arguments != nil {
					pending += *tc.Function.Arguments
				}
				state.pendingArgs[index] = pending
				state.toolCalls[index] = &chatStreamToolState{ID: toolID}
				continue
			}
			toolID := *tc.ID
			toolName := *tc.Function.Name
			tool = &chatStreamToolState{ID: toolID, Name: toolName}
			state.toolCalls[index] = tool
			if tc.ExtraContent.Google.ThoughtSignature != nil {
				sig := *tc.ExtraContent.Google.ThoughtSignature
				tool.ThoughtSig = &sig
			}
			// Hydrate any buffered arguments.
			if pending, ok := state.pendingArgs[index]; ok && pending != "" {
				tool.Args.WriteString(pending)
				delete(state.pendingArgs, index)
			}
			if tc.Function.Arguments != nil {
				tool.Args.WriteString(*tc.Function.Arguments)
			}
			parts <- StreamToolInputStart{ID: toolID, ToolName: toolName}
			if tool.Args.Len() > 0 {
				parts <- StreamToolInputDelta{ID: toolID, Delta: tool.Args.String()}
			}
			if json.Valid([]byte(tool.Args.String())) {
				emitChatToolCall(parts, tool)
			}
			continue
		}
		if tool.Finished {
			continue
		}
		// Tool exists. If name not yet known, hydrate it now.
		if tool.Name == "" && tc.Function.Name != nil && *tc.Function.Name != "" {
			tool.Name = *tc.Function.Name
			if pending, ok := state.pendingArgs[index]; ok && pending != "" {
				tool.Args.WriteString(pending)
				delete(state.pendingArgs, index)
			}
			if tc.Function.Arguments != nil {
				tool.Args.WriteString(*tc.Function.Arguments)
			}
			parts <- StreamToolInputStart{ID: tool.ID, ToolName: tool.Name}
			if tool.Args.Len() > 0 {
				parts <- StreamToolInputDelta{ID: tool.ID, Delta: tool.Args.String()}
			}
			if json.Valid([]byte(tool.Args.String())) {
				emitChatToolCall(parts, tool)
			}
			continue
		}
		if tc.Function.Arguments != nil {
			tool.Args.WriteString(*tc.Function.Arguments)
		}
		deltaText := ""
		if tc.Function.Arguments != nil {
			deltaText = *tc.Function.Arguments
		}
		if tool.Started {
			parts <- StreamToolInputDelta{ID: tool.ID, Delta: deltaText}
		} else if tool.Name != "" && deltaText != "" {
			// Late argument with name known but start not yet emitted.
			parts <- StreamToolInputStart{ID: tool.ID, ToolName: tool.Name}
			parts <- StreamToolInputDelta{ID: tool.ID, Delta: deltaText}
			tool.Started = true
		}
		if json.Valid([]byte(tool.Args.String())) {
			emitChatToolCall(parts, tool)
		}
	}
}

func resolveToolCallIndex(state *chatStreamState, index *int) int {
	if index != nil {
		return *index
	}
	current := state.toolCallCount
	state.toolCallCount++
	return current
}

func emitChatToolCall(parts chan<- StreamPart, tool *chatStreamToolState) {
	inputJSON := tool.Args.String()
	if inputJSON == "" {
		inputJSON = "{}"
	}
	if !tool.Started && tool.Name != "" {
		parts <- StreamToolInputStart{ID: tool.ID, ToolName: tool.Name}
		tool.Started = true
	}
	parts <- StreamToolInputEnd{ID: tool.ID}
	parts <- StreamToolCall{
		ToolCallContent: ToolCallContent{
			ToolCallContentEmbed: ToolCallContentEmbed{
				ToolCallID: tool.ID,
				ToolName:   tool.Name,
				Input:      json.RawMessage(inputJSON),
			},
		},
	}
	tool.Finished = true
}

func (s *chatStreamState) flushChatStreamState(parts chan<- StreamPart) {
	if s.reasoningActive {
		parts <- StreamReasoningEnd{ID: "reasoning-0"}
		s.reasoningActive = false
	}
	if s.textStarted {
		parts <- StreamTextEnd{ID: "txt-0"}
		s.textStarted = false
	}
	for _, tool := range s.toolCalls {
		if !tool.Finished {
			inputJSON := tool.Args.String()
			if !json.Valid([]byte(inputJSON)) {
				inputJSON = "{}"
			}
			parts <- StreamToolInputEnd{ID: tool.ID}
			parts <- StreamToolCall{
				ToolCallContent: ToolCallContent{
					ToolCallContentEmbed: ToolCallContentEmbed{
						ToolCallID: tool.ID,
						ToolName:   tool.Name,
						Input:      json.RawMessage(inputJSON),
					},
				},
			}
		}
	}
}

func buildChatUsage(raw json.RawMessage, shape chatUsageShape) Usage {
	if raw == nil || string(raw) == "null" {
		return Usage{}
	}
	usage := Usage{Raw: cloneRawMessage(raw)}
	if shape.PromptTokens > 0 {
		tokens := shape.PromptTokens
		usage.InputTokens.Total = &tokens
	}
	if shape.CompletionTokens > 0 {
		tokens := shape.CompletionTokens
		usage.OutputTokens.Total = &tokens
	}
	if shape.TotalTokens > 0 {
		tokens := shape.TotalTokens
		// Add to input if not already populated; otherwise the spec
		// for chat puts the unified total in OutputTokens.Total.
		if usage.OutputTokens.Total == nil {
			usage.OutputTokens.Total = &tokens
		}
	}
	return usage
}

func buildChatProviderMetadata(name, metadataKey string, opts ProviderOptions, shape chatUsageShape) ProviderMetadata {
	pm := ProviderMetadata{}
	_ = name
	_ = opts
	_ = shape
	_ = metadataKey
	return pm
}

// buildChatProviderMetadataFromUsage extracts
// acceptedPredictionTokens and rejectedPredictionTokens from the raw
// usage JSON (which carries completion_tokens_details).
func buildChatProviderMetadataFromUsage(raw json.RawMessage) ProviderMetadata {
	if len(raw) == 0 {
		return nil
	}
	var usage map[string]any
	if err := json.Unmarshal(raw, &usage); err != nil {
		return nil
	}
	cd, ok := usage["completion_tokens_details"].(map[string]any)
	if !ok {
		return nil
	}
	om := map[string]any{}
	if v, ok := cd["accepted_prediction_tokens"].(float64); ok {
		om["acceptedPredictionTokens"] = int(v)
	}
	if v, ok := cd["rejected_prediction_tokens"].(float64); ok {
		om["rejectedPredictionTokens"] = int(v)
	}
	if len(om) == 0 {
		return nil
	}
	return ProviderMetadata{"openai": om}
}

// mergeProviderMetadata merges two ProviderMetadata maps. Values from
// the second map override the first.
func mergeProviderMetadata(a, b ProviderMetadata) ProviderMetadata {
	if len(a) == 0 {
		return b
	}
	if len(b) == 0 {
		return a
	}
	out := ProviderMetadata{}
	for k, v := range a {
		out[k] = v
	}
	for k, v := range b {
		out[k] = v
	}
	return out
}

func cloneRawMessage(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	out := make(json.RawMessage, len(raw))
	copy(out, raw)
	return out
}

func providerMetadataKey(name string, opts ProviderOptions) string {
	_ = opts
	return name
}

func processSSEStream(body io.Reader, onChunk func([]byte) bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 1<<16), 1<<24)
	var buf []byte
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if len(buf) > 0 {
				if !onChunk(buf) {
					return
				}
				buf = nil
			}
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			return
		}
		buf = append(buf, payload...)
	}
	if len(buf) > 0 {
		_ = onChunk(buf)
	}
	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		_ = err
	}
}

// keep imports stable
var _ = http.MethodPost
