package openaicompatible

import (
	"context"
	"encoding/json"
	"sort"
	"strings"
	"time"
)

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
	Content          json.RawMessage      `json:"content"`
	ReasoningContent string               `json:"reasoning_content"`
	Reasoning        string               `json:"reasoning"`
	ToolCalls        []chatStreamToolCall `json:"tool_calls"`
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
	Finished   bool
	ThoughtSig *string
	// Index is the wire tool-call index this slot belongs to; split
	// slots (index reused for a new id) share their parent's Index.
	Index int
	// Seq is the creation sequence, tie-breaking slots that share Index.
	Seq int
}

func (m *openAICompatibleChatLanguageModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildChatRequest(opts.GenerateOptions)
	if err != nil {
		return nil, err
	}
	body := config.Body
	body["stream"] = true
	if m.provider.includeUsage {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if m.provider.transformRequestBody != nil {
		body = m.provider.transformRequestBody(body)
		if body == nil {
			body = map[string]any{}
		}
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	reqBody := append([]byte(nil), bodyBytes...)
	resp, err := m.provider.executeStream(ctx, endpointChatCompletions, bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}

	parts := make(chan StreamPart)
	sresp := &StreamResponse{Headers: cloneHeader(resp.Headers)}
	result := &StreamResult{Stream: parts, Parts: parts, Request: RequestMetadata{Body: reqBody}, Response: sresp}
	go m.runChatStream(ctx, resp, parts, sresp, config.Warnings, opts)
	return result, nil
}

func (m *openAICompatibleChatLanguageModel) runChatStream(ctx context.Context, resp *httpStreamResponse, parts chan<- StreamPart, sresp *StreamResponse, warnings []Warning, opts StreamOptions) {
	defer close(parts)
	defer resp.Body.Close()

	parts <- StreamStart{Warnings: warnings}

	var streamExtractor StreamMetadataExtractor
	if m.provider.metadataExtractor != nil {
		streamExtractor = m.provider.metadataExtractor.CreateStreamExtractor()
	}

	metadataKey := metadataKeyForProviderOptions(m.provider.name, cloneProviderOptions(opts.ProviderOptions))

	state := &chatStreamState{
		finishReason:    FinishReason{Unified: "other", Raw: ""},
		toolCalls:       map[int]*chatStreamToolState{},
		providerOptions: cloneProviderOptions(opts.ProviderOptions),
		usageShape:      new(chatUsageShape),
		metadataKey:     metadataKey,
	}
	var latestRawUsage json.RawMessage

	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}

		rawCopy := append([]byte(nil), raw...)

		if opts.IncludeRawChunks {
			var decodedForRaw map[string]any
			decodedMapErr := json.Unmarshal(rawCopy, &decodedForRaw)
			var decodedCopy map[string]any
			if decodedMapErr == nil {
				decodedCopy = cloneDecodedMap(decodedForRaw)
			}
			parts <- StreamRaw{Raw: append([]byte(nil), raw...), Decoded: decodedCopy}
		}

		var decodedMap map[string]any
		_ = json.Unmarshal(rawCopy, &decodedMap)

		var chunk chatStreamChunk
		if err := json.Unmarshal(rawCopy, &chunk); err != nil {
			if streamExtractor != nil {
				streamExtractor.ProcessChunk(rawCopy, decodedMap)
			}
			parts <- StreamError{Err: InvalidResponseDataError{Message: err.Error(), Data: string(rawCopy)}}
			parts <- StreamFinish{FinishReason: errorFinishReason(), Usage: state.buildUsage(latestRawUsage, m.provider), ProviderMetadata: state.buildMetadata(m.provider.name, m.provider.metadataExtractor, streamExtractor, m.provider.convertUsage)}
			state.fatal = true
			return false
		}

		if streamExtractor != nil {
			streamExtractor.ProcessChunk(rawCopy, cloneDecodedMap(decodedMap))
		}

		if chunk.Error != nil {
			parts <- StreamError{Err: APIError{Message: chunk.Error.Message, Type: chunk.Error.Type, Param: chunk.Error.Param, Code: chunk.Error.Code}}
			parts <- StreamFinish{FinishReason: errorFinishReason(), Usage: state.buildUsage(latestRawUsage, m.provider), ProviderMetadata: state.buildMetadata(m.provider.name, m.provider.metadataExtractor, streamExtractor, m.provider.convertUsage)}
			state.fatal = true
			return false
		}

		if !state.metadataSent {
			state.metadataSent = true
			id := chunk.ID
			model := chunk.Model
			sresp.ID = id
			sresp.ModelID = model
			if chunk.Created != nil {
				ts := time.Unix(*chunk.Created, 0)
				sresp.Timestamp = &ts
				parts <- StreamResponseMetadata{ID: id, ModelID: model, Timestamp: &ts}
			} else {
				parts <- StreamResponseMetadata{ID: id, ModelID: model}
			}
		}

		if len(chunk.Usage) > 0 && string(chunk.Usage) != "null" {
			latestRawUsage = cloneRawMessage(chunk.Usage)
			var usageShape chatUsageShape
			if err := json.Unmarshal(chunk.Usage, &usageShape); err == nil {
				*state.usageShape = usageShape
			}
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				state.finishReason = finishReasonFromOpenAI(*choice.FinishReason)
				state.finishSeen = true
			}
			m.processChatStreamDelta(parts, state, choice.Delta)
		}
		return true
	})

	if state.fatal {
		return
	}

	state.flushStreamState(parts)
	if !state.finishSeen && state.finishReason.Unified != "error" {
		// The stream ended without a finish reason: it was truncated,
		// so it must not be reported as a successful finish.
		parts <- StreamError{Err: InvalidResponseDataError{Message: "Stream ended without a finish reason."}}
		state.finishReason = errorFinishReason()
	}
	finalMeta := state.buildMetadata(m.provider.name, m.provider.metadataExtractor, streamExtractor, m.provider.convertUsage)
	parts <- StreamFinish{
		FinishReason:     state.finishReason,
		Usage:            state.buildUsage(latestRawUsage, m.provider),
		ProviderMetadata: finalMeta,
	}
	sresp.ProviderMetadata = finalMeta
}

type chatStreamState struct {
	reasoningActive bool
	textStarted     bool
	finishReason    FinishReason
	finishSeen      bool
	metadataSent    bool
	toolCalls       map[int]*chatStreamToolState
	toolCallCount   int
	toolSeqNext     int
	latestToolIndex int
	hasLatestTool   bool
	providerOptions ProviderOptions
	usageShape      *chatUsageShape
	metadataKey     string
	fatal           bool
}

func (s *chatStreamState) buildUsage(latestRaw json.RawMessage, provider *openAICompatibleProvider) Usage {
	u := s.usageShape.toPublic()
	if u != nil {
		u.Raw = cloneRawMessage(latestRaw)
	}
	converted := defaultChatUsage(u)
	if u != nil && provider.convertUsage != nil {
		converted = provider.convertUsage(*u)
	}
	return converted
}

func (s *chatStreamState) buildMetadata(name string, extractor MetadataExtractor, streamExtractor StreamMetadataExtractor, convertUsage func(OpenAICompatibleTokenUsage) Usage) ProviderMetadata {
	u := s.usageShape.toPublic()
	metadata := predictionTokenMetadata(name, s.providerOptions, u)
	if streamExtractor != nil {
		build := streamExtractor.BuildMetadata()
		mergeProviderMetadata(metadata, build)
	}
	return metadata
}

// sortedToolKeys returns tool slots in deterministic emission order:
// numeric wire index first, creation sequence breaking ties between a
// slot and its index-reuse splits. Map iteration order is random, so
// flushing must never range s.toolCalls directly.
func (s *chatStreamState) sortedToolKeys() []int {
	keys := make([]int, 0, len(s.toolCalls))
	for k := range s.toolCalls {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := s.toolCalls[keys[i]], s.toolCalls[keys[j]]
		if a.Index != b.Index {
			return a.Index < b.Index
		}
		return a.Seq < b.Seq
	})
	return keys
}

func (s *chatStreamState) flushStreamState(parts chan<- StreamPart) {
	if s.reasoningActive {
		parts <- StreamReasoningEnd{ID: "reasoning-0"}
		s.reasoningActive = false
	}
	if s.textStarted {
		parts <- StreamTextEnd{ID: "txt-0"}
		s.textStarted = false
	}
	for _, k := range s.sortedToolKeys() {
		tool := s.toolCalls[k]
		if tool.Finished {
			continue
		}
		tool.Finished = true
		parts <- StreamToolInputStart{ID: tool.ID, ToolName: tool.Name}
		input := tool.Args.String()
		if input == "" || json.Valid([]byte(input)) {
			if input == "" {
				input = "{}"
			}
			if tool.Args.Len() > 0 {
				parts <- StreamToolInputDelta{ID: tool.ID, Delta: tool.Args.String()}
			}
			parts <- StreamToolInputEnd{ID: tool.ID}
			toolMetadata := ProviderMetadata(nil)
			if tool.ThoughtSig != nil {
				toolMetadata = ProviderMetadata{s.metadataKey: map[string]any{"thoughtSignature": *tool.ThoughtSig}}
			}
			parts <- StreamToolCall{ToolCallContent: ToolCallContent{
				ToolCallID:       tool.ID,
				ToolName:         tool.Name,
				Input:            json.RawMessage(input),
				ProviderMetadata: toolMetadata,
			}}
			continue
		}
		// The accumulation never became valid JSON (truncated input or
		// trailing garbage): surface an error instead of executing it.
		parts <- StreamToolInputEnd{ID: tool.ID}
		parts <- StreamError{Err: InvalidResponseDataError{Message: "Invalid tool call arguments JSON.", Data: input}}
		s.finishReason = errorFinishReason()
	}
}

func (m *openAICompatibleChatLanguageModel) processChatStreamDelta(parts chan<- StreamPart, state *chatStreamState, delta chatStreamDelta) {
	reasoning := delta.ReasoningContent
	if reasoning == "" {
		reasoning = delta.Reasoning
	}
	deltaText, deltaThinking := parseChatContentValue(delta.Content)
	reasoning += deltaThinking

	if reasoning != "" {
		if !state.reasoningActive {
			state.reasoningActive = true
			parts <- StreamReasoningStart{ID: "reasoning-0"}
		}
		parts <- StreamReasoningDelta{ID: "reasoning-0", Text: reasoning}
	}

	if deltaText != "" {
		if state.reasoningActive {
			parts <- StreamReasoningEnd{ID: "reasoning-0"}
			state.reasoningActive = false
		}
		if !state.textStarted {
			state.textStarted = true
			parts <- StreamTextStart{ID: "txt-0"}
		}
		parts <- StreamTextDelta{ID: "txt-0", Text: deltaText}
	}

	if len(delta.ToolCalls) > 0 {
		if state.reasoningActive {
			parts <- StreamReasoningEnd{ID: "reasoning-0"}
			state.reasoningActive = false
		}
	}

	for _, tc := range delta.ToolCalls {
		var id string
		if tc.ID != nil {
			id = *tc.ID
		}
		index := state.resolveToolCallIndex(tc.Index, id)
		tool, exists := state.toolCalls[index]
		if !exists {
			if tc.ID == nil || *tc.ID == "" {
				parts <- StreamError{Err: InvalidResponseDataError{Message: "Expected 'id' to be a string."}}
				state.finishReason = errorFinishReason()
				return
			}
			if tc.Function.Name == nil || *tc.Function.Name == "" {
				parts <- StreamError{Err: InvalidResponseDataError{Message: "Expected 'function.name' to be a string."}}
				state.finishReason = errorFinishReason()
				return
			}
			toolID := *tc.ID
			toolName := *tc.Function.Name
			wireIndex := index
			if tc.Index != nil {
				wireIndex = *tc.Index
			}
			state.toolCalls[index] = &chatStreamToolState{ID: toolID, Name: toolName, Index: wireIndex, Seq: state.toolSeqNext}
			state.toolSeqNext++
			tool = state.toolCalls[index]
			if tc.ExtraContent.Google.ThoughtSignature != nil {
				sig := *tc.ExtraContent.Google.ThoughtSignature
				tool.ThoughtSig = &sig
			}
			if tc.Function.Arguments != nil {
				tool.Args.WriteString(*tc.Function.Arguments)
			}
			// Buffer only: a prefix that parses as JSON today may still
			// be followed by more fragments (or garbage), so the
			// ToolCall is finalized only at flush. Emitting mid-stream
			// would execute a truncated input while later fragments
			// are dropped.
			continue
		}
		if tc.Function.Arguments != nil {
			tool.Args.WriteString(*tc.Function.Arguments)
		}
	}
}

func (s *chatStreamState) resolveToolCallIndex(index *int, id string) int {
	if index != nil {
		if tool, ok := s.toolCalls[*index]; ok && id != "" && tool.ID != "" && tool.ID != id {
			// The index was reused for a different tool call id
			// (provider fallback): split into a fresh slot instead of
			// merging into the existing tool's arguments.
			slot := s.toolCallCount
			for {
				if _, taken := s.toolCalls[slot]; !taken {
					break
				}
				slot++
			}
			s.toolCallCount = slot + 1
			s.latestToolIndex = slot
			s.hasLatestTool = true
			return slot
		}
		s.latestToolIndex = *index
		s.hasLatestTool = true
		return *index
	}
	// A missing index sticks to the latest open slot (provider
	// fallback that omits per-fragment indices).
	if s.hasLatestTool {
		return s.latestToolIndex
	}
	current := s.toolCallCount
	s.toolCallCount++
	s.latestToolIndex = current
	s.hasLatestTool = true
	return current
}

func cloneDecodedMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
