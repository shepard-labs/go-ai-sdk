package openaicompatible

import (
	"context"
	"encoding/json"
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
	Content          string               `json:"content"`
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
			}
			m.processChatStreamDelta(parts, state, choice.Delta)
		}
		return true
	})

	if state.fatal {
		return
	}

	state.flushStreamState(parts)
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
	metadataSent    bool
	toolCalls       map[int]*chatStreamToolState
	toolCallCount   int
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

func (s *chatStreamState) flushStreamState(parts chan<- StreamPart) {
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
			toolMetadata := ProviderMetadata(nil)
			if tool.ThoughtSig != nil {
				toolMetadata = ProviderMetadata{s.metadataKey: map[string]any{"thoughtSignature": *tool.ThoughtSig}}
			}
			parts <- StreamToolCall{ToolCallContent: ToolCallContent{
				ToolCallID:       tool.ID,
				ToolName:         tool.Name,
				Input:            json.RawMessage(inputJSON),
				ProviderMetadata: toolMetadata,
			}}
		}
	}
}

func (m *openAICompatibleChatLanguageModel) processChatStreamDelta(parts chan<- StreamPart, state *chatStreamState, delta chatStreamDelta) {
	reasoning := delta.ReasoningContent
	if reasoning == "" {
		reasoning = delta.Reasoning
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
		index := state.resolveToolCallIndex(tc.Index)
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
			state.toolCalls[index] = &chatStreamToolState{ID: toolID, Name: toolName}
			tool = state.toolCalls[index]
			if tc.ExtraContent.Google.ThoughtSignature != nil {
				sig := *tc.ExtraContent.Google.ThoughtSignature
				tool.ThoughtSig = &sig
			}
			if tc.Function.Arguments != nil {
				tool.Args.WriteString(*tc.Function.Arguments)
			}
			parts <- StreamToolInputStart{ID: toolID, ToolName: toolName}
			if tool.Args.Len() > 0 {
				parts <- StreamToolInputDelta{ID: toolID, Delta: tool.Args.String()}
			}
			if json.Valid([]byte(tool.Args.String())) {
				maybeEmitToolCall(parts, state, tool)
			}
			continue
		}
		if tool.Finished {
			continue
		}
		if tc.Function.Arguments != nil {
			tool.Args.WriteString(*tc.Function.Arguments)
		}
		deltaText := ""
		if tc.Function.Arguments != nil {
			deltaText = *tc.Function.Arguments
		}
		parts <- StreamToolInputDelta{ID: tool.ID, Delta: deltaText}
		if json.Valid([]byte(tool.Args.String())) {
			maybeEmitToolCall(parts, state, tool)
		}
	}
}

func maybeEmitToolCall(parts chan<- StreamPart, state *chatStreamState, tool *chatStreamToolState) {
	inputJSON := tool.Args.String()
	if inputJSON == "" {
		inputJSON = "{}"
	}
	parts <- StreamToolInputEnd{ID: tool.ID}
	toolMetadata := ProviderMetadata(nil)
	if tool.ThoughtSig != nil {
		toolMetadata = ProviderMetadata{state.metadataKey: map[string]any{"thoughtSignature": *tool.ThoughtSig}}
	}
	parts <- StreamToolCall{ToolCallContent: ToolCallContent{
		ToolCallID:       tool.ID,
		ToolName:         tool.Name,
		Input:            json.RawMessage(inputJSON),
		ProviderMetadata: toolMetadata,
	}}
	tool.Finished = true
}

func (s *chatStreamState) resolveToolCallIndex(index *int) int {
	if index != nil {
		return *index
	}
	current := s.toolCallCount
	s.toolCallCount++
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
