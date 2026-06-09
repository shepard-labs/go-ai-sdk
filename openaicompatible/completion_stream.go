package openaicompatible

import (
	"context"
	"encoding/json"
	"time"
)

// completionStreamChunk is the SSE chunk shape for the /completions endpoint.
type completionStreamChunk struct {
	ID      string                      `json:"id"`
	Created *int64                      `json:"created"`
	Model   string                      `json:"model"`
	Choices []completionStreamChoice    `json:"choices"`
	Usage   json.RawMessage             `json:"usage"`
	Error   *completionStreamErrorShape `json:"error"`
}

type completionStreamChoice struct {
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
	Index        int     `json:"index"`
}

// completionStreamErrorShape mirrors the error field in streaming completion
// chunks; identical structure to the chat stream error shape.
type completionStreamErrorShape struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   any    `json:"param"`
	Code    any    `json:"code"`
}

// DoStream implements LanguageModel for the completion model.
func (m *openAICompatibleCompletionLanguageModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildCompletionRequest(opts.GenerateOptions)
	if err != nil {
		return nil, err
	}
	body := config.Body
	body["stream"] = true
	if m.provider.includeUsage {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	// Do NOT apply TransformRequestBody for completion requests (per spec).
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	reqBody := append([]byte(nil), bodyBytes...)
	resp, err := m.provider.executeStream(ctx, endpointCompletions, bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}

	parts := make(chan StreamPart)
	sresp := &StreamResponse{Headers: cloneHeader(resp.Headers)}
	result := &StreamResult{Stream: parts, Parts: parts, Request: RequestMetadata{Body: reqBody}, Response: sresp}
	go m.runCompletionStream(ctx, resp, parts, sresp, config.Warnings, opts)
	return result, nil
}

// runCompletionStream is the goroutine that drives the SSE loop for completion
// streams. It mirrors the structure of runChatStream but is completion-specific.
func (m *openAICompatibleCompletionLanguageModel) runCompletionStream(
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

	state := &completionStreamState{
		finishReason: FinishReason{Unified: "other", Raw: ""},
	}
	var latestRawUsage json.RawMessage

	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}

		rawCopy := append([]byte(nil), raw...)

		// Emit StreamRaw before any decoded events when requested.
		if opts.IncludeRawChunks {
			var decodedForRaw map[string]any
			decodedMapErr := json.Unmarshal(rawCopy, &decodedForRaw)
			var decodedCopy map[string]any
			if decodedMapErr == nil {
				decodedCopy = cloneDecodedMap(decodedForRaw)
			}
			parts <- StreamRaw{Raw: append([]byte(nil), raw...), Decoded: decodedCopy}
		}

		var chunk completionStreamChunk
		if err := json.Unmarshal(rawCopy, &chunk); err != nil {
			parts <- StreamError{Err: InvalidResponseDataError{Message: err.Error(), Data: string(rawCopy)}}
			parts <- StreamFinish{
				FinishReason:     errorFinishReason(),
				Usage:            completionStreamUsage(latestRawUsage),
				ProviderMetadata: ProviderMetadata(nil),
			}
			state.fatal = true
			return false
		}

		// Handle error chunks.
		if chunk.Error != nil {
			parts <- StreamError{Err: APIError{
				Message: chunk.Error.Message,
				Type:    chunk.Error.Type,
				Param:   chunk.Error.Param,
				Code:    chunk.Error.Code,
			}}
			parts <- StreamFinish{
				FinishReason:     errorFinishReason(),
				Usage:            completionStreamUsage(latestRawUsage),
				ProviderMetadata: ProviderMetadata(nil),
			}
			state.fatal = true
			return false
		}

		// On first valid normal chunk, emit response metadata then text start.
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
			parts <- StreamTextStart{ID: "0"}
		}

		// Track latest usage chunk.
		if len(chunk.Usage) > 0 && string(chunk.Usage) != "null" {
			latestRawUsage = cloneRawMessage(chunk.Usage)
		}

		// Update finish reason if first choice provides one.
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				state.finishReason = finishReasonFromOpenAI(*choice.FinishReason)
			}
			// Emit text delta if text is present.
			if choice.Text != "" {
				state.seenText = true
				parts <- StreamTextDelta{ID: "0", Text: choice.Text}
			}
		}
		return true
	})

	if state.fatal {
		return
	}

	// On flush, if at least one valid normal chunk was seen, emit text end.
	if state.metadataSent {
		parts <- StreamTextEnd{ID: "0"}
	}

	// Completion streams: ProviderMetadata is the zero value (no metadata extractor).
	parts <- StreamFinish{
		FinishReason:     state.finishReason,
		Usage:            completionStreamUsage(latestRawUsage),
		ProviderMetadata: ProviderMetadata(nil),
	}
}

// completionStreamState tracks mutable state during completion SSE streaming.
type completionStreamState struct {
	finishReason FinishReason
	metadataSent bool
	seenText     bool
	fatal        bool
}

// completionStreamUsage converts the latest raw usage chunk to a Usage struct.
func completionStreamUsage(raw json.RawMessage) Usage {
	if len(raw) == 0 || string(raw) == "null" {
		return Usage{}
	}
	var shape completionUsageShape
	if err := json.Unmarshal(raw, &shape); err != nil {
		return Usage{}
	}
	pub := shape.toPublic(raw)
	return completionUsage(pub)
}
