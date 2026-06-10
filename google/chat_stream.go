package google

// chat_stream.go implements the Google Generative AI chat streaming
// model. The streaming path reuses the same buildChatRequest /
// convertToGoogleMessages / mapPartsToContent that DoGenerate uses, then
// walks the SSE stream chunk-by-chunk converting each APIGenerateContentResponse
// into a sequence of StreamPart events.
//
// Mirrors the upstream google-language-model.ts doStream path. The
// accumulator (json_accumulator.go) handles partial-args fragment
// assembly for client-side function calls.

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// DoStream performs a streaming generation call against
// POST {baseURL}/models/{modelPath}:streamGenerateContent?alt=sse.
//
// The body is built with the same buildChatRequest used by DoGenerate.
// The returned StreamResult exposes a channel of StreamPart events.
func (m *googleLanguageModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}

	args, err := m.getArgs(opts.GenerateOptions)
	if err != nil {
		return nil, err
	}
	contents, system, convertWarnings, err := ConvertPrompt(m.modelID, opts.GenerateOptions)
	if err != nil {
		return nil, err
	}
	args.Warnings = append(args.Warnings, convertWarnings...)

	body, extraHeaders, buildWarnings, err := m.buildChatRequest(args, opts.GenerateOptions, contents, system)
	if err != nil {
		return nil, err
	}
	args.Warnings = append(args.Warnings, buildWarnings...)

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	perCall := cloneHeader(opts.Headers)
	for k, values := range extraHeaders {
		perCall.Del(k)
		for _, v := range values {
			perCall.Add(k, v)
		}
	}

	path := "/" + getModelPath(m.modelID) + ":streamGenerateContent?alt=sse"
	resp, err := m.provider.executeStream(ctx, path, bodyBytes, perCall)
	if err != nil {
		return nil, err
	}

	parts := make(chan StreamPart)
	sresp := streamResponseFromHeaders(resp.Headers)
	result := &StreamResult{
		Stream:   parts,
		Parts:    parts,
		Request:  RequestMetadata{Body: append([]byte(nil), bodyBytes...)},
		Response: sresp,
	}
	go m.runChatStream(ctx, resp, parts, sresp, args.Warnings, opts)
	return result, nil
}

// runChatStream walks the SSE stream chunk-by-chunk and emits StreamParts.
func (m *googleLanguageModel) runChatStream(
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

	state := newChatStreamState(m, opts, sresp)

	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			parts <- StreamError{Err: ctx.Err()}
			state.fatal = true
			return false
		default:
		}

		rawCopy := append([]byte(nil), raw...)

		if opts.IncludeRawChunks {
			decoded := decodeForRaw(rawCopy)
			parts <- StreamRaw{Raw: append([]byte(nil), rawCopy...), Decoded: decoded}
		}

		var chunk internal.APIGenerateContentResponse
		if err := json.Unmarshal(rawCopy, &chunk); err != nil {
			// Skip malformed chunk; record a warning and continue.
			state.warnings = append(state.warnings, Warning{
				Type:    "other",
				Feature: "malformed-sse-chunk",
				Message: "skipped SSE chunk: " + err.Error(),
			})
			return true
		}
		m.processChatStreamChunk(parts, &chunk, state)
		return true
	})

	if state.fatal {
		return
	}
	state.flushStreamState(parts)
	finalMeta := state.buildMetadata()
	sresp.ProviderMetadata = finalMeta
	parts <- StreamFinish{
		FinishReason:     state.finishReason,
		Usage:            state.usage,
		ProviderMetadata: finalMeta,
	}
}

func decodeForRaw(raw []byte) map[string]any {
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil
	}
	return out
}

// streamToolState holds per-tool-call state across chunks.
type streamToolState struct {
	ID               string
	ToolName         string
	ProviderExecuted bool
	Dynamic          bool
	IsServer         bool
	Accumulator      *GoogleJSONAccumulator
	AccruedArgs      []byte
	HasStarted       bool
	HasEnded         bool
	Emitted          bool
	ThoughtSignature string
	ServerToolType   string
	IsCodeExecution  bool
}

// chatStreamState holds the open-block state for a single streaming call.
type chatStreamState struct {
	model *googleLanguageModel
	opts  StreamOptions
	sresp *StreamResponse

	textID         string
	textOpen       bool
	reasoningID    string
	reasoningOpen  bool
	toolState      map[string]*streamToolState
	seenSourceURLs map[string]struct{}
	sources        []Source

	usage            Usage
	finishReason     FinishReason
	warnings         []Warning
	fatal            bool
	lastCodeExecID   string
	lastServerToolID string
}

func newChatStreamState(m *googleLanguageModel, opts StreamOptions, sresp *StreamResponse) *chatStreamState {
	return &chatStreamState{
		model:          m,
		opts:           opts,
		sresp:          sresp,
		toolState:      map[string]*streamToolState{},
		seenSourceURLs: map[string]struct{}{},
	}
}

// buildMetadata assembles the final ProviderMetadata emitted on StreamFinish.
func (s *chatStreamState) buildMetadata() ProviderMetadata {
	pm := ProviderMetadata{}
	gm := map[string]any{}
	pm["google"] = gm
	if s.sresp != nil {
		if s.sresp.ID != "" {
			gm["responseId"] = s.sresp.ID
		}
		if s.sresp.ModelID != "" {
			gm["modelVersion"] = s.sresp.ModelID
		}
	}
	if len(s.sources) > 0 {
		gm["sources"] = sourcesToPublic(s.sources)
	}
	if len(s.warnings) > 0 {
		gm["warnings"] = warningsToMap(s.warnings)
	}
	return pm
}

func warningsToMap(ws []Warning) []map[string]any {
	out := make([]map[string]any, 0, len(ws))
	for _, w := range ws {
		entry := map[string]any{"type": w.Type}
		if w.Feature != "" {
			entry["feature"] = w.Feature
		}
		if w.Message != "" {
			entry["message"] = w.Message
		}
		if w.Details != "" {
			entry["details"] = w.Details
		}
		out = append(out, entry)
	}
	return out
}

// processChatStreamChunk walks a single SSE chunk's parts and emits StreamParts.
func (m *googleLanguageModel) processChatStreamChunk(parts chan<- StreamPart, chunk *internal.APIGenerateContentResponse, state *chatStreamState) {
	// Stash usage and response metadata even when no parts.
	if chunk.ResponseID != "" {
		state.sresp.ID = chunk.ResponseID
	}
	if chunk.ModelVersion != "" {
		state.sresp.ModelID = chunk.ModelVersion
	}
	if chunk.UsageMetadata != nil {
		state.usage = convertGoogleUsage(chunk.UsageMetadata)
	}
	if len(chunk.Candidates) == 0 {
		return
	}
	cand := chunk.Candidates[0]
	if cand.FinishReason != "" {
		state.finishReason = mapGoogleFinishReason(cand.FinishReason, false)
	}
	for i := range cand.Content.Parts {
		m.processChatStreamPart(parts, &cand.Content.Parts[i], state, &cand)
	}
}

// processChatStreamPart handles one APIPart and emits the corresponding
// StreamParts. The candidate is needed to access GroundingMetadata for
// source extraction.
func (m *googleLanguageModel) processChatStreamPart(parts chan<- StreamPart, p *internal.APIPart, state *chatStreamState, cand *internal.APICandidate) {
	_ = http.Header{}
	// Body filled in by Task 6.
	_ = parts
	_ = p
	_ = state
	_ = cand
}

// flushStreamState closes any still-open text/reasoning/tool blocks at
// the end of the stream.
func (s *chatStreamState) flushStreamState(parts chan<- StreamPart) {
	if s.reasoningOpen {
		parts <- StreamReasoningEnd{ID: s.reasoningID}
		s.reasoningOpen = false
	}
	if s.textOpen {
		parts <- StreamTextEnd{ID: s.textID}
		s.textOpen = false
	}
	for id, st := range s.toolState {
		if st.HasStarted && !st.HasEnded {
			parts <- StreamToolInputEnd{ID: id}
			st.HasEnded = true
		}
	}
}
