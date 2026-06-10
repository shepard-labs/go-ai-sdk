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
	if p == nil {
		return
	}
	if cand != nil && cand.GroundingMetadata != nil {
		state.extractAndEmitSources(parts, cand.GroundingMetadata)
	}
	// Reasoning text.
	if p.Text != "" {
		if p.Thought != nil && *p.Thought {
			if !state.reasoningOpen {
				state.reasoningOpen = true
				state.reasoningID = "reasoning-0"
				parts <- StreamReasoningStart{ID: state.reasoningID}
			}
			parts <- StreamReasoningDelta{ID: state.reasoningID, Text: p.Text}
			return
		}
		// Plain text.
		if state.reasoningOpen {
			parts <- StreamReasoningEnd{ID: state.reasoningID}
			state.reasoningOpen = false
		}
		if !state.textOpen {
			state.textOpen = true
			state.textID = "txt-0"
			parts <- StreamTextStart{ID: state.textID}
		}
		parts <- StreamTextDelta{ID: state.textID, Text: p.Text}
		return
	}
	// Inline data (file/image) — close any open text/reasoning.
	if p.InlineData != nil {
		if state.textOpen {
			parts <- StreamTextEnd{ID: state.textID}
			state.textOpen = false
		}
		if state.reasoningOpen {
			parts <- StreamReasoningEnd{ID: state.reasoningID}
			state.reasoningOpen = false
		}
		if p.Thought != nil && *p.Thought {
			parts <- StreamReasoningFile{Data: p.InlineData.Data, MediaType: p.InlineData.MimeType}
		} else {
			parts <- StreamFile{Data: p.InlineData.Data, MediaType: p.InlineData.MimeType}
		}
		return
	}
	// File data: surface the URI as a metadata-only event. We do NOT
	// emit StreamFile with empty data; the URL is recorded in
	// providerMetadata via a StreamRaw-style side channel. For now,
	// close any open text block and move on.
	if p.FileData != nil {
		if state.textOpen {
			parts <- StreamTextEnd{ID: state.textID}
			state.textOpen = false
		}
		if state.reasoningOpen {
			parts <- StreamReasoningEnd{ID: state.reasoningID}
			state.reasoningOpen = false
		}
		// The URL is conveyed as provider metadata; skip the file emit.
		return
	}
	// Executable code / code execution result.
	if p.ExecutableCode != nil {
		m.handleCodeExecutionStart(parts, p, state)
		return
	}
	if p.CodeExecutionResult != nil {
		m.handleCodeExecutionResult(parts, p, state)
		return
	}
	// Function call (client-side).
	if p.FunctionCall != nil {
		m.handleFunctionCall(parts, p, state)
		return
	}
	// Server tool call.
	if p.ToolCall != nil {
		m.handleServerToolCall(parts, p, state)
		return
	}
	// Server tool response.
	if p.ToolResponse != nil {
		m.handleServerToolResponse(parts, p, state)
		return
	}
	// Unknown part: record a warning, do not crash.
	state.warnings = append(state.warnings, Warning{
		Type:    "other",
		Feature: "unknown-part",
		Message: "unrecognized part keys; preserved in providerMetadata",
	})
}

// extractAndEmitSources walks the grounding metadata, deduplicates by
// URL, and emits a StreamSource for each new source.
func (s *chatStreamState) extractAndEmitSources(parts chan<- StreamPart, gm *internal.APIGroundingMetadata) {
	srcs := extractGroundingSources(gm)
	for _, src := range srcs {
		if src.URL == "" {
			continue
		}
		if _, seen := s.seenSourceURLs[src.URL]; seen {
			continue
		}
		s.seenSourceURLs[src.URL] = struct{}{}
		s.sources = append(s.sources, src)
		parts <- StreamSource{Source: src, ProviderMetadata: nil}
	}
}

// flushStreamState closes any still-open text/reasoning/tool blocks at
// the end of the stream and emits any deferred tool calls.
func (s *chatStreamState) flushStreamState(parts chan<- StreamPart) {
	if s.reasoningOpen {
		parts <- StreamReasoningEnd{ID: s.reasoningID}
		s.reasoningOpen = false
	}
	if s.textOpen {
		parts <- StreamTextEnd{ID: s.textID}
		s.textOpen = false
	}
	for _, st := range s.toolState {
		if st.HasStarted && !st.HasEnded {
			parts <- StreamToolInputEnd{ID: st.ID}
			st.HasEnded = true
		}
	}
	// Emit any deferred tool calls. For streaming-chunk calls the
	// Input must include the accumulator's closing characters.
	for _, st := range s.toolState {
		if st.Emitted {
			continue
		}
		if !st.HasStarted {
			continue
		}
		var input json.RawMessage
		if st.Accumulator != nil {
			closing, err := st.Accumulator.Finalize()
			if err == nil && closing != "" {
				st.AccruedArgs = append(st.AccruedArgs, closing...)
			}
			if len(st.AccruedArgs) > 0 {
				input = st.AccruedArgs
			} else {
				input = json.RawMessage("{}")
			}
		} else {
			if len(st.AccruedArgs) > 0 {
				input = st.AccruedArgs
			} else {
				input = json.RawMessage("{}")
			}
		}
		pm := providerMetadataWithSig(s.model.provider.name, st.ThoughtSignature)
		parts <- StreamToolCall{ToolCall: ToolCallContent{
			ToolCallID:       st.ID,
			ToolName:         st.ToolName,
			Input:            input,
			ProviderExecuted: st.ProviderExecuted,
			Dynamic:          st.Dynamic,
			ProviderMetadata: pm,
		}}
		st.Emitted = true
	}
}

// ---- Part handlers (filled in by subsequent tasks) ----

func (m *googleLanguageModel) handleCodeExecutionStart(parts chan<- StreamPart, p *internal.APIPart, state *chatStreamState) {
	if p.ExecutableCode == nil {
		return
	}
	id := m.provider.generateID()
	input, _ := json.Marshal(map[string]any{
		"language": p.ExecutableCode.Language,
		"code":     p.ExecutableCode.Code,
	})
	pm := ProviderMetadata{"google": map[string]any{
		"providerExecuted": true,
		"thoughtSignature": p.ThoughtSignature,
	}}
	state.toolState[id] = &streamToolState{
		ID:               id,
		ToolName:         "code_execution",
		ProviderExecuted: true,
		IsServer:         true,
		IsCodeExecution:  true,
		ThoughtSignature: p.ThoughtSignature,
		HasStarted:       true,
	}
	state.lastCodeExecID = id
	state.toolState[id].Emitted = true
	parts <- StreamToolCall{ToolCall: ToolCallContent{
		ToolCallID:       id,
		ToolName:         "code_execution",
		Input:            input,
		ProviderExecuted: true,
		ProviderMetadata: pm,
	}}
}

func (m *googleLanguageModel) handleCodeExecutionResult(parts chan<- StreamPart, p *internal.APIPart, state *chatStreamState) {
	if p.CodeExecutionResult == nil {
		return
	}
	id := state.lastCodeExecID
	if id == "" {
		id = "code_execution-0"
	}
	output, _ := json.Marshal(map[string]any{
		"outcome": p.CodeExecutionResult.Outcome,
		"output":  p.CodeExecutionResult.Output,
	})
	parts <- StreamToolResult{ToolResult: ToolResultContent{
		ToolCallID: id,
		Output:     ToolResultOutput{Type: "json", Value: json.RawMessage(output)},
	}}
}

func (m *googleLanguageModel) handleFunctionCall(parts chan<- StreamPart, p *internal.APIPart, state *chatStreamState) {
	if p.FunctionCall == nil {
		return
	}
	fc := p.FunctionCall
	id := fc.ID
	if id == "" {
		id = m.provider.generateID()
	}
	st, exists := state.toolState[id]
	if !exists {
		st = &streamToolState{
			ID:               id,
			ToolName:         fc.Name,
			ProviderExecuted: false,
			ThoughtSignature: p.ThoughtSignature,
		}
		state.toolState[id] = st
	}
	// Streaming chunk path: partialArgs present, or willContinue=true.
	if len(fc.PartialArgs) > 0 || (fc.WillContinue != nil && *fc.WillContinue) {
		m.handleFunctionCallStreamingChunk(parts, fc, st, state)
		return
	}
	// Complete call path: no partialArgs, no willContinue.
	if !st.HasStarted {
		st.HasStarted = true
		parts <- StreamToolInputStart{ID: id, ToolName: fc.Name}
	}
	if len(fc.Args) > 0 {
		st.AccruedArgs = append(st.AccruedArgs, fc.Args...)
		parts <- StreamToolInputDelta{ID: id, Delta: string(fc.Args)}
	}
	if !st.HasEnded {
		parts <- StreamToolInputEnd{ID: id}
		st.HasEnded = true
	}
	// StreamToolCall is emitted by flushStreamState so the Input field
	// can include the accumulator's closing characters.
}

// handleFunctionCallStreamingChunk processes the accumulator path for
// client-side function calls that arrive with partialArgs.
func (m *googleLanguageModel) handleFunctionCallStreamingChunk(parts chan<- StreamPart, fc *internal.APIFunctionCall, st *streamToolState, state *chatStreamState) {
	if st.Accumulator == nil {
		st.Accumulator = &GoogleJSONAccumulator{}
	}
	if !st.HasStarted {
		st.HasStarted = true
		parts <- StreamToolInputStart{ID: st.ID, ToolName: st.ToolName}
	}
	for i := range fc.PartialArgs {
		arg := fc.PartialArgs[i]
		frag, err := st.Accumulator.Push(arg)
		if err != nil {
			state.warnings = append(state.warnings, Warning{Type: "other", Feature: "partial-args", Message: err.Error()})
			continue
		}
		if frag != "" {
			st.AccruedArgs = append(st.AccruedArgs, frag...)
			parts <- StreamToolInputDelta{ID: st.ID, Delta: frag}
		}
	}
	willContinue := fc.WillContinue != nil && *fc.WillContinue
	if willContinue {
		// Stream continues; do not emit End yet.
		return
	}
	// Stream is done — close the input stream but DEFER the tool call
	// emission until flushStreamState so the Input field can include
	// the accumulator's closing characters (which are only known after
	// Finalize).
	if !st.HasEnded {
		parts <- StreamToolInputEnd{ID: st.ID}
		st.HasEnded = true
	}
}

func (m *googleLanguageModel) handleFunctionResponse(parts chan<- StreamPart, p *internal.APIPart, state *chatStreamState) {
	_ = parts
	_ = p
	_ = state
}

func providerMetadataWithSig(name, sig string) ProviderMetadata {
	if sig == "" {
		return nil
	}
	return ProviderMetadata{name: map[string]any{"thoughtSignature": sig}}
}

func (m *googleLanguageModel) handleServerToolCall(parts chan<- StreamPart, p *internal.APIPart, state *chatStreamState) {
	if p.ToolCall == nil {
		return
	}
	tc := p.ToolCall
	id := tc.ID
	if id == "" {
		id = m.provider.generateID()
	}
	input := tc.Args
	if len(input) == 0 {
		input = json.RawMessage("{}")
	}
	pm := ProviderMetadata{"google": map[string]any{
		"serverToolType":  tc.ToolType,
		"thoughtSignature": p.ThoughtSignature,
	}}
	state.toolState[id] = &streamToolState{
		ID:               id,
		ToolName:         tc.ToolType,
		ProviderExecuted: true,
		IsServer:         true,
		Dynamic:          true,
		ServerToolType:   tc.ToolType,
		ThoughtSignature: p.ThoughtSignature,
		HasStarted:       true,
		Emitted:          true,
	}
	state.lastServerToolID = id
	parts <- StreamToolCall{ToolCall: ToolCallContent{
		ToolCallID:       id,
		ToolName:         tc.ToolType,
		Input:            input,
		ProviderExecuted: true,
		Dynamic:          true,
		ProviderMetadata: pm,
	}}
}

func (m *googleLanguageModel) handleServerToolResponse(parts chan<- StreamPart, p *internal.APIPart, state *chatStreamState) {
	if p.ToolResponse == nil {
		return
	}
	tr := p.ToolResponse
	id := tr.ID
	if id == "" {
		id = state.lastServerToolID
	}
	pm := ProviderMetadata{"google": map[string]any{"serverToolType": tr.ToolType}}
	parts <- StreamToolResult{ToolResult: ToolResultContent{
		ToolCallID:       id,
		Output:           ToolResultOutput{Type: "json", Value: tr.Response},
		ProviderExecuted: true,
		ProviderMetadata: pm,
	}}
}
