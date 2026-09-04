package openai

import (
	"context"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// responsesStreamState holds the rolling state of a Responses stream.
type responsesStreamState struct {
	metadataSent bool
	finishReason FinishReason
	text         map[string]*responsesTextAccum
	tools        map[string]*responsesToolAccum
	reasoning    map[string]*responsesReasonAccum
	approvals    map[string]*responsesApprovalAccum
	usage        *Usage
	annotations  map[string][]map[string]any
	finishMeta   ProviderMetadata
}

type responsesApprovalAccum struct {
	approvalID string
	toolCallID string
}

type responsesTextAccum struct {
	id          string
	text        strings.Builder
	phase       string
	annotations []map[string]any
}

type responsesToolAccum struct {
	id       string
	input    strings.Builder
	toolName string
	started  bool
}

type responsesReasonAccum struct {
	id          string
	text        strings.Builder
	encrypted   string
	active      bool
	canConclude bool
}

func newResponsesStreamState() *responsesStreamState {
	return &responsesStreamState{
		text:        map[string]*responsesTextAccum{},
		tools:       map[string]*responsesToolAccum{},
		reasoning:   map[string]*responsesReasonAccum{},
		approvals:   map[string]*responsesApprovalAccum{},
		annotations: map[string][]map[string]any{},
	}
}

// responsesAccumKey returns the canonical map key for a Responses stream
// accumulator: the output_index when present, otherwise the item id (or
// the fallback when both are absent).
func responsesAccumKey(outputIndex *int, itemID, fallback string) string {
	if outputIndex != nil {
		return "index:" + strconv.Itoa(*outputIndex)
	}
	if itemID != "" {
		return "id:" + itemID
	}
	return "id:" + fallback
}

// responsesAccumRotated reports whether an accumulator stored under an
// output_index key now faces an event for a different item id: the index
// was reused and the old accumulator must be closed first.
func responsesAccumRotated(storedID, eventID string) bool {
	return storedID != "" && eventID != "" && storedID != eventID
}

// responsesDisplayID returns the id to surface in stream parts, falling
// back when the accumulator never learned its item id.
func responsesDisplayID(storedID, fallback string) string {
	if storedID != "" {
		return storedID
	}
	return fallback
}

// lookupResponsesTextAccum finds the text accumulator for an event by its
// canonical key, falling back to an item-id scan for events that omit
// output_index. It returns the key to store under on a miss.
func lookupResponsesTextAccum(state *responsesStreamState, outputIndex *int, itemID, fallback string) (string, *responsesTextAccum, bool) {
	key := responsesAccumKey(outputIndex, itemID, fallback)
	if acc, ok := state.text[key]; ok {
		return key, acc, true
	}
	if itemID != "" {
		for k, acc := range state.text {
			if acc.id == itemID {
				return k, acc, true
			}
		}
	}
	return key, nil, false
}

// lookupResponsesToolAccum finds the tool accumulator for an event by its
// canonical key, falling back to an item-id scan for events that omit
// output_index. It returns the key to store under on a miss.
func lookupResponsesToolAccum(state *responsesStreamState, outputIndex *int, itemID string) (string, *responsesToolAccum, bool) {
	key := responsesAccumKey(outputIndex, itemID, "")
	if acc, ok := state.tools[key]; ok {
		return key, acc, true
	}
	if itemID != "" {
		for k, acc := range state.tools {
			if acc.id == itemID {
				return k, acc, true
			}
		}
	}
	return key, nil, false
}

// lookupResponsesReasoningAccum finds the reasoning accumulator for an
// event by its canonical key, falling back to an item-id scan for events
// that omit output_index. It returns the key to store under on a miss.
func lookupResponsesReasoningAccum(state *responsesStreamState, outputIndex *int, itemID, fallback string) (string, *responsesReasonAccum, bool) {
	key := responsesAccumKey(outputIndex, itemID, fallback)
	if acc, ok := state.reasoning[key]; ok {
		return key, acc, true
	}
	if itemID != "" {
		for k, acc := range state.reasoning {
			if acc.id == itemID {
				return k, acc, true
			}
		}
	}
	return key, nil, false
}

// textAccumForDelta returns the text accumulator for a delta event,
// closing and replacing it first when the output_index was reused with a
// new item id.
func textAccumForDelta(parts chan<- StreamPart, state *responsesStreamState, event responsesStreamEvent) *responsesTextAccum {
	key, acc, ok := lookupResponsesTextAccum(state, event.OutputIndex, event.ItemID, "txt-0")
	if ok && responsesAccumRotated(acc.id, event.ItemID) {
		closeResponsesTextAccum(parts, state, key)
		ok = false
	}
	if ok {
		if acc.id == "" && event.ItemID != "" {
			acc.id = event.ItemID
		}
		return acc
	}
	acc = &responsesTextAccum{id: event.ItemID}
	state.text[key] = acc
	return acc
}

// toolAccumForDelta returns the tool accumulator for a delta event,
// closing and replacing it first when the output_index was reused with a
// new item id. defaultName applies when the event carries no name.
func toolAccumForDelta(parts chan<- StreamPart, state *responsesStreamState, event responsesStreamEvent, defaultName string) *responsesToolAccum {
	key, acc, ok := lookupResponsesToolAccum(state, event.OutputIndex, event.ItemID)
	if ok && responsesAccumRotated(acc.id, event.ItemID) {
		closeResponsesToolAccum(parts, state, key, nil)
		ok = false
	}
	if ok {
		if acc.id == "" && event.ItemID != "" {
			acc.id = event.ItemID
		}
		if event.Name != "" && acc.toolName == "" {
			acc.toolName = event.Name
		}
		// Lazily emit Start once the tool name is known, so deltas that
		// arrived before output_item.added are still surfaced.
		if !acc.started && acc.toolName != "" {
			parts <- StreamToolInputStart{ID: acc.id, ToolName: acc.toolName}
			acc.started = true
		}
		return acc
	}
	name := event.Name
	if name == "" {
		name = defaultName
	}
	acc = &responsesToolAccum{id: event.ItemID, toolName: name}
	state.tools[key] = acc
	return acc
}

// reasoningAccumForDelta returns the reasoning accumulator for a delta
// event, closing and replacing it first when the output_index was reused
// with a new item id.
func reasoningAccumForDelta(parts chan<- StreamPart, state *responsesStreamState, event responsesStreamEvent) *responsesReasonAccum {
	key, acc, ok := lookupResponsesReasoningAccum(state, event.OutputIndex, event.ItemID, "reasoning-0")
	if ok && responsesAccumRotated(acc.id, event.ItemID) {
		closeResponsesReasoningAccum(parts, state, key)
		ok = false
	}
	if ok {
		if acc.id == "" && event.ItemID != "" {
			acc.id = event.ItemID
		}
		return acc
	}
	acc = &responsesReasonAccum{id: event.ItemID, active: true}
	state.reasoning[key] = acc
	return acc
}

// closeResponsesTextAccum emits the end part for a text accumulator and
// removes it. It is a no-op when the key holds nothing.
func closeResponsesTextAccum(parts chan<- StreamPart, state *responsesStreamState, key string) {
	acc, ok := state.text[key]
	if !ok {
		return
	}
	delete(state.text, key)
	endPM := ProviderMetadata{}
	if len(acc.annotations) > 0 {
		endPM["openai"] = map[string]any{"annotations": acc.annotations}
	}
	parts <- StreamTextEnd{ID: responsesDisplayID(acc.id, "txt-0"), ProviderMetadata: endPM}
}

// closeResponsesToolAccum finalizes a tool accumulator and removes it. It
// is a no-op when the key holds nothing. Valid input (or empty input,
// coerced to "{}") closes with input end plus a tool call; invalid input
// is never executed as a tool call — it closes with input end plus a
// StreamError and latches an error finish reason.
func closeResponsesToolAccum(parts chan<- StreamPart, state *responsesStreamState, key string, extraPM ProviderMetadata) {
	acc, ok := state.tools[key]
	if !ok {
		return
	}
	delete(state.tools, key)
	input := acc.input.String()
	parts <- StreamToolInputEnd{ID: acc.id}
	if input != "" && !json.Valid([]byte(input)) {
		parts <- StreamError{Err: InvalidResponseDataError{Message: "Invalid tool call arguments JSON.", Data: input}}
		state.finishReason = errorFinishReason()
		return
	}
	if input == "" {
		input = "{}"
	}
	parts <- StreamToolCall{ToolCallContent: ToolCallContent{
		ToolCallContentEmbed: ToolCallContentEmbed{
			ToolCallID:       acc.id,
			ToolName:         acc.toolName,
			Input:            json.RawMessage(input),
			ProviderMetadata: extraPM,
		},
	}}
}

// closeResponsesReasoningAccum emits the end part for a reasoning
// accumulator and removes it. It is a no-op when the key holds nothing.
func closeResponsesReasoningAccum(parts chan<- StreamPart, state *responsesStreamState, key string) {
	acc, ok := state.reasoning[key]
	if !ok {
		return
	}
	delete(state.reasoning, key)
	endPM := ProviderMetadata{}
	om := map[string]any{}
	if acc.id != "" {
		om["itemId"] = acc.id
	}
	if acc.encrypted != "" {
		om["reasoningEncryptedContent"] = acc.encrypted
	}
	if len(om) > 0 {
		endPM["openai"] = om
	}
	parts <- StreamReasoningEnd{ID: responsesDisplayID(acc.id, "reasoning-0"), ProviderMetadata: endPM}
}

// runResponsesStream drives a Responses API SSE stream, translating events
// into StreamPart messages.
func (m *openaiResponsesModel) runResponsesStream(
	ctx context.Context,
	resp *httpStreamResponse,
	parts chan<- StreamPart,
	sresp *StreamResponse,
	warnings []Warning,
	opts ResponsesStreamOptions,
) {
	defer close(parts)
	defer resp.Body.Close()
	parts <- StreamStart{Warnings: warnings}
	headers := resp.Headers.Clone()
	sresp.Headers = headers

	state := newResponsesStreamState()
	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}
		if opts.IncludeRawChunks {
			parts <- StreamRaw{Raw: append([]byte(nil), raw...)}
		}
		// A Chat Completions SSE chunk sent to the Responses parser
		// (choices-shaped, no Responses "type") would otherwise match no
		// case and surface as a generic unknown-event error. Fail fast
		// with an actionable wrong-surface error instead.
		if isChoicesShapedChunk(raw) {
			parts <- StreamError{Err: &APICallError{
				Message: "wrong API surface: received a Chat Completions chunk (\"choices\") on a Responses stream; use the chat model instead of the responses model for Chat Completions SSE",
			}}
			state.finishReason = errorFinishReason()
			return false
		}
		var event responsesStreamEvent
		if err := json.Unmarshal(raw, &event); err != nil {
			parts <- StreamError{Err: InvalidResponseDataError{Message: err.Error(), Data: string(raw)}}
			state.finishReason = errorFinishReason()
			return false
		}
		event.Raw = string(raw)
		m.processResponsesEvent(parts, sresp, state, event)
		return true
	})
	// Close any open accumulators in deterministic key order (map
	// iteration order is random). Invalid tool input is rejected with a
	// StreamError, never executed as a tool call.
	textKeys := make([]string, 0, len(state.text))
	for k := range state.text {
		textKeys = append(textKeys, k)
	}
	sort.Strings(textKeys)
	for _, k := range textKeys {
		closeResponsesTextAccum(parts, state, k)
	}
	toolKeys := make([]string, 0, len(state.tools))
	for k := range state.tools {
		toolKeys = append(toolKeys, k)
	}
	sort.Strings(toolKeys)
	for _, k := range toolKeys {
		closeResponsesToolAccum(parts, state, k, nil)
	}
	reasoningKeys := make([]string, 0, len(state.reasoning))
	for k := range state.reasoning {
		reasoningKeys = append(reasoningKeys, k)
	}
	sort.Strings(reasoningKeys)
	for _, k := range reasoningKeys {
		closeResponsesReasoningAccum(parts, state, k)
	}
	finalUsage := Usage{}
	if state.usage != nil {
		finalUsage = *state.usage
	}
	parts <- StreamFinish{
		FinishReason:     state.finishReason,
		Usage:            finalUsage,
		ProviderMetadata: state.finishMeta,
	}
}

// isChoicesShapedChunk reports whether a raw SSE payload is a Chat
// Completions chunk (a top-level non-empty "choices" array of objects
// carrying delta/message, with no Responses event "type") rather than a
// Responses event. Requiring an object element with delta/message avoids
// flagging null/[]/scalar choices payloads as the wrong surface.
func isChoicesShapedChunk(raw []byte) bool {
	var probe struct {
		Type    string           `json:"type"`
		Choices []map[string]any `json:"choices"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	if probe.Type != "" || len(probe.Choices) == 0 {
		return false
	}
	first := probe.Choices[0]
	if _, ok := first["delta"]; ok {
		return true
	}
	if _, ok := first["message"]; ok {
		return true
	}
	return false
}

// responsesStreamEvent is the union of fields a Responses API event can
// carry.
type responsesStreamEvent struct {
	Type         string                `json:"type"`
	Response     *responsesStreamResp  `json:"response"`
	Item         map[string]any        `json:"item"`
	OutputIndex  *int                  `json:"output_index"`
	ContentIndex *int                  `json:"content_index"`
	Delta        string                `json:"delta"`
	Text         string                `json:"text"`
	Arguments    string                `json:"arguments"`
	ItemID       string                `json:"item_id"`
	CallID       string                `json:"call_id"`
	Name         string                `json:"name"`
	Code         string                `json:"code"`
	Output       string                `json:"output"`
	Annotation   map[string]any        `json:"annotation"`
	SummaryIndex *int                  `json:"summary_index"`
	Encrypted    string                `json:"encrypted_content"`
	Operation    map[string]any        `json:"operation"`
	Tools        []map[string]any      `json:"tools"`
	Execution    string                `json:"execution"`
	ServerLabel  string                `json:"server_label"`
	Result       string                `json:"result"`
	Queries      []any                 `json:"queries"`
	Results      []any                 `json:"results"`
	Error        *chatStreamErrorShape `json:"error"`
	// Raw carries the original SSE payload for error reporting. It is
	// populated by runResponsesStream after unmarshalling and is never
	// part of the wire format.
	Raw string `json:"-"`
}

type responsesStreamResp struct {
	ID          string           `json:"id"`
	Model       string           `json:"model"`
	CreatedAt   *int64           `json:"created_at"`
	Status      string           `json:"status"`
	ServiceTier string           `json:"service_tier"`
	Usage       map[string]any   `json:"usage"`
	Output      []map[string]any `json:"output"`
}

// processResponsesEvent applies a single Responses SSE event.
func (m *openaiResponsesModel) processResponsesEvent(parts chan<- StreamPart, sresp *StreamResponse, state *responsesStreamState, event responsesStreamEvent) {
	switch event.Type {
	case "response.created":
		if event.Response != nil && !state.metadataSent {
			state.metadataSent = true
			if event.Response.ID != "" {
				sresp.ID = event.Response.ID
			}
			if event.Response.Model != "" {
				sresp.ModelID = event.Response.Model
			}
			if event.Response.CreatedAt != nil {
				ts := time.Unix(*event.Response.CreatedAt, 0)
				sresp.Timestamp = &ts
			}
			parts <- StreamResponseMetadata{ID: sresp.ID, ModelID: sresp.ModelID, Timestamp: sresp.Timestamp}
		}
	case "response.output_item.added":
		handleResponsesItemAdded(parts, state, event)
	case "response.output_item.done":
		handleResponsesItemDone(parts, state, event)
	case "response.output_text.delta":
		acc := textAccumForDelta(parts, state, event)
		acc.text.WriteString(event.Delta)
		parts <- StreamTextDelta{ID: responsesDisplayID(acc.id, "txt-0"), Text: event.Delta}
	case "response.reasoning_summary_text.delta":
		acc := reasoningAccumForDelta(parts, state, event)
		acc.text.WriteString(event.Delta)
		parts <- StreamReasoningDelta{ID: responsesDisplayID(acc.id, "reasoning-0"), Text: event.Delta}
	case "response.reasoning_summary_part.added":
		// A new reasoning summary part is starting (only seen for
		// multi-part summaries). If the index is > 0, any earlier
		// `can-conclude` parts are now eligible to be concluded.
		if event.SummaryIndex != nil && *event.SummaryIndex > 0 {
			concludeCanConcludeReasoning(parts, state)
		}
	case "response.reasoning_summary_part.done":
		// The summary part has finished streaming its text. Behavior:
		//   store=true: conclude immediately.
		//   store=false: mark can-conclude (defer conclusion to allow
		//     the final part with encrypted_content to attach to
		//     reasoning-end providerMetadata).
		//   store=nil (default): conclude immediately to match server
		//     default of store=true.
		acc := reasoningAccumForDelta(parts, state, event)
		id := responsesDisplayID(acc.id, "reasoning-0")
		concludeImmediately := m.store == nil || *m.store
		if concludeImmediately {
			acc.active = false
			parts <- StreamReasoningEnd{ID: id}
		} else {
			acc.canConclude = true
		}
	case "response.function_call_arguments.delta":
		acc := toolAccumForDelta(parts, state, event, "")
		acc.input.WriteString(event.Delta)
		// Emit Start lazily so that deltas arriving before the
		// output_item.added (which carries function.name) are still
		// surfaced with the correct tool name.
		if !acc.started && acc.toolName != "" {
			parts <- StreamToolInputStart{ID: acc.id, ToolName: acc.toolName}
			acc.started = true
		}
		if acc.started {
			parts <- StreamToolInputDelta{ID: acc.id, Delta: event.Delta}
		}
	case "response.custom_tool_call_input.delta":
		acc := toolAccumForDelta(parts, state, event, "")
		acc.input.WriteString(event.Delta)
		if !acc.started && acc.toolName != "" {
			parts <- StreamToolInputStart{ID: acc.id, ToolName: acc.toolName}
			acc.started = true
		}
		if acc.started {
			parts <- StreamToolInputDelta{ID: acc.id, Delta: event.Delta}
		}
	case "response.code_interpreter_call_code.delta":
		acc := toolAccumForDelta(parts, state, event, "code_interpreter")
		acc.input.WriteString(event.Delta)
		// Emit Start lazily so that deltas arriving before the
		// output_item.added are still surfaced with the correct tool
		// name, and Start always precedes the first Delta.
		if !acc.started && acc.toolName != "" {
			parts <- StreamToolInputStart{ID: acc.id, ToolName: acc.toolName}
			acc.started = true
		}
		if acc.started {
			parts <- StreamToolInputDelta{ID: acc.id, Delta: event.Delta}
		}
	case "response.image_generation_call.partial_image":
		// Per spec: emit a preliminary ToolResultContent-shaped event with
		// the partial base64 image as Output.Value and preliminary=true in
		// ProviderMetadata. We use StreamCustomPart to carry the typed
		// ToolResultContent since StreamPart has no dedicated
		// StreamToolResultContent variant.
		partial := ToolResultContent{
			ToolResultContent: openaicompatible.ToolResultContent{
				ToolCallID: event.ItemID,
				Output: openaicompatible.ToolResultOutput{
					Type:  "content",
					Value: event.Result,
				},
			},
			ToolName: "image_generation",
			ProviderMetadata: ProviderMetadata{
				"openai": map[string]any{"preliminary": true},
			},
		}
		parts <- StreamCustomPart{Kind: "image_generation.partial_image", Data: partial}
	case "response.apply_patch_call_operation_diff.delta":
		// apply_patch streams its operation as a series of diffs; each
		// one is emitted as a tool input delta (the upstream SDK
		// JSON-escapes the diff before emitting).
		acc := toolAccumForDelta(parts, state, event, "apply_patch")
		escaped, _ := json.Marshal(event.Delta)
		_, _ = acc.input.Write([]byte(event.Delta))
		// Emit Start lazily so that deltas arriving before the
		// output_item.added are still surfaced with the correct tool
		// name, and Start always precedes the first Delta.
		if !acc.started && acc.toolName != "" {
			parts <- StreamToolInputStart{ID: acc.id, ToolName: acc.toolName}
			acc.started = true
		}
		if acc.started {
			parts <- StreamToolInputDelta{ID: acc.id, Delta: string(escaped)}
		}
	case "response.apply_patch_call_operation_diff.done":
		// The spec says apply_patch's output_item.done is what normally
		// closes the tool input, but the diff.done event also signals
		// completion. Mark the accumulator as ended so the next
		// output_item.done doesn't emit a duplicate.
		if _, acc, ok := lookupResponsesToolAccum(state, event.OutputIndex, event.ItemID); ok {
			acc.started = true
		}
	case "response.output_text.annotation.added":
		annotationType, _ := event.Annotation["type"].(string)
		id, _ := event.Annotation["id"].(string)
		url, _ := event.Annotation["url"].(string)
		title, _ := event.Annotation["title"].(string)
		parts <- SourceContent{SourceType: annotationType, ID: id, URL: url, Title: title}
		// Stash on the corresponding text accumulator so the
		// StreamTextEnd can carry the full annotations list in
		// ProviderMetadata["openai"].
		if event.ItemID != "" {
			key, acc, ok := lookupResponsesTextAccum(state, event.OutputIndex, event.ItemID, "")
			if !ok {
				acc = &responsesTextAccum{id: event.ItemID}
				state.text[key] = acc
			}
			acc.annotations = append(acc.annotations, event.Annotation)
		}
	case "response.completed", "response.incomplete":
		if event.Response != nil {
			// A latched error finish (unknown event type, stream
			// error, failure) is never overwritten by a later
			// completed/stop.
			if state.finishReason != errorFinishReason() {
				state.finishReason = responsesStreamFinishReason(event.Response.Status, false)
			}
			usage := parseResponsesUsage(event.Response.Usage)
			state.usage = &usage
			// Per spec: responseId and serviceTier belong on the
			// stream-level ProviderMetadata.
			om := map[string]any{}
			if event.Response.ID != "" {
				om["responseId"] = event.Response.ID
			}
			if event.Response.ServiceTier != "" {
				om["serviceTier"] = event.Response.ServiceTier
			}
			if len(om) > 0 {
				if state.finishMeta == nil {
					state.finishMeta = ProviderMetadata{"openai": om}
				} else if _, ok := state.finishMeta["openai"]; ok {
					om2 := state.finishMeta["openai"].(map[string]any)
					for k, v := range om {
						om2[k] = v
					}
				}
			}
		}
	case "response.failed":
		state.finishReason = errorFinishReason()
	case "error":
		if event.Error != nil {
			// Per spec: a stream-level error event surfaces as a
			// StreamProviderError wrapping an APICallError with a
			// derived status code from the error type.
			status, _ := openAIErrorTypeStatusCode(event.Error.Type)
			parts <- StreamError{Err: &StreamProviderError{
				Provider: "openai",
				Code:     event.Error.Code,
				Message:  event.Error.Message,
				Cause: &APICallError{
					Message: event.Error.Message,
					Type:    event.Error.Type,
					Param:   event.Error.Param,
					Code:    event.Error.Code,
					Status:  status,
				},
			}}
		}
	default:
		// Unknown event types must surface instead of being silently
		// dropped. The stream continues, but the error finish latches:
		// a later completed/stop must not overwrite it.
		parts <- StreamError{Err: InvalidResponseDataError{
			Message: "Unknown event type " + strconv.Quote(event.Type),
			Data:    event.Raw,
		}}
		state.finishReason = errorFinishReason()
	}
}

// handleResponsesItemAdded handles the output_item.added event.
func handleResponsesItemAdded(parts chan<- StreamPart, state *responsesStreamState, event responsesStreamEvent) {
	if event.Item == nil {
		return
	}
	t, _ := event.Item["type"].(string)
	id, _ := event.Item["id"].(string)
	key := responsesAccumKey(event.OutputIndex, id, "")
	switch t {
	case "compaction":
		// Compaction items are emitted as a single StreamCustomPart on
		// output_item.added; the corresponding .done is a no-op (per
		// spec).
		encrypted, _ := event.Item["encrypted_content"].(string)
		parts <- StreamCustomPart{Kind: "openai.compaction", Data: map[string]any{
			"id":                id,
			"encrypted_content": encrypted,
		}}
		return
	case "message":
		phase, _ := event.Item["phase"].(string)
		// A reused output_index with a new item id closes the previous
		// accumulator before the new one starts.
		if acc, ok := state.text[key]; ok && responsesAccumRotated(acc.id, id) {
			closeResponsesTextAccum(parts, state, key)
		}
		state.text[key] = &responsesTextAccum{id: id, phase: phase}
		startPM := ProviderMetadata{}
		om := map[string]any{}
		if id != "" {
			om["itemId"] = id
		}
		if phase != "" {
			om["phase"] = phase
		}
		if len(om) > 0 {
			startPM["openai"] = om
		}
		parts <- StreamTextStart{ID: id, ProviderMetadata: startPM}
	case "reasoning":
		encrypted, _ := event.Item["encrypted_content"].(string)
		if acc, ok := state.reasoning[key]; ok && responsesAccumRotated(acc.id, id) {
			closeResponsesReasoningAccum(parts, state, key)
		}
		state.reasoning[key] = &responsesReasonAccum{id: id, active: true, encrypted: encrypted}
		startPM := ProviderMetadata{}
		om := map[string]any{}
		if id != "" {
			om["itemId"] = id
		}
		if encrypted != "" {
			om["reasoningEncryptedContent"] = encrypted
		}
		if len(om) > 0 {
			startPM["openai"] = om
		}
		parts <- StreamReasoningStart{ID: id, ProviderMetadata: startPM}
	case "mcp_approval_request":
		toolCallID := "mcp-approval-" + id
		state.approvals[id] = &responsesApprovalAccum{approvalID: id, toolCallID: toolCallID}
		name, _ := event.Item["name"].(string)
		serverLabel, _ := event.Item["server_label"].(string)
		args, _ := event.Item["arguments"].(string)
		inputBytes, _ := json.Marshal(map[string]any{
			"arguments":    args,
			"name":         name,
			"server_label": serverLabel,
		})
		parts <- StreamToolCall{ToolCallContent: ToolCallContent{
			ToolCallContentEmbed: ToolCallContentEmbed{
				ToolCallID: toolCallID,
				ToolName:   "mcp",
				Input:      inputBytes,
				ProviderMetadata: ProviderMetadata{
					"openai": map[string]any{
						"approvalRequestId": id,
					},
				},
			},
			ProviderExecuted: true,
			Dynamic:          true,
		}}
		parts <- StreamToolApprovalRequest{ApprovalID: id, ToolCallID: toolCallID}
	case "function_call", "custom_tool_call", "local_shell_call", "shell_call", "apply_patch_call", "mcp_call", "tool_search_call", "web_search_call", "file_search_call", "image_generation_call", "code_interpreter_call":
		toolName := t
		if t == "function_call" || t == "custom_tool_call" {
			if n, ok := event.Item["name"].(string); ok {
				toolName = n
			}
		}
		// If deltas arrived before the item.added (no name yet), keep
		// the buffered accumulator; just update the name and emit the
		// start (plus the buffered deltas).
		acc, exists := state.tools[key]
		if exists && responsesAccumRotated(acc.id, id) {
			closeResponsesToolAccum(parts, state, key, nil)
			acc, exists = state.tools[key]
		}
		if !exists {
			acc = &responsesToolAccum{id: id, toolName: toolName}
			state.tools[key] = acc
		} else {
			if acc.id == "" && id != "" {
				acc.id = id
			}
			if toolName != "" && (acc.toolName == "" || acc.toolName == t) {
				acc.toolName = toolName
			}
		}
		if !acc.started {
			parts <- StreamToolInputStart{ID: acc.id, ToolName: acc.toolName}
			acc.started = true
		}
	}
}

// handleResponsesItemDone handles the output_item.done event.
func handleResponsesItemDone(parts chan<- StreamPart, state *responsesStreamState, event responsesStreamEvent) {
	if event.Item == nil {
		return
	}
	t, _ := event.Item["type"].(string)
	id, _ := event.Item["id"].(string)
	switch t {
	case "message":
		// The lookup key is the actual key holding the accumulator
		// (canonical index key or a scanned item-id match), so a
		// done without output_index still closes index-keyed state.
		key, acc, _ := lookupResponsesTextAccum(state, event.OutputIndex, id, "")
		endPM := ProviderMetadata{}
		if acc != nil && len(acc.annotations) > 0 {
			endPM["openai"] = map[string]any{"annotations": acc.annotations}
		}
		parts <- StreamTextEnd{ID: id, ProviderMetadata: endPM}
		delete(state.text, key)
	case "function_call", "custom_tool_call":
		key, acc, ok := lookupResponsesToolAccum(state, event.OutputIndex, id)
		if !ok {
			return
		}
		if acc.id == "" {
			acc.id = id
		}
		input := acc.input.String()
		if input != "" && !json.Valid([]byte(input)) {
			// Function-type calls expect JSON arguments: invalid input
			// must never execute as a tool call. Close the input with
			// a StreamError and latch an error finish reason.
			parts <- StreamToolInputEnd{ID: acc.id}
			parts <- StreamError{Err: InvalidResponseDataError{Message: "Invalid tool call arguments JSON.", Data: input}}
			state.finishReason = errorFinishReason()
			delete(state.tools, key)
			return
		}
		if input == "" {
			input = "{}"
		}
		parts <- StreamToolInputEnd{ID: acc.id}
		parts <- StreamToolCall{ToolCallContent: ToolCallContent{
			ToolCallContentEmbed: ToolCallContentEmbed{
				ToolCallID: acc.id,
				ToolName:   acc.toolName,
				Input:      json.RawMessage(input),
			},
		}}
		delete(state.tools, key)
	case "apply_patch_call", "local_shell_call", "shell_call", "code_interpreter_call", "mcp_call", "tool_search_call", "web_search_call", "file_search_call", "image_generation_call":
		key, acc, ok := lookupResponsesToolAccum(state, event.OutputIndex, id)
		if !ok {
			return
		}
		if acc.id == "" {
			acc.id = id
		}
		input := acc.input.String()
		// Hosted/executed tools stream non-JSON payloads (diffs, code,
		// shell output), so invalid input is coerced to "{}" and still
		// surfaced as a tool call instead of erroring.
		if !json.Valid([]byte(input)) {
			input = "{}"
		}
		parts <- StreamToolInputEnd{ID: acc.id}
		embed := ToolCallContentEmbed{
			ToolCallID: acc.id,
			ToolName:   acc.toolName,
			Input:      json.RawMessage(input),
		}
		// shell_call and apply_patch_call need itemId round-trip.
		if t == "shell_call" || t == "apply_patch_call" {
			embed.ProviderMetadata = ProviderMetadata{
				"openai": map[string]any{"itemId": acc.id},
			}
		}
		parts <- StreamToolCall{ToolCallContent: ToolCallContent{
			ToolCallContentEmbed: embed,
		}}
		delete(state.tools, key)
	case "reasoning":
		// Conclude all remaining `active` and `can-conclude` summary
		// parts (per spec: output_item.done for reasoning flushes the
		// state).
		concludeCanConcludeReasoning(parts, state)
		if key, acc, ok := lookupResponsesReasoningAccum(state, event.OutputIndex, id, ""); ok {
			acc.active = false
			em := ProviderMetadata{}
			om := map[string]any{}
			if acc.id != "" {
				om["itemId"] = acc.id
			}
			if acc.encrypted != "" {
				om["reasoningEncryptedContent"] = acc.encrypted
			}
			if len(om) > 0 {
				em["openai"] = om
			}
			parts <- StreamReasoningEnd{ID: id, ProviderMetadata: em}
			delete(state.reasoning, key)
		}
	case "mcp_approval_request":
		delete(state.approvals, id)
	}
}

// responsesStreamFinishReason maps Responses status to AI-SDK finish reason.
func responsesStreamFinishReason(status string, hasToolCall bool) FinishReason {
	switch status {
	case "completed":
		if hasToolCall {
			return FinishReason{Unified: "tool-calls", Raw: status}
		}
		return FinishReason{Unified: "stop", Raw: status}
	case "incomplete":
		return FinishReason{Unified: "length", Raw: status}
	case "failed":
		return FinishReason{Unified: "error", Raw: status}
	case "cancelled", "canceled":
		return FinishReason{Unified: "other", Raw: status}
	}
	return FinishReason{Unified: "other", Raw: status}
}

// concludeCanConcludeReasoning emits StreamReasoningEnd for any
// reasoning accumulators that have been flagged as `can-conclude` and
// resets their state. Used to flush deferred conclusions when a new
// summary part arrives or when the reasoning output item finishes.
func concludeCanConcludeReasoning(parts chan<- StreamPart, state *responsesStreamState) {
	for _, acc := range state.reasoning {
		if acc.canConclude {
			em := ProviderMetadata{}
			om := map[string]any{}
			if acc.id != "" {
				om["itemId"] = acc.id
			}
			if acc.encrypted != "" {
				om["reasoningEncryptedContent"] = acc.encrypted
			}
			if len(om) > 0 {
				em["openai"] = om
			}
			parts <- StreamReasoningEnd{ID: responsesDisplayID(acc.id, "reasoning-0"), ProviderMetadata: em}
			acc.active = false
			acc.canConclude = false
		}
	}
}
