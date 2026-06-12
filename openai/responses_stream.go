package openai

import (
	"context"
	"encoding/json"
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
		var event responsesStreamEvent
		if err := json.Unmarshal(raw, &event); err != nil {
			parts <- StreamError{Err: InvalidResponseDataError{Message: err.Error(), Data: string(raw)}}
			state.finishReason = errorFinishReason()
			return false
		}
		m.processResponsesEvent(parts, sresp, state, event)
		return true
	})
	// Close any open accumulators.
	for _, acc := range state.text {
		endPM := ProviderMetadata{}
		if len(acc.annotations) > 0 {
			endPM["openai"] = map[string]any{"annotations": acc.annotations}
		}
		parts <- StreamTextEnd{ID: acc.id, ProviderMetadata: endPM}
	}
	for _, acc := range state.tools {
		input := acc.input.String()
		if !json.Valid([]byte(input)) {
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
	}
	for _, acc := range state.reasoning {
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
		parts <- StreamReasoningEnd{ID: acc.id, ProviderMetadata: endPM}
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
		id := event.ItemID
		if id == "" {
			id = "txt-0"
		}
		acc, ok := state.text[id]
		if !ok {
			acc = &responsesTextAccum{id: id}
			state.text[id] = acc
		}
		acc.text.WriteString(event.Delta)
		parts <- StreamTextDelta{ID: id, Text: event.Delta}
	case "response.reasoning_summary_text.delta":
		id := event.ItemID
		if id == "" {
			id = "reasoning-0"
		}
		acc, ok := state.reasoning[id]
		if !ok {
			acc = &responsesReasonAccum{id: id, active: true}
			state.reasoning[id] = acc
		}
		acc.text.WriteString(event.Delta)
		parts <- StreamReasoningDelta{ID: id, Text: event.Delta}
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
		id := event.ItemID
		if id == "" {
			id = "reasoning-0"
		}
		acc, ok := state.reasoning[id]
		if !ok {
			acc = &responsesReasonAccum{id: id, active: true}
			state.reasoning[id] = acc
		}
		concludeImmediately := m.store == nil || *m.store
		if concludeImmediately {
			acc.active = false
			parts <- StreamReasoningEnd{ID: id}
		} else {
			acc.canConclude = true
		}
	case "response.function_call_arguments.delta":
		id := event.ItemID
		acc, ok := state.tools[id]
		if !ok {
			acc = &responsesToolAccum{id: id, toolName: event.Name}
			state.tools[id] = acc
		}
		acc.input.WriteString(event.Delta)
		// Emit Start lazily so that deltas arriving before the
		// output_item.added (which carries function.name) are still
		// surfaced with the correct tool name.
		if !acc.started && acc.toolName != "" {
			parts <- StreamToolInputStart{ID: id, ToolName: acc.toolName}
			acc.started = true
		}
		if acc.started {
			parts <- StreamToolInputDelta{ID: id, Delta: event.Delta}
		}
	case "response.custom_tool_call_input.delta":
		id := event.ItemID
		acc, ok := state.tools[id]
		if !ok {
			acc = &responsesToolAccum{id: id, toolName: event.Name}
			state.tools[id] = acc
		}
		acc.input.WriteString(event.Delta)
		if !acc.started && acc.toolName != "" {
			parts <- StreamToolInputStart{ID: id, ToolName: acc.toolName}
			acc.started = true
		}
		if acc.started {
			parts <- StreamToolInputDelta{ID: id, Delta: event.Delta}
		}
	case "response.code_interpreter_call_code.delta":
		id := event.ItemID
		acc, ok := state.tools[id]
		if !ok {
			acc = &responsesToolAccum{id: id, toolName: "code_interpreter"}
			state.tools[id] = acc
		}
		acc.input.WriteString(event.Delta)
		parts <- StreamToolInputDelta{ID: id, Delta: event.Delta}
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
		id := event.ItemID
		acc, ok := state.tools[id]
		if !ok {
			acc = &responsesToolAccum{id: id, toolName: "apply_patch"}
			state.tools[id] = acc
		}
		escaped, _ := json.Marshal(event.Delta)
		_, _ = acc.input.Write([]byte(event.Delta))
		parts <- StreamToolInputDelta{ID: id, Delta: string(escaped)}
	case "response.apply_patch_call_operation_diff.done":
		// The spec says apply_patch's output_item.done is what normally
		// closes the tool input, but the diff.done event also signals
		// completion. Mark the accumulator as ended so the next
		// output_item.done doesn't emit a duplicate.
		id := event.ItemID
		if acc, ok := state.tools[id]; ok {
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
			acc, ok := state.text[event.ItemID]
			if !ok {
				acc = &responsesTextAccum{id: event.ItemID}
				state.text[event.ItemID] = acc
			}
			acc.annotations = append(acc.annotations, event.Annotation)
		}
	case "response.completed", "response.incomplete":
		if event.Response != nil {
			state.finishReason = responsesStreamFinishReason(event.Response.Status, false)
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
			// Per spec: a stream-level error event should surface as
			// an APICallError with a derived status code from the
			// error type.
			status, _ := openAIErrorTypeStatusCode(event.Error.Type)
			parts <- StreamError{Err: &APICallError{
				Message: event.Error.Message,
				Type:    event.Error.Type,
				Param:   event.Error.Param,
				Code:    event.Error.Code,
				Status:  status,
			}}
		}
	}
}

// handleResponsesItemAdded handles the output_item.added event.
func handleResponsesItemAdded(parts chan<- StreamPart, state *responsesStreamState, event responsesStreamEvent) {
	if event.Item == nil {
		return
	}
	t, _ := event.Item["type"].(string)
	id, _ := event.Item["id"].(string)
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
		state.text[id] = &responsesTextAccum{id: id, phase: phase}
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
		state.reasoning[id] = &responsesReasonAccum{id: id, active: true, encrypted: encrypted}
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
	case "function_call", "custom_tool_call", "local_shell_call", "shell_call", "apply_patch_call", "tool_search_call", "web_search_call", "file_search_call", "image_generation_call", "code_interpreter_call":
		toolName := t
		if t == "function_call" || t == "custom_tool_call" {
			if n, ok := event.Item["name"].(string); ok {
				toolName = n
			}
		}
		// If deltas arrived before the item.added (no name yet), keep
		// the buffered accumulator; just update the name and emit the
		// start (plus the buffered deltas).
		acc, exists := state.tools[id]
		if !exists {
			acc = &responsesToolAccum{id: id, toolName: toolName}
			state.tools[id] = acc
		} else if toolName != "" && (acc.toolName == "" || acc.toolName == t) {
			acc.toolName = toolName
		}
		if !acc.started {
			parts <- StreamToolInputStart{ID: id, ToolName: acc.toolName}
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
		endPM := ProviderMetadata{}
		if acc, ok := state.text[id]; ok && len(acc.annotations) > 0 {
			endPM["openai"] = map[string]any{"annotations": acc.annotations}
		}
		parts <- StreamTextEnd{ID: id, ProviderMetadata: endPM}
		delete(state.text, id)
	case "function_call", "custom_tool_call", "apply_patch_call", "local_shell_call", "shell_call", "tool_search_call", "web_search_call", "file_search_call", "image_generation_call", "code_interpreter_call":
		acc, ok := state.tools[id]
		if !ok {
			return
		}
		input := acc.input.String()
		if !json.Valid([]byte(input)) {
			input = "{}"
		}
		parts <- StreamToolInputEnd{ID: id}
		embed := ToolCallContentEmbed{
			ToolCallID: id,
			ToolName:   acc.toolName,
			Input:      json.RawMessage(input),
		}
		// shell_call and apply_patch_call need itemId round-trip.
		if t == "shell_call" || t == "apply_patch_call" {
			embed.ProviderMetadata = ProviderMetadata{
				"openai": map[string]any{"itemId": id},
			}
		}
		parts <- StreamToolCall{ToolCallContent: ToolCallContent{
			ToolCallContentEmbed: embed,
		}}
		delete(state.tools, id)
	case "reasoning":
		// Conclude all remaining `active` and `can-conclude` summary
		// parts (per spec: output_item.done for reasoning flushes the
		// state).
		concludeCanConcludeReasoning(parts, state)
		if acc, ok := state.reasoning[id]; ok {
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
		}
		delete(state.reasoning, id)
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
	for id, acc := range state.reasoning {
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
			parts <- StreamReasoningEnd{ID: id, ProviderMetadata: em}
			acc.active = false
			acc.canConclude = false
		}
	}
}
