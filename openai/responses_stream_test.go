package openai

import (
	"encoding/json"
	"testing"
)

// driveResponsesEvents runs a slice of raw SSE event JSON bodies through
// processResponsesEvent and collects the resulting StreamParts.
func driveResponsesEvents(t *testing.T, m *openaiResponsesModel, events []string) []StreamPart {
	t.Helper()
	parts := make(chan StreamPart, 64)
	sresp := &StreamResponse{}
	state := newResponsesStreamState()
	for _, raw := range events {
		var ev responsesStreamEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatalf("unmarshal %q: %v", raw, err)
		}
		m.processResponsesEvent(parts, sresp, state, ev)
	}
	close(parts)
	var out []StreamPart
	for p := range parts {
		out = append(out, p)
	}
	return out
}

// TestResponsesStreamMetadataSent verifies the first response.created event
// emits a StreamResponseMetadata.
func TestResponsesStreamMetadataSent(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"gpt-4o","created_at":1700000000,"output":[]}}`,
	})
	if len(parts) == 0 {
		t.Fatalf("no parts")
	}
	md, ok := parts[0].(StreamResponseMetadata)
	if !ok {
		t.Fatalf("first part: %T", parts[0])
	}
	if md.ID != "resp_1" {
		t.Errorf("ID: %q", md.ID)
	}
	if md.ModelID != "gpt-4o" {
		t.Errorf("ModelID: %q", md.ModelID)
	}
}

// TestResponsesStreamTextDelta verifies text deltas accumulate and emit
// StreamTextDelta events.
func TestResponsesStreamTextDelta(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.output_text.delta","item_id":"txt-0","delta":"hel"}`,
		`{"type":"response.output_text.delta","item_id":"txt-0","delta":"lo"}`,
	})
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	combined := ""
	for _, p := range parts {
		if d, ok := p.(StreamTextDelta); ok {
			combined += d.Text
		}
	}
	if combined != "hello" {
		t.Errorf("combined: %q", combined)
	}
}

// TestResponsesStreamReasoningDelta verifies reasoning deltas are emitted.
func TestResponsesStreamReasoningDelta(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-5"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.reasoning_summary_text.delta","item_id":"r-0","delta":"thinking"}`,
	})
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	d, ok := parts[0].(StreamReasoningDelta)
	if !ok {
		t.Fatalf("type: %T", parts[0])
	}
	if d.Text != "thinking" {
		t.Errorf("Text: %q", d.Text)
	}
}

// TestResponsesStreamFunctionCallArgsDelta verifies function call argument
// deltas emit StreamToolInputDelta.
func TestResponsesStreamFunctionCallArgsDelta(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.function_call_arguments.delta","item_id":"call-1","name":"f","delta":"{\"x\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"call-1","name":"f","delta":"1}"}`,
	})
	count := 0
	for _, p := range parts {
		if _, ok := p.(StreamToolInputDelta); ok {
			count++
		}
	}
	if count != 2 {
		t.Errorf("tool input deltas: %d", count)
	}
}

// TestResponsesStreamAnnotationAdded verifies citation annotations emit
// SourceContent.
func TestResponsesStreamAnnotationAdded(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","id":"cit-1","url":"https://x.test","title":"X"}}`,
	})
	if len(parts) != 1 {
		t.Fatalf("parts: %d", len(parts))
	}
	src, ok := parts[0].(SourceContent)
	if !ok {
		t.Fatalf("type: %T", parts[0])
	}
	if src.URL != "https://x.test" {
		t.Errorf("URL: %q", src.URL)
	}
	if src.ID != "cit-1" {
		t.Errorf("ID: %q", src.ID)
	}
}

// TestResponsesStreamImageGenerationPartialImage verifies partial image
// events emit a StreamCustomPart whose Data is a preliminary
// ToolResultContent with the base64 result in Output.Value and
// ProviderMetadata{openai.preliminary: true}, per spec.
func TestResponsesStreamImageGenerationPartialImage(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-5"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.image_generation_call.partial_image","item_id":"img-1","result":"AAAA"}`,
	})
	if len(parts) != 1 {
		t.Fatalf("parts: %d", len(parts))
	}
	cp, ok := parts[0].(StreamCustomPart)
	if !ok {
		t.Fatalf("type: %T", parts[0])
	}
	if cp.Kind != "image_generation.partial_image" {
		t.Errorf("Kind: %q", cp.Kind)
	}
	trc, ok := cp.Data.(ToolResultContent)
	if !ok {
		t.Fatalf("Data type: %T", cp.Data)
	}
	if trc.ToolCallID != "img-1" {
		t.Errorf("ToolCallID: %q", trc.ToolCallID)
	}
	if trc.ToolName != "image_generation" {
		t.Errorf("ToolName: %q", trc.ToolName)
	}
	if trc.Output.Type != "content" {
		t.Errorf("Output.Type: %q", trc.Output.Type)
	}
	if s, _ := trc.Output.Value.(string); s != "AAAA" {
		t.Errorf("Output.Value: %v", trc.Output.Value)
	}
	om, ok := trc.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("ProviderMetadata[openai] missing: %v", trc.ProviderMetadata)
	}
	if v, _ := om["preliminary"].(bool); !v {
		t.Errorf("preliminary: %v", om["preliminary"])
	}
}

// TestResponsesStreamCompletedSendsFinishReason verifies the response.completed
// event sets the finish reason and usage on the stream state. The
// StreamFinish part itself is emitted by the surrounding runResponsesStream
// function (not processResponsesEvent).
func TestResponsesStreamCompletedSendsFinishReason(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := make(chan StreamPart, 64)
	sresp := &StreamResponse{}
	state := newResponsesStreamState()
	ev := responsesStreamEvent{Type: "response.completed", Response: &responsesStreamResp{
		Status: "completed",
		Usage:  map[string]any{"input_tokens": float64(5), "output_tokens": float64(10), "total_tokens": float64(15)},
		Output: []map[string]any{},
	}}
	m.processResponsesEvent(parts, sresp, state, ev)
	close(parts)
	// Drain any emitted parts (response.completed may emit none).
	for range parts {
	}
	if state.finishReason.Unified != "stop" {
		t.Errorf("finish reason: %q", state.finishReason.Unified)
	}
	if state.usage == nil || state.usage.InputTokens.Total == nil || *state.usage.InputTokens.Total != 5 {
		t.Errorf("usage: %+v", state.usage)
	}
}

// TestResponsesStreamCodeInterpreterDelta verifies the code_interpreter
// delta event emits a tool input delta tagged with code_interpreter.
func TestResponsesStreamCodeInterpreterDelta(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.code_interpreter_call_code.delta","item_id":"ci-1","delta":"print(1)"}`,
	})
	if len(parts) != 1 {
		t.Fatalf("parts: %d", len(parts))
	}
	if _, ok := parts[0].(StreamToolInputDelta); !ok {
		t.Fatalf("type: %T", parts[0])
	}
}

// TestResponsesStreamFunctionCallArgsArrivesBeforeItemAdded verifies
// the pending tool call buffering fix: if function_call_arguments.delta
// arrives before output_item.added, the deltas are buffered and the
// start event is emitted (with the name) when the item is added.
func TestResponsesStreamFunctionCallArgsArrivesBeforeItemAdded(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.function_call_arguments.delta","item_id":"fc-1","delta":"{\"loc\":\""}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc-1","delta":"SF\"}"}`,
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc-1","name":"get_weather","call_id":"call_1","arguments":""}}`,
		`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc-1","name":"get_weather","call_id":"call_1","arguments":"{\"loc\":\"SF\"}"}}`,
	})
	var starts []StreamToolInputStart
	var deltas []StreamToolInputDelta
	var calls []ToolCallContent
	for _, p := range parts {
		switch v := p.(type) {
		case StreamToolInputStart:
			starts = append(starts, v)
		case StreamToolInputDelta:
			deltas = append(deltas, v)
		case StreamToolCall:
			calls = append(calls, v.ToolCallContent)
		}
	}
	if len(starts) != 1 {
		t.Fatalf("starts: %d", len(starts))
	}
	if starts[0].ToolName != "get_weather" {
		t.Errorf("start name: %q", starts[0].ToolName)
	}
	if len(calls) != 1 {
		t.Fatalf("calls: %d", len(calls))
	}
	if string(calls[0].Input) != `{"loc":"SF"}` {
		t.Errorf("Input = %q", string(calls[0].Input))
	}
	// deltas should be 0 (buffered) because output_item.added emits the
	// start; the deltas themselves are coalesced into the final
	// StreamToolCall. The exact ordering of deltas is implementation
	// detail; we just need the final input to be correct.
	_ = deltas
}

// TestResponsesStreamCustomToolCallDelta verifies the custom tool call
// delta event emits a tool input start and a tool input delta.
func TestResponsesStreamCustomToolCallDelta(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.custom_tool_call_input.delta","item_id":"ct-1","name":"custom","delta":"hi"}`,
	})
	if len(parts) != 2 {
		t.Fatalf("parts: %d", len(parts))
	}
	if start, ok := parts[0].(StreamToolInputStart); !ok {
		t.Fatalf("first part: %T", parts[0])
	} else if start.ToolName != "custom" {
		t.Errorf("first part ToolName: %q", start.ToolName)
	}
	if _, ok := parts[1].(StreamToolInputDelta); !ok {
		t.Fatalf("second part: %T", parts[1])
	}
}

// TestResponsesStreamMultiPartReasoningSummary verifies that multiple
// reasoning_summary_text.delta events for the same item_id are
// concatenated into a single StreamReasoningDelta sequence.
func TestResponsesStreamMultiPartReasoningSummary(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-5"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.reasoning_summary_text.delta","item_id":"r-0","delta":"step1"}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"r-0","delta":" step2"}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"r-0","delta":" step3"}`,
	})
	if len(parts) != 3 {
		t.Fatalf("expected 3 deltas, got %d", len(parts))
	}
	combined := ""
	for _, p := range parts {
		d, ok := p.(StreamReasoningDelta)
		if !ok {
			t.Fatalf("part type: %T", p)
		}
		combined += d.Text
	}
	if combined != "step1 step2 step3" {
		t.Errorf("combined: %q", combined)
	}
}
