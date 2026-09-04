package openai

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// TestResponsesStreamOutputIndexRotationStaysCoherent verifies the R8 fix:
// accumulators correlate by output_index, and a reused index with a new
// item id closes the old accumulator before starting the new one. This
// Copilot-style rotation fixture (same output_index, rotating item_ids)
// must yield a single coherent part sequence with no crash and no
// orphaned accumulator.
func TestResponsesStreamOutputIndexRotationStaysCoherent(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := driveResponsesEvents(t, m, []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg-1"}}`,
		`{"type":"response.output_text.delta","output_index":0,"item_id":"msg-1","delta":"hel"}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg-2"}}`,
		`{"type":"response.output_text.delta","output_index":0,"item_id":"msg-2","delta":"lo"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg-2"}}`,
	})

	var seq []string
	var text strings.Builder
	for _, p := range parts {
		switch v := p.(type) {
		case StreamTextStart:
			seq = append(seq, "start:"+v.ID)
		case StreamTextDelta:
			seq = append(seq, "delta:"+v.ID)
			text.WriteString(v.Text)
		case StreamTextEnd:
			seq = append(seq, "end:"+v.ID)
		default:
			seq = append(seq, "other")
		}
	}
	want := []string{"start:msg-1", "delta:msg-1", "end:msg-1", "start:msg-2", "delta:msg-2", "end:msg-2"}
	if strings.Join(seq, ",") != strings.Join(want, ",") {
		t.Fatalf("sequence = %v, want %v", seq, want)
	}
	if text.String() != "hello" {
		t.Fatalf("combined text = %q, want %q", text.String(), "hello")
	}
}

// TestResponsesStreamUnknownEventTypeEmitsErrorAndLatchesFinish verifies
// the R16 fix: an unknown event.type surfaces as a StreamError carrying
// an InvalidResponseDataError that names the type with the raw payload,
// and a later response.completed must not overwrite the error finish.
func TestResponsesStreamUnknownEventTypeEmitsErrorAndLatchesFinish(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	parts := make(chan StreamPart, 64)
	sresp := &StreamResponse{}
	state := newResponsesStreamState()

	raw := `{"type":"response.future_thing","output_index":0}`
	var ev responsesStreamEvent
	if err := json.Unmarshal([]byte(raw), &ev); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	ev.Raw = raw
	m.processResponsesEvent(parts, sresp, state, ev)
	m.processResponsesEvent(parts, sresp, state, responsesStreamEvent{
		Type:     "response.completed",
		Response: &responsesStreamResp{Status: "completed"},
	})
	close(parts)

	var errs []StreamError
	for p := range parts {
		if se, ok := p.(StreamError); ok {
			errs = append(errs, se)
		}
	}
	if len(errs) != 1 {
		t.Fatalf("expected 1 StreamError, got %d", len(errs))
	}
	inv, ok := errs[0].Err.(InvalidResponseDataError)
	if !ok {
		t.Fatalf("error = %T (%v), want InvalidResponseDataError", errs[0].Err, errs[0].Err)
	}
	if !strings.Contains(inv.Message, "response.future_thing") {
		t.Errorf("error message %q does not name the event type", inv.Message)
	}
	if data, _ := inv.Data.(string); data != raw {
		t.Errorf("error data = %q, want raw payload", data)
	}
	if state.finishReason != errorFinishReason() {
		t.Errorf("finish reason = %+v, want latched error", state.finishReason)
	}
}

// drainResponsesStream runs raw SSE through runResponsesStream and returns
// every emitted part in order.
func drainResponsesStream(t *testing.T, sseBody string) []StreamPart {
	t.Helper()
	parts := make(chan StreamPart, 64)
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	go func() {
		m.runResponsesStream(context.Background(), makeStreamResp(sseBody), parts, &StreamResponse{}, nil, ResponsesStreamOptions{})
	}()
	var out []StreamPart
	for p := range parts {
		out = append(out, p)
	}
	return out
}

func streamErrorOf(parts []StreamPart) StreamError {
	for _, p := range parts {
		if se, ok := p.(StreamError); ok {
			return se
		}
	}
	return StreamError{}
}

// TestResponsesStreamChoicesChunkWrongSurface verifies that a Chat
// Completions SSE chunk (choices-shaped) fed to the Responses parser
// surfaces an actionable wrong-API-surface APICallError.
func TestResponsesStreamChoicesChunkWrongSurface(t *testing.T) {
	sseBody := "data: {\"id\":\"x\",\"created\":10,\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
	parts := drainResponsesStream(t, sseBody)
	got := streamErrorOf(parts)
	if got.Err == nil {
		t.Fatalf("expected StreamError, got: %v", parts)
	}
	var apiErr *APICallError
	if !errors.As(got.Err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", got.Err, got.Err)
	}
	if !strings.Contains(strings.ToLower(apiErr.Message), "chat") {
		t.Errorf("message should point at the chat surface, got: %q", apiErr.Message)
	}
}

// TestIsChoicesShapedChunk is a table test for the choices-shape probe.
func TestIsChoicesShapedChunk(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		want bool
	}{
		{"chat delta chunk", `{"id":"x","choices":[{"delta":{"content":"hi"}}]}`, true},
		{"chat message chunk", `{"id":"x","choices":[{"message":{"content":"hi"}}]}`, true},
		{"chat error chunk", `{"id":"x","choices":[],"error":{"message":"boom"}}`, false},
		{"null choices", `{"id":"x","choices":null}`, false},
		{"responses event", `{"type":"response.completed","response":{"status":"completed"}}`, false},
		{"responses delta", `{"type":"response.output_text.delta","delta":"hi"}`, false},
		{"invalid json", `not json`, false},
		{"empty object", `{}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := isChoicesShapedChunk([]byte(tc.raw)); got != tc.want {
				t.Errorf("isChoicesShapedChunk = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestResponsesStreamErrorSurfacesStreamProviderError verifies that a
// mid-stream provider error event surfaces a *StreamProviderError with
// provider/code/message fields, wrapping the derived *APICallError.
func TestResponsesStreamErrorSurfacesStreamProviderError(t *testing.T) {
	sseBody := "data: {\"type\":\"error\",\"error\":{\"message\":\"forbidden\",\"type\":\"permission_error\",\"code\":\"forbidden\"}}\n\n"
	parts := drainResponsesStream(t, sseBody)
	got := streamErrorOf(parts)
	if got.Err == nil {
		t.Fatal("expected StreamError, got none")
	}
	spe, ok := got.Err.(*StreamProviderError)
	if !ok {
		t.Fatalf("expected *StreamProviderError, got %T: %v", got.Err, got.Err)
	}
	if spe.Provider != "openai" {
		t.Errorf("Provider = %q, want openai", spe.Provider)
	}
	if spe.Message != "forbidden" {
		t.Errorf("Message = %q, want forbidden", spe.Message)
	}
	if spe.Code != "forbidden" {
		t.Errorf("Code = %v, want forbidden", spe.Code)
	}
	var apiErr *APICallError
	if !errors.As(got.Err, &apiErr) {
		t.Fatalf("expected errors.As to find *APICallError, got %T", got.Err)
	}
	if apiErr.Status != 403 {
		t.Errorf("Status = %d, want 403 (derived from permission_error)", apiErr.Status)
	}
	var spe2 *StreamProviderError
	if !errors.As(got.Err, &spe2) {
		t.Fatalf("expected errors.As to find *StreamProviderError")
	}
	if spe2 != spe {
		t.Errorf("errors.As identity mismatch")
	}
}
