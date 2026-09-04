package openai

import (
	"encoding/json"
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
