package openai

import "testing"

// TestChatStreamTruncatedArgsProduceNoToolCall verifies that a truncated
// '{"a":1' accumulation split across chunks must never execute as a tool
// call. Flush closes the input with a StreamError and latches an error
// finish reason instead.
func TestChatStreamTruncatedArgsProduceNoToolCall(t *testing.T) {
	ch, m := newTestStreamParts()
	state := newChatStreamState()
	id := "call-1"
	name := "get_data"
	idx := 0
	frag1 := `{"a":`
	frag2 := `1`
	m.processChatStreamDelta(ch, state, chatStreamDelta{
		ToolCalls: []chatStreamToolCall{{
			ID:    &id,
			Index: &idx,
			Function: chatStreamFunction{
				Name:      &name,
				Arguments: &frag1,
			},
		}},
	})
	m.processChatStreamDelta(ch, state, chatStreamDelta{
		ToolCalls: []chatStreamToolCall{{
			ID:    &id,
			Index: &idx,
			Function: chatStreamFunction{
				Arguments: &frag2,
			},
		}},
	})
	state.flushChatStreamState(ch)
	close(ch)
	parts := drainParts(ch)

	for _, p := range parts {
		if _, ok := p.(StreamToolCall); ok {
			t.Fatalf("truncated input must not produce StreamToolCall")
		}
	}
	var sawErr, sawEnd bool
	for _, p := range parts {
		switch e := p.(type) {
		case StreamToolInputEnd:
			if e.ID == id {
				sawEnd = true
			}
		case StreamError:
			if _, ok := e.Err.(InvalidResponseDataError); !ok {
				t.Fatalf("expected InvalidResponseDataError, got %T", e.Err)
			}
			sawErr = true
		}
	}
	if !sawEnd {
		t.Errorf("expected StreamToolInputEnd for %q", id)
	}
	if !sawErr {
		t.Errorf("expected StreamError for truncated input")
	}
	if state.finishReason != errorFinishReason() {
		t.Errorf("finish reason = %+v, want error", state.finishReason)
	}
}

// TestChatStreamGarbageSuffixProducesNoToolCall verifies that a parsable
// prefix '{"a":1}' followed by '<garbage>' must not execute the prefix as
// a tool call while silently dropping the garbage. The whole accumulation
// is rejected with a StreamError and no ToolCall.
func TestChatStreamGarbageSuffixProducesNoToolCall(t *testing.T) {
	ch, m := newTestStreamParts()
	state := newChatStreamState()
	id := "call-1"
	name := "get_data"
	idx := 0
	prefix := `{"a":1}`
	garbage := `<garbage>`
	m.processChatStreamDelta(ch, state, chatStreamDelta{
		ToolCalls: []chatStreamToolCall{{
			ID:    &id,
			Index: &idx,
			Function: chatStreamFunction{
				Name:      &name,
				Arguments: &prefix,
			},
		}},
	})
	m.processChatStreamDelta(ch, state, chatStreamDelta{
		ToolCalls: []chatStreamToolCall{{
			ID:    &id,
			Index: &idx,
			Function: chatStreamFunction{
				Arguments: &garbage,
			},
		}},
	})
	state.flushChatStreamState(ch)
	close(ch)
	parts := drainParts(ch)

	for _, p := range parts {
		if call, ok := p.(StreamToolCall); ok {
			t.Fatalf("garbage-suffixed input must not produce StreamToolCall, got input %q", string(call.Input))
		}
	}
	var sawErr bool
	for _, p := range parts {
		if e, ok := p.(StreamError); ok {
			if _, ok := e.Err.(InvalidResponseDataError); !ok {
				t.Fatalf("expected InvalidResponseDataError, got %T", e.Err)
			}
			sawErr = true
		}
	}
	if !sawErr {
		t.Errorf("expected StreamError for garbage-suffixed input")
	}
	if state.finishReason != errorFinishReason() {
		t.Errorf("finish reason = %+v, want error", state.finishReason)
	}
}
