package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
)

// runResponsesStreamSSE drives a list of raw SSE event JSON bodies
// through the openaiResponsesModel streaming path and returns the
// resulting StreamParts in order. Tests that exercise store-related
// behavior use the model's `store` field directly.
func runResponsesStreamSSE(t *testing.T, m *openaiResponsesModel, events []string) []StreamPart {
	t.Helper()
	// The test runs processResponsesEvent directly (synchronously)
	// rather than through a goroutine.
	parts := make(chan StreamPart, 128)
	sresp := &StreamResponse{}
	state := newResponsesStreamState()
	for _, raw := range events {
		var ev responsesStreamEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatalf("unmarshal: %v", err)
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

// TestResponsesSummaryPartDoneConcludesImmediatelyStoreTrue verifies
// that with store=true, a reasoning_summary_part.done event emits a
// StreamReasoningEnd for that part.
func TestResponsesSummaryPartDoneConcludesImmediatelyStoreTrue(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o", store: boolPtr(true)}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"reasoning","id":"r1"}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"r1","summary_index":0,"delta":"hmm"}`,
		`{"type":"response.reasoning_summary_part.done","item_id":"r1","summary_index":0}`,
	}
	parts := runResponsesStreamSSE(t, m, events)
	endCount := 0
	for _, p := range parts {
		if e, ok := p.(StreamReasoningEnd); ok && e.ID == "r1" {
			endCount++
		}
	}
	if endCount != 1 {
		t.Errorf("expected 1 StreamReasoningEnd, got %d", endCount)
	}
}

// TestResponsesSummaryPartDoneDeferredStoreFalse verifies that with
// store=false, the done event sets can-conclude (does not emit End).
// End is emitted when the next summary part starts (or on item done).
func TestResponsesSummaryPartDoneDeferredStoreFalse(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o", store: boolPtr(false)}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"reasoning","id":"r1"}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"r1","summary_index":0,"delta":"thinking"}`,
		`{"type":"response.reasoning_summary_part.done","item_id":"r1","summary_index":0}`,
		`{"type":"response.reasoning_summary_part.added","summary_index":1,"item_id":"r1"}`,
	}
	parts := runResponsesStreamSSE(t, m, events)
	found := false
	for _, p := range parts {
		if e, ok := p.(StreamReasoningEnd); ok && e.ID == "r1" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected deferred StreamReasoningEnd after part 1 added, got: %v", parts)
	}
}

// TestResponsesSummaryPartAddedConcludesAllCanConclude verifies that
// a new summary part with index > 0 concludes ALL outstanding
// can-conclude parts.
func TestResponsesSummaryPartAddedConcludesAllCanConclude(t *testing.T) {
	// Two reasoning items, both with can-conclude set.
	state := newResponsesStreamState()
	parts := make(chan StreamPart, 64)
	concludeCanConcludeReasoning(parts, state) // no-op
	// Manually set up state with two can-conclude accumulators
	state.reasoning["r1"] = &responsesReasonAccum{id: "r1", canConclude: true}
	state.reasoning["r2"] = &responsesReasonAccum{id: "r2", canConclude: true}
	concludeCanConcludeReasoning(parts, state)
	close(parts)
	ended := map[string]bool{}
	for p := range parts {
		if e, ok := p.(StreamReasoningEnd); ok {
			ended[e.ID] = true
		}
	}
	if !ended["r1"] || !ended["r2"] {
		t.Errorf("expected both r1 and r2 to be ended, got: %v", ended)
	}
}

// TestResponsesItemDoneConcludesReasoning verifies that the
// output_item.done for a reasoning item emits a StreamReasoningEnd.
func TestResponsesItemDoneConcludesReasoning(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o", store: boolPtr(true)}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"reasoning","id":"r1"}}`,
		`{"type":"response.output_item.done","item":{"type":"reasoning","id":"r1"}}`,
	}
	parts := runResponsesStreamSSE(t, m, events)
	found := false
	for _, p := range parts {
		if e, ok := p.(StreamReasoningEnd); ok && e.ID == "r1" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected StreamReasoningEnd on output_item.done, got: %v", parts)
	}
}

// TestResponsesDoStreamCapturesStore verifies that DoStream sets the
// model's store field, used by streaming event handlers.
func TestResponsesDoStreamCapturesStore(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	m := p.Responses("gpt-4o").(*openaiResponsesModel)
	if m.store != nil {
		t.Errorf("store should be nil before DoStream, got: %v", *m.store)
	}
	storeFalse := false
	_, err := m.DoStream(context.Background(), ResponsesStreamOptions{
		ResponsesGenerateOptions: ResponsesGenerateOptions{
			Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
			Store:    &storeFalse,
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	if m.store == nil || *m.store != false {
		t.Errorf("store not captured, got: %v", m.store)
	}
}
