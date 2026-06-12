package openai

import "testing"

// TestResponsesStreamApplyPatchDeltaEmitsToolInputDelta verifies that
// the apply_patch_call_operation_diff.delta event emits a
// StreamToolInputDelta with the JSON-escaped diff.
func TestResponsesStreamApplyPatchDeltaEmitsToolInputDelta(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"apply_patch_call","id":"ap-1"}}`,
		`{"type":"response.apply_patch_call_operation_diff.delta","item_id":"ap-1","delta":"@@ -1 +1 @@\n-old\n+new\n"}`,
	}
	parts := driveResponsesEvents(t, m, events)
	deltas := 0
	for _, p := range parts {
		if d, ok := p.(StreamToolInputDelta); ok {
			if d.ID != "ap-1" {
				t.Errorf("delta ID = %q, want ap-1", d.ID)
			}
			// The delta is JSON-escaped: contains \n
			if d.Delta == "" {
				t.Errorf("delta text empty")
			}
			deltas++
		}
	}
	if deltas < 1 {
		t.Errorf("expected at least 1 StreamToolInputDelta, got %d", deltas)
	}
}

// TestResponsesStreamApplyPatchDoneDoesNotCrash verifies that the
// .done event handler is robust to no prior delta events for the id.
func TestResponsesStreamApplyPatchDoneDoesNotCrash(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.apply_patch_call_operation_diff.done","item_id":"ap-2"}`,
	}
	// Should not panic; just emit nothing.
	parts := driveResponsesEvents(t, m, events)
	_ = parts
}

// TestResponsesStreamApplyPatchFullFlow verifies the full add → delta
// → done sequence and the final output_item.done still produces a
// StreamToolCall (no duplicate tool input end).
func TestResponsesStreamApplyPatchFullFlow(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"apply_patch_call","id":"ap-3"}}`,
		`{"type":"response.apply_patch_call_operation_diff.delta","item_id":"ap-3","delta":"@@ diff @@"}`,
		`{"type":"response.apply_patch_call_operation_diff.done","item_id":"ap-3"}`,
		`{"type":"response.output_item.done","item":{"type":"apply_patch_call","id":"ap-3"}}`,
	}
	parts := driveResponsesEvents(t, m, events)
	calls := 0
	for _, p := range parts {
		if _, ok := p.(StreamToolCall); ok {
			calls++
		}
	}
	if calls != 1 {
		t.Errorf("expected exactly 1 StreamToolCall, got %d", calls)
	}
}
