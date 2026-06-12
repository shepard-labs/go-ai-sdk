package openai

import "testing"

// TestResponsesStreamCompactionItemAdded verifies that a compaction
// output item emits a StreamCustomPart with Kind "openai.compaction"
// and the encrypted content in Data.
func TestResponsesStreamCompactionItemAdded(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"compaction","id":"cmp-1","encrypted_content":"enc-blob"}}`,
	}
	parts := driveResponsesEvents(t, m, events)
	found := false
	for _, p := range parts {
		if c, ok := p.(StreamCustomPart); ok && c.Kind == "openai.compaction" {
			found = true
			data, _ := c.Data.(map[string]any)
			if data["id"] != "cmp-1" {
				t.Errorf("id = %v", data["id"])
			}
			if data["encrypted_content"] != "enc-blob" {
				t.Errorf("encrypted_content = %v", data["encrypted_content"])
			}
		}
	}
	if !found {
		t.Errorf("expected StreamCustomPart with Kind=openai.compaction, got: %v", parts)
	}
}

// TestResponsesStreamCompactionItemDoneNoOp verifies that the
// output_item.done for compaction does NOT emit a second StreamCustomPart
// (per spec: it's a no-op since .add already emitted it).
func TestResponsesStreamCompactionItemDoneNoOp(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"compaction","id":"cmp-2","encrypted_content":"enc"}}`,
		`{"type":"response.output_item.done","item":{"type":"compaction","id":"cmp-2"}}`,
	}
	parts := driveResponsesEvents(t, m, events)
	count := 0
	for _, p := range parts {
		if c, ok := p.(StreamCustomPart); ok && c.Kind == "openai.compaction" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected exactly 1 compaction StreamCustomPart, got %d", count)
	}
}

// TestResponsesStreamCompactionItemAddedNoEncrypted verifies that
// compaction items without encrypted_content still emit the custom
// part (with empty encrypted_content).
func TestResponsesStreamCompactionItemAddedNoEncrypted(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"compaction","id":"cmp-3"}}`,
	}
	parts := driveResponsesEvents(t, m, events)
	found := false
	for _, p := range parts {
		if c, ok := p.(StreamCustomPart); ok && c.Kind == "openai.compaction" {
			found = true
			data, _ := c.Data.(map[string]any)
			if data["id"] != "cmp-3" {
				t.Errorf("id = %v", data["id"])
			}
			if data["encrypted_content"] != "" {
				t.Errorf("encrypted_content should be empty, got: %v", data["encrypted_content"])
			}
		}
	}
	if !found {
		t.Errorf("expected StreamCustomPart for compaction")
	}
}
