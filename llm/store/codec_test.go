package store

import (
	"encoding/json"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

func TestMarshalRoundTrip(t *testing.T) {
	state := &RunState{
		ID: "run-1",
		Messages: []llm.Message{
			{Role: "user", Content: []llm.Content{llm.TextContent{Text: "hello"}}},
			{Role: "assistant", Content: []llm.Content{
				llm.TextContent{Text: "thinking"},
				llm.ToolUseContent{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"go"}`)},
			}},
			{Role: "user", Content: []llm.Content{llm.ToolResultContent{ToolUseID: "call-1", Text: "result", IsError: true}}},
		},
		Metadata: map[string]string{"user": "alice"},
	}
	data, err := MarshalState(state)
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	got, err := UnmarshalState(data)
	if err != nil {
		t.Fatalf("UnmarshalState: %v", err)
	}
	if got.ID != "run-1" || got.Metadata["user"] != "alice" || len(got.Messages) != 3 {
		t.Fatalf("state = %#v", got)
	}
	use := got.Messages[1].Content[1].(llm.ToolUseContent)
	if use.ID != "call-1" || use.Name != "search" || string(use.Input) != `{"q":"go"}` {
		t.Fatalf("tool use = %#v", use)
	}
	res := got.Messages[2].Content[0].(llm.ToolResultContent)
	if res.ToolUseID != "call-1" || res.Text != "result" || !res.IsError {
		t.Fatalf("tool result = %#v", res)
	}
}

// TestMarshalSetsVersion1 asserts MarshalState stamps the current schema
// version (1) into the JSON, and that a state round-trips through Unmarshal.
func TestMarshalSetsVersion1(t *testing.T) {
	state := &RunState{
		ID:       "run-v",
		Messages: []llm.Message{{Role: "user", Content: []llm.Content{llm.TextContent{Text: "hi"}}}},
	}
	data, err := MarshalState(state)
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	var probe struct {
		Version int `json:"version"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		t.Fatalf("unmarshal probe: %v", err)
	}
	if probe.Version != 1 {
		t.Fatalf("version = %d, want 1", probe.Version)
	}
	got, err := UnmarshalState(data)
	if err != nil {
		t.Fatalf("UnmarshalState: %v", err)
	}
	if got.ID != "run-v" || len(got.Messages) != 1 {
		t.Fatalf("round-trip state = %#v", got)
	}
}

// TestUnmarshalLegacyVersion0 verifies that JSON with no version field (the
// legacy version-0 format) decodes successfully, treating a missing version
// as 0.
func TestUnmarshalLegacyVersion0(t *testing.T) {
	// Legacy payload: no "version" key present.
	payload := []byte(`{"id":"old","messages":[{"role":"user","content":[{"type":"text","text":"legacy"}]}]}`)
	state, err := UnmarshalState(payload)
	if err != nil {
		t.Fatalf("UnmarshalState legacy: %v", err)
	}
	if state.ID != "old" || len(state.Messages) != 1 {
		t.Fatalf("legacy state = %#v", state)
	}
	text, ok := state.Messages[0].Content[0].(llm.TextContent)
	if !ok || text.Text != "legacy" {
		t.Fatalf("legacy content = %#v", state.Messages[0].Content[0])
	}
}

func TestUnmarshalUnknownType(t *testing.T) {
	_, err := UnmarshalState([]byte(`{"id":"x","messages":[{"role":"user","content":[{"type":"bogus"}]}]}`))
	if err == nil {
		t.Fatal("expected error for unknown content type")
	}
}

// TestReasoningAndImageRoundTrip verifies the codec round-trips the new
// reasoning and image content tags added in spec §1.2.
func TestReasoningAndImageRoundTrip(t *testing.T) {
	imgBytes := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A}
	state := &RunState{
		ID: "run-2",
		Messages: []llm.Message{
			{Role: "assistant", Content: []llm.Content{
				llm.ReasoningContent{Text: "I should check the file first."},
				llm.TextContent{Text: "Let me look."},
			}},
			{Role: "user", Content: []llm.Content{
				llm.ImageContent{Source: llm.ImageInlineSource{Data: imgBytes}, MIME: "image/png"},
				llm.ImageContent{Source: llm.ImageURLSource{URL: "https://example.com/a.png"}, MIME: "image/png"},
			}},
		},
	}
	data, err := MarshalState(state)
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	got, err := UnmarshalState(data)
	if err != nil {
		t.Fatalf("UnmarshalState: %v", err)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages len = %d, want 2", len(got.Messages))
	}
	rc, ok := got.Messages[0].Content[0].(llm.ReasoningContent)
	if !ok || rc.Text != "I should check the file first." {
		t.Fatalf("reasoning = %#v, want ReasoningContent", got.Messages[0].Content[0])
	}
	// Inline image: bytes must round-trip exactly.
	inline, ok := got.Messages[1].Content[0].(llm.ImageContent)
	if !ok || inline.MIME != "image/png" {
		t.Fatalf("inline image = %#v, want ImageContent{image/png}", got.Messages[1].Content[0])
	}
	inlineSrc, ok := inline.Source.(llm.ImageInlineSource)
	if !ok {
		t.Fatalf("inline source = %#v, want ImageInlineSource", inline.Source)
	}
	if string(inlineSrc.Data) != string(imgBytes) {
		t.Fatalf("inline data = %v, want %v", inlineSrc.Data, imgBytes)
	}
	// URL image: URL must round-trip.
	urlImg, ok := got.Messages[1].Content[1].(llm.ImageContent)
	if !ok || urlImg.MIME != "image/png" {
		t.Fatalf("url image = %#v, want ImageContent{image/png}", got.Messages[1].Content[1])
	}
	urlSrc, ok := urlImg.Source.(llm.ImageURLSource)
	if !ok || urlSrc.URL != "https://example.com/a.png" {
		t.Fatalf("url source = %#v, want ImageURLSource", urlImg.Source)
	}
}

// TestCodecBackwardCompatExistingTags verifies existing tags still decode
// unchanged after the reasoning/image additions (purely additive, spec §1.2).
func TestCodecBackwardCompatExistingTags(t *testing.T) {
	// A payload with only legacy tags must decode without touching the new ones.
	payload := []byte(`{"id":"r","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"tool_use","id":"c","name":"t","input":{}},{"type":"tool_result","tool_use_id":"c","text":"ok","is_error":false}]}]}`)
	state, err := UnmarshalState(payload)
	if err != nil {
		t.Fatalf("UnmarshalState: %v", err)
	}
	if len(state.Messages[0].Content) != 3 {
		t.Fatalf("content len = %d, want 3", len(state.Messages[0].Content))
	}
	if _, ok := state.Messages[0].Content[0].(llm.TextContent); !ok {
		t.Fatalf("content 0 = %#v, want TextContent", state.Messages[0].Content[0])
	}
}
