package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestResponsesDropsNonOpenAIReasoning verifies that a ReasoningContent
// with no ItemID and no EncryptedContent is dropped with a warning.
func TestResponsesDropsNonOpenAIReasoning(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{AssistantMessage{Content: []AssistantContent{
			ReasoningContent{Summary: []string{"thinking..."}},
		}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	found := false
	for _, w := range res.Warnings {
		if w.Message == "Non-OpenAI reasoning parts are not supported. Skipping." {
			found = true
		}
	}
	if !found {
		t.Errorf("expected reasoning drop warning, got: %+v", res.Warnings)
	}
}

// TestResponsesKeepsReasoningWithEncryptedContent verifies that a
// ReasoningContent with EncryptedContent is preserved (not dropped).
func TestResponsesKeepsReasoningWithEncryptedContent(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{AssistantMessage{Content: []AssistantContent{
			ReasoningContent{EncryptedContent: "encblob", Summary: []string{"ok"}},
		}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	for _, w := range res.Warnings {
		if w.Message == "Non-OpenAI reasoning parts are not supported. Skipping." {
			t.Errorf("unexpected reasoning drop: %+v", w)
		}
	}
	body := decodeRequestBody(t, res.Request.Body)
	input, _ := body["input"].([]any)
	hasReasoning := false
	for _, item := range input {
		m, _ := item.(map[string]any)
		if m["type"] == "reasoning" {
			hasReasoning = true
		}
	}
	if !hasReasoning {
		t.Errorf("expected reasoning item in input, got: %v", input)
	}
}

// TestResponsesDropsUnencryptedReasoningWhenStoreFalse verifies the
// post-processing step that drops reasoning without encrypted_content
// when store=false.
func TestResponsesDropsUnencryptedReasoningWhenStoreFalse(t *testing.T) {
	storeFalse := false
	in := []any{
		map[string]any{"type": "reasoning", "summary": []any{}},
		map[string]any{"type": "reasoning", "encrypted_content": "x", "summary": []any{}},
	}
	out, dropped := dropUnencryptedReasoning(in)
	if !dropped {
		t.Errorf("expected dropped=true")
	}
	if len(out) != 1 {
		t.Errorf("expected 1 item, got %d", len(out))
	}
	m, _ := out[0].(map[string]any)
	if m["encrypted_content"] != "x" {
		t.Errorf("kept item is wrong: %v", m)
	}
	_ = storeFalse
}

// TestDropUnencryptedReasoningEmpty verifies the helper on empty input.
func TestDropUnencryptedReasoningEmpty(t *testing.T) {
	out, dropped := dropUnencryptedReasoning(nil)
	if dropped {
		t.Errorf("expected dropped=false")
	}
	if len(out) != 0 {
		t.Errorf("expected empty output, got: %v", out)
	}
}
