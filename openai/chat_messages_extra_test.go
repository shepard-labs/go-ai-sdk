package openai

import (
	"encoding/json"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestExtraConvertAssistantMessageTextOnly verifies that an assistant message
// with only text content emits a string content.
func TestExtraConvertAssistantMessageTextOnly(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertAssistantMessage(AssistantMessage{Content: []AssistantContent{
		TextContent{Text: "hi there"},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["role"] != "assistant" {
		t.Errorf("role: %v", out["role"])
	}
	if out["content"] != "hi there" {
		t.Errorf("content: %v", out["content"])
	}
	if _, has := out["tool_calls"]; has {
		t.Errorf("tool_calls should not be set")
	}
}

// TestExtraConvertAssistantMessageToolCallsAndText verifies that an assistant
// message carrying both text and tool calls emits content + tool_calls.
func TestExtraConvertAssistantMessageToolCallsAndText(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertAssistantMessage(AssistantMessage{Content: []AssistantContent{
		TextContent{Text: "Let me check..."},
		ToolCallContent{ToolCallContentEmbed: ToolCallContentEmbed{ToolCallID: "c1", ToolName: "search", Input: json.RawMessage(`{"q":"x"}`)}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["content"] != "Let me check..." {
		t.Errorf("content: %v", out["content"])
	}
	tcs, _ := out["tool_calls"].([]map[string]any)
	if len(tcs) != 1 {
		t.Fatalf("tool_calls: %v", out["tool_calls"])
	}
	if tcs[0]["id"] != "c1" {
		t.Errorf("id: %v", tcs[0])
	}
	if tcs[0]["type"] != "function" {
		t.Errorf("type: %v", tcs[0])
	}
	fn, _ := tcs[0]["function"].(map[string]any)
	if fn["name"] != "search" {
		t.Errorf("name: %v", fn)
	}
	if fn["arguments"] != `{"q":"x"}` {
		t.Errorf("arguments: %v", fn["arguments"])
	}
}

// TestExtraConvertAssistantMessageToolCallsNoTextNilContent verifies that
// when there are tool calls and no text, the wire content is nil.
func TestExtraConvertAssistantMessageToolCallsNoTextNilContent(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertAssistantMessage(AssistantMessage{Content: []AssistantContent{
		ToolCallContent{ToolCallContentEmbed: ToolCallContentEmbed{ToolCallID: "c1", ToolName: "f", Input: json.RawMessage(`{}`)}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["content"] != nil {
		t.Errorf("content should be nil: %v", out["content"])
	}
	if _, has := out["tool_calls"]; !has {
		t.Errorf("tool_calls missing: %v", out)
	}
}

// TestConvertAssistantMessageReasoningIgnored verifies that a
// ReasoningContent in an assistant message is silently dropped on the
// chat path (the chat completions API doesn't carry reasoning on input).
func TestConvertAssistantMessageReasoningIgnored(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertAssistantMessage(AssistantMessage{Content: []AssistantContent{
		TextContent{Text: "ok"},
		ReasoningContent{EncryptedContent: "x", Summary: []string{"thinking"}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["content"] != "ok" {
		t.Errorf("content: %v", out["content"])
	}
}

// TestConvertAssistantMessageUnsupportedType verifies that an
// unrecognized assistant content type returns InvalidPromptError.
func TestConvertAssistantMessageUnsupportedType(t *testing.T) {
	m := newTestChatModel()
	type bogusAssistant struct{ AssistantContent }
	_, err := m.convertAssistantMessage(AssistantMessage{Content: []AssistantContent{
		bogusAssistant{},
	}})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestExtraConvertToolMessageTextResult verifies a tool message with a text
// output type is forwarded as a string content.
func TestExtraConvertToolMessageTextResult(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "text", Value: "the answer"}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len: %d", len(out))
	}
	if out[0]["role"] != "tool" {
		t.Errorf("role: %v", out[0]["role"])
	}
	if out[0]["tool_call_id"] != "c1" {
		t.Errorf("tool_call_id: %v", out[0]["tool_call_id"])
	}
	if out[0]["content"] != "the answer" {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestExtraConvertToolMessageJSONResult verifies a tool message with a JSON
// output type is marshaled to a string.
func TestExtraConvertToolMessageJSONResult(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "json", Value: map[string]any{"a": 1.0}}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out[0]["content"] != `{"a":1}` {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestExtraConvertToolMessageSkipsApprovalResponse verifies that the
// tool-approval-response output type is filtered out on the chat path.
func TestExtraConvertToolMessageSkipsApprovalResponse(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "text", Value: "result"}}},
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "tool-approval-response", Value: "approve"}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(out) != 1 {
		t.Errorf("len: %d, want 1 (approval-response skipped)", len(out))
	}
}

// TestExtraConvertToolMessageErrorTextResult verifies that the "error-text"
// output type is forwarded as a string content.
func TestExtraConvertToolMessageErrorTextResult(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "error-text", Value: "boom"}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out[0]["content"] != "boom" {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestExtraConvertToolMessageExecutionDenied verifies that the
// "execution-denied" output type emits a fallback message when no
// reason is supplied.
func TestExtraConvertToolMessageExecutionDenied(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "execution-denied"}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out[0]["content"] != "Tool call execution denied." {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestExtraConvertToolMessageExecutionDeniedWithReason verifies that
// "execution-denied" with a reason uses the reason.
func TestExtraConvertToolMessageExecutionDeniedWithReason(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "execution-denied", Reason: "policy"}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out[0]["content"] != "policy" {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestExtraConvertToolMessageContentResult verifies that the "content"
// output type is JSON-marshaled.
func TestExtraConvertToolMessageContentResult(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "content", Value: map[string]any{"k": "v"}}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out[0]["content"] != `{"k":"v"}` {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestExtraConvertToolMessageErrorJSONResult verifies that the "error-json"
// output type is JSON-marshaled.
func TestExtraConvertToolMessageErrorJSONResult(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "error-json", Value: map[string]any{"err": "oops"}}}},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out[0]["content"] != `{"err":"oops"}` {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestExtraConvertToolMessageBadType verifies that an unrecognized output
// type throws InvalidPromptError.
func TestExtraConvertToolMessageBadType(t *testing.T) {
	_, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolResultContent: openaicompatible.ToolResultContent{ToolCallID: "c1", Output: ToolResultOutput{Type: "unknown"}}},
	}})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestConvertToolMessageUnsupportedContent verifies that a non-
// ToolResultContent in a tool message throws InvalidPromptError.
func TestConvertToolMessageUnsupportedContent(t *testing.T) {
	type bogusTool struct{ ToolContent }
	_, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		bogusTool{},
	}})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestConvertChatMessagesUnsupportedType verifies that an unsupported
// message type returns InvalidPromptError.
func TestConvertChatMessagesUnsupportedType(t *testing.T) {
	m := newTestChatModel()
	type bogusMsg struct{ Message }
	_, err := m.convertChatMessages([]Message{bogusMsg{}}, nil, map[string]any{})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestImageDetailFromOptionsNilAndEmpty verifies the helper returns ""
// for nil or empty provider options.
func TestImageDetailFromOptionsNilAndEmpty(t *testing.T) {
	if got := imageDetailFromOptions(nil); got != "" {
		t.Errorf("nil: %q", got)
	}
	if got := imageDetailFromOptions(ProviderMetadata{}); got != "" {
		t.Errorf("empty: %q", got)
	}
	if got := imageDetailFromOptions(ProviderMetadata{"openai": "not-a-map"}); got != "" {
		t.Errorf("non-map: %q", got)
	}
	if got := imageDetailFromOptions(ProviderMetadata{"openai": map[string]any{"foo": "bar"}}); got != "" {
		t.Errorf("missing key: %q", got)
	}
	if got := imageDetailFromOptions(ProviderMetadata{"openai": map[string]any{"imageDetail": "high"}}); got != "high" {
		t.Errorf("imageDetail: %q", got)
	}
}
