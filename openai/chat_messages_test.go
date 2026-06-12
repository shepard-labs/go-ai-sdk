package openai

import (
	"encoding/json"
	"net/url"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// newTestChatModel builds a minimal openaiChatLanguageModel for direct
// calls into the message-conversion helpers.
func newTestChatModel() *openaiChatLanguageModel {
	return &openaiChatLanguageModel{
		modelID: "gpt-4o",
		provider: &openaiProvider{
			apiKey:                "test",
			fileIDPrefixes:        []string{"file-"},
			passThroughUnsupportedFiles: false,
		},
	}
}

// TestConvertChatMessagesSystemAsSystem verifies the default system role.
func TestConvertChatMessagesSystemAsSystem(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertChatMessages([]Message{
		SystemMessage{Content: "you are helpful"},
		UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
	}, nil, map[string]any{})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("len = %d", len(out))
	}
	if out[0]["role"] != "system" {
		t.Errorf("system role: %v", out[0]["role"])
	}
	if out[0]["content"] != "you are helpful" {
		t.Errorf("system content: %v", out[0]["content"])
	}
	if out[1]["role"] != "user" {
		t.Errorf("user role: %v", out[1]["role"])
	}
}

// TestConvertChatMessagesSystemAsDeveloper verifies developer mode routes
// the system message into the developer role.
func TestConvertChatMessagesSystemAsDeveloper(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertChatMessages([]Message{
		SystemMessage{Content: "be brief"},
	}, nil, map[string]any{"__systemMessageMode": "developer"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out[0]["role"] != "developer" {
		t.Errorf("developer role: %v", out[0]["role"])
	}
	if out[0]["content"] != "be brief" {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestConvertChatMessagesSystemRemoved verifies the "remove" mode drops
// system messages entirely.
func TestConvertChatMessagesSystemRemoved(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertChatMessages([]Message{
		SystemMessage{Content: "you are helpful"},
		UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
	}, nil, map[string]any{"__systemMessageMode": "remove"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	if out[0]["role"] != "user" {
		t.Errorf("role: %v", out[0]["role"])
	}
}

// TestConvertUserMessageSingleTextShortCircuit verifies that a user message
// with a single text content emits a string content (not a parts array).
func TestConvertUserMessageSingleTextShortCircuit(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserMessage(UserMessage{Content: []UserContent{TextContent{Text: "hello"}}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["role"] != "user" {
		t.Errorf("role: %v", out["role"])
	}
	if out["content"] != "hello" {
		t.Errorf("content: %v", out["content"])
	}
}

// TestConvertUserMessageMultipleParts verifies a user message with multiple
// parts emits a content parts array.
func TestConvertUserMessageMultipleParts(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserMessage(UserMessage{Content: []UserContent{
		TextContent{Text: "what is this?"},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	parts, ok := out["content"].([]map[string]any)
	if !ok {
		// Single text short-circuits; add a second text to force parts form.
		out, err = m.convertUserMessage(UserMessage{Content: []UserContent{
			TextContent{Text: "what is this?"},
			TextContent{Text: "and this?"},
		}})
		if err != nil {
			t.Fatalf("convert2: %v", err)
		}
		parts, ok = out["content"].([]map[string]any)
		if !ok {
			t.Fatalf("expected parts, got: %T %v", out["content"], out["content"])
		}
	}
	if len(parts) == 0 {
		t.Errorf("parts: %v", parts)
	}
}

// TestConvertAssistantMessageTextOnly verifies a pure text assistant
// message emits a content string with no tool_calls.
func TestConvertAssistantMessageTextOnly(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertAssistantMessage(AssistantMessage{Content: []AssistantContent{TextContent{Text: "hi"}}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["role"] != "assistant" {
		t.Errorf("role: %v", out["role"])
	}
	if out["content"] != "hi" {
		t.Errorf("content: %v", out["content"])
	}
	if _, has := out["tool_calls"]; has {
		t.Errorf("expected no tool_calls: %v", out["tool_calls"])
	}
}

// TestConvertAssistantMessageWithToolCalls verifies tool calls emit
// tool_calls entries and content: nil when there's no text.
func TestConvertAssistantMessageWithToolCalls(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertAssistantMessage(AssistantMessage{Content: []AssistantContent{
		openaicompatible.ToolCallContent{
			ToolCallID: "call-1",
			ToolName:   "f",
			Input:      json.RawMessage(`{"x":1}`),
		},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["content"] != nil {
		t.Errorf("content: %v", out["content"])
	}
	tcs, ok := out["tool_calls"].([]map[string]any)
	if !ok || len(tcs) != 1 {
		t.Fatalf("tool_calls: %v", out["tool_calls"])
	}
	if tcs[0]["id"] != "call-1" {
		t.Errorf("id: %v", tcs[0]["id"])
	}
}

// TestConvertToolMessageSingleResult verifies a single tool result emits
// a {role: tool, tool_call_id, content} entry.
func TestConvertToolMessageSingleResult(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{
			ToolResultContent: openaicompatible.ToolResultContent{
				ToolCallID: "call-1",
				Output: openaicompatible.ToolResultOutput{
					Type:  "text",
					Value: "sunny",
				},
			},
		},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len = %d", len(out))
	}
	if out[0]["role"] != "tool" {
		t.Errorf("role: %v", out[0]["role"])
	}
	if out[0]["tool_call_id"] != "call-1" {
		t.Errorf("id: %v", out[0]["tool_call_id"])
	}
	if out[0]["content"] != "sunny" {
		t.Errorf("content: %v", out[0]["content"])
	}
}

// TestConvertToolMessageSkipsApprovalResponses verifies tool-approval-response
// parts are skipped (no chat-completion equivalent).
func TestConvertToolMessageSkipsApprovalResponses(t *testing.T) {
	out, err := convertToolMessage(ToolMessage{Content: []ToolContent{
		ToolResultContent{
			ToolResultContent: openaicompatible.ToolResultContent{
				ToolCallID: "call-1",
				Output: openaicompatible.ToolResultOutput{
					Type: "tool-approval-response",
				},
			},
		},
	}})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected 0 outputs (approval skipped), got %d", len(out))
	}
}

// TestConvertToolResultOutput verifies text/json mapping.
func TestConvertToolResultOutput(t *testing.T) {
	cases := []struct {
		name  string
		input openaicompatible.ToolResultOutput
		want  string
	}{
		{
			"text",
			openaicompatible.ToolResultOutput{Type: "text", Value: "hello"},
			"hello",
		},
		{
			"json",
			openaicompatible.ToolResultOutput{Type: "json", Value: map[string]any{"a": float64(1)}},
			`{"a":1}`,
		},
		{
			"error-text",
			openaicompatible.ToolResultOutput{Type: "error-text", Value: "boom"},
			"boom",
		},
		{
			"execution-denied-default",
			openaicompatible.ToolResultOutput{Type: "execution-denied"},
			"Tool call execution denied.",
		},
		{
			"execution-denied-reason",
			openaicompatible.ToolResultOutput{Type: "execution-denied", Reason: "user said no"},
			"user said no",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertToolResultOutput(c.input)
			if err != nil {
				t.Fatalf("convert: %v", err)
			}
			if got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

// TestConvertToolResultOutputUnsupportedType verifies unsupported output
// types return an error.
func TestConvertToolResultOutputUnsupportedType(t *testing.T) {
	_, err := convertToolResultOutput(openaicompatible.ToolResultOutput{Type: "weird"})
	if err == nil {
		t.Errorf("expected error for unsupported type")
	}
}

// TestIsURLDataHelper verifies the URL detection logic.
func TestIsURLDataHelper(t *testing.T) {
	u, _ := url.Parse("https://example.com/x.png")
	if !isURLData(u) {
		t.Error("*url.URL should be a URL")
	}
	if isURLData("not a url") {
		t.Error("plain string should not be a URL")
	}
	if isURLData([]byte("data")) {
		t.Error("bytes should not be a URL")
	}
	if isURLData(nil) {
		t.Error("nil should not be a URL")
	}
}

// TestFileDataURLBytes verifies bytes inputs get base64-encoded into a
// data URL with the right media type.
func TestFileDataURLBytes(t *testing.T) {
	got, err := fileDataURL([]byte{0x01, 0x02}, "image/png")
	if err != nil {
		t.Fatalf("fileDataURL: %v", err)
	}
	if !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Errorf("prefix: %q", got)
	}
}

// TestBase64DataBytes verifies base64Data encodes []byte correctly.
func TestBase64DataBytes(t *testing.T) {
	got, err := base64Data([]byte("hello"))
	if err != nil {
		t.Fatalf("base64Data: %v", err)
	}
	if got != "aGVsbG8=" {
		t.Errorf("got %q, want aGVsbG8=", got)
	}
}

// TestBase64DataString verifies base64Data passes a string through as-is
// (treats it as already-encoded base64).
func TestBase64DataString(t *testing.T) {
	got, err := base64Data("aGVsbG8=")
	if err != nil {
		t.Fatalf("base64Data: %v", err)
	}
	if got != "aGVsbG8=" {
		t.Errorf("got %q, want aGVsbG8=", got)
	}
}

// TestImageDetailFromOptions verifies image detail selection.
func TestImageDetailFromOptions(t *testing.T) {
	cases := []struct {
		opts ProviderMetadata
		want string
	}{
		{nil, ""},
		{ProviderMetadata{}, ""},
		{ProviderMetadata{"openai": map[string]any{"imageDetail": "low"}}, "low"},
		{ProviderMetadata{"openai": map[string]any{"imageDetail": "high"}}, "high"},
		{ProviderMetadata{"openai": map[string]any{"imageDetail": "auto"}}, "auto"},
		{ProviderMetadata{"openai": map[string]any{}}, ""},
	}
	for _, c := range cases {
		if got := imageDetailFromOptions(c.opts); got != c.want {
			t.Errorf("opts=%+v: got %q, want %q", c.opts, got, c.want)
		}
	}
}
