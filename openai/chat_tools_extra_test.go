package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestChatConvertFunctionToolStrictDefault verifies that a function tool
// without strict provider option gets strict=true by default.
func TestChatConvertFunctionToolStrictDefault(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertFunctionTool(Tool{
		Type:        "function",
		Name:        "search",
		Description: "search the web",
		InputSchema: map[string]any{"type": "object"},
	})
	if err != nil {
		t.Fatalf("convertFunctionTool: %v", err)
	}
	if out["type"] != "function" {
		t.Errorf("type: %v", out["type"])
	}
	fn, _ := out["function"].(map[string]any)
	if fn["strict"] != true {
		t.Errorf("strict should be true by default: %v", fn["strict"])
	}
	if fn["name"] != "search" {
		t.Errorf("name: %v", fn["name"])
	}
}

// TestChatConvertFunctionToolStrictFalse verifies that strict=false in
// the tool's provider options disables strict mode.
func TestChatConvertFunctionToolStrictFalse(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertFunctionTool(Tool{
		Type: "function",
		Name: "x",
		ProviderOptions: ProviderMetadata{
			"openai": map[string]any{"strict": false},
		},
	})
	if err != nil {
		t.Fatalf("convertFunctionTool: %v", err)
	}
	fn, _ := out["function"].(map[string]any)
	if _, has := fn["strict"]; has {
		t.Errorf("strict should not be set: %v", fn["strict"])
	}
}

// TestChatConvertFunctionToolPassThroughExtraFields verifies that other
// openai provider options are spread into the function payload.
func TestChatConvertFunctionToolPassThroughExtraFields(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertFunctionTool(Tool{
		Type: "function",
		Name: "x",
		ProviderOptions: ProviderMetadata{
			"openai": map[string]any{
				"strict":       true,
				"addHeaders":   map[string]any{"X-Trace": "abc"},
				"withHints":    []any{"a", "b"},
				"deferLoading": true,
			},
		},
	})
	if err != nil {
		t.Fatalf("convertFunctionTool: %v", err)
	}
	fn, _ := out["function"].(map[string]any)
	if _, has := fn["addHeaders"]; !has {
		t.Errorf("addHeaders missing: %v", fn)
	}
	if _, has := fn["withHints"]; !has {
		t.Errorf("withHints missing: %v", fn)
	}
	if _, has := fn["deferLoading"]; !has {
		t.Errorf("deferLoading missing: %v", fn)
	}
}

// TestChatConvertProviderToolBadID verifies that a provider tool whose
// ID doesn't start with "openai." is rejected.
func TestChatConvertProviderToolBadID(t *testing.T) {
	m := newTestChatModel()
	_, _, err := m.convertProviderTool(Tool{
		Type: "provider",
		ID:   "anthropic.webSearch",
		Args: map[string]any{},
	})
	if err == nil {
		t.Fatal("expected error for non-openai provider ID")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestChatConvertProviderToolUnknownKind verifies that an unknown
// provider kind is rejected.
func TestChatConvertProviderToolUnknownKind(t *testing.T) {
	m := newTestChatModel()
	_, _, err := m.convertProviderTool(Tool{
		Type: "provider",
		ID:   "openai.bogus_kind",
		Args: map[string]any{},
	})
	if err == nil {
		t.Fatal("expected error for unknown kind")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestChatConvertChatToolsUnsupportedType verifies that a tool with an
// unsupported type throws InvalidPromptError.
func TestChatConvertChatToolsUnsupportedType(t *testing.T) {
	m := newTestChatModel()
	_, _, err := m.convertChatTools([]Tool{
		{Type: "mcp-server", Name: "x"},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestChatToolChoiceToolForNonFunctionUnsupported verifies that
// tool_choice: "tool" for non-function tools throws
// UnsupportedFunctionalityError.
func TestChatToolChoiceToolForNonFunctionUnsupported(t *testing.T) {
	m := newTestChatModel()
	_, err := m.convertChatToolChoice("tool", nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// TestChatToolChoiceToolNameMap verifies that a {toolName: "foo"} shape
// is converted to the OpenAI function tool_choice format.
func TestChatToolChoiceToolNameMap(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertChatToolChoice(map[string]any{"toolName": "mytool"}, nil)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	m_, _ := out.(map[string]any)
	if m_["type"] != "function" {
		t.Errorf("type: %v", m_)
	}
	fn, _ := m_["function"].(map[string]any)
	if fn["name"] != "mytool" {
		t.Errorf("name: %v", fn)
	}
}

// TestChatToolChoiceBadString verifies that an unknown string
// tool_choice throws InvalidPromptError.
func TestChatToolChoiceBadString(t *testing.T) {
	m := newTestChatModel()
	_, err := m.convertChatToolChoice("bogus", nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestChatBuildChatRequestWithAllTools verifies that a chat request
// with both a function tool and a provider tool serializes both.
func TestChatBuildChatRequestWithAllTools(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []openaicompatible.Tool{
			{Type: "function", Name: "search", Description: "search"},
			{Type: "provider", ID: "openai.webSearch", Args: WebSearchArgs{}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if !strings.Contains(string(result.Request.Body), "function") {
		t.Errorf("function tool missing: %s", result.Request.Body)
	}
	if !strings.Contains(string(result.Request.Body), "web_search") {
		t.Errorf("provider tool missing: %s", result.Request.Body)
	}
}
