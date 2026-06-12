package openai

import (
	"testing"
)

// TestConvertChatToolChoiceToolStringIsUnsupported verifies that
// tool_choice: "tool" surfaces as UnsupportedFunctionalityError per
// the spec, since chat completions don't support the "tool" string.
func TestConvertChatToolChoiceToolStringIsUnsupported(t *testing.T) {
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	_, err := m.convertChatToolChoice("tool", nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// TestConvertChatToolChoiceAutoPassesThrough verifies that
// tool_choice: "auto" passes through.
func TestConvertChatToolChoiceAutoPassesThrough(t *testing.T) {
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	got, err := m.convertChatToolChoice("auto", nil)
	if err != nil {
		t.Fatalf("convertChatToolChoice: %v", err)
	}
	if got != "auto" {
		t.Errorf("got %v, want auto", got)
	}
}

// TestConvertChatToolChoiceToolNameMap verifies the spec conversion
// for {Type: "tool", ToolName: "X"} → {type: "function", function: {name: "X"}}.
func TestConvertChatToolChoiceToolNameMap(t *testing.T) {
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	got, err := m.convertChatToolChoice(map[string]any{"toolName": "get_weather"}, nil)
	if err != nil {
		t.Fatalf("convertChatToolChoice: %v", err)
	}
	gotMap, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("got %T, want map[string]any", got)
	}
	if gotMap["type"] != "function" {
		t.Errorf("type = %v", gotMap["type"])
	}
	fn, _ := gotMap["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("name = %v", fn["name"])
	}
}
