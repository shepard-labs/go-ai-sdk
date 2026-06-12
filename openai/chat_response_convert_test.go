package openai

import (
	"encoding/json"
	"testing"
)

// TestChatModelAccessors verifies the chat model's ModelID, Provider,
// and SupportURLs accessors.
func TestChatModelAccessors(t *testing.T) {
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	if m.ModelID() != "gpt-4o" {
		t.Errorf("ModelID: %q", m.ModelID())
	}
	if m.Provider() != "openai.chat" {
		t.Errorf("Provider: %q", m.Provider())
	}
	if urls := m.SupportURLs(); urls == nil || len(urls) != 0 {
		t.Errorf("SupportURLs: %v", urls)
	}
}

// TestConvertChatToolCallFromResponseValid verifies the happy path.
func TestConvertChatToolCallFromResponseValid(t *testing.T) {
	tc := map[string]any{
		"id":   "call_1",
		"type": "function",
		"function": map[string]any{
			"name":      "search",
			"arguments": `{"q":"x"}`,
		},
		"provider_executed": true,
		"dynamic":           true,
	}
	got, err := convertChatToolCallFromResponse(tc)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if got.ToolCallID != "call_1" {
		t.Errorf("id: %q", got.ToolCallID)
	}
	if got.ToolName != "search" {
		t.Errorf("name: %q", got.ToolName)
	}
	if string(got.Input) != `{"q":"x"}` {
		t.Errorf("input: %s", got.Input)
	}
	if !got.ProviderExecuted {
		t.Errorf("ProviderExecuted: %v", got.ProviderExecuted)
	}
	if !got.Dynamic {
		t.Errorf("Dynamic: %v", got.Dynamic)
	}
}

// TestConvertChatToolCallFromResponseMissingName verifies that an
// unnamed function returns InvalidResponseDataError.
func TestConvertChatToolCallFromResponseMissingName(t *testing.T) {
	tc := map[string]any{
		"id": "call_1",
		"function": map[string]any{
			"arguments": "{}",
		},
	}
	_, err := convertChatToolCallFromResponse(tc)
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T", err)
	}
}

// TestConvertChatToolCallFromResponseEmptyArgs verifies that missing
// arguments become "{}".
func TestConvertChatToolCallFromResponseEmptyArgs(t *testing.T) {
	tc := map[string]any{
		"id": "call_1",
		"function": map[string]any{
			"name": "f",
		},
	}
	got, err := convertChatToolCallFromResponse(tc)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if string(got.Input) != "{}" {
		t.Errorf("input: %s", got.Input)
	}
}

// TestConvertChatToolCallFromResponseNoFunctionMap verifies behavior
// when the function block is missing: returns an error for missing name.
func TestConvertChatToolCallFromResponseNoFunctionMap(t *testing.T) {
	tc := map[string]any{
		"id": "call_1",
	}
	_, err := convertChatToolCallFromResponse(tc)
	if err == nil {
		t.Fatal("expected error for missing function name")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T", err)
	}
}

// TestConvertProviderMetadataOpenAIOnly verifies single-vendor passthrough.
func TestConvertProviderMetadataOpenAIOnly(t *testing.T) {
	raw := map[string]any{"openai": map[string]any{"foo": "bar"}}
	pm, err := convertProviderMetadata(raw)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if pm["openai"].(map[string]any)["foo"] != "bar" {
		t.Errorf("openai: %v", pm)
	}
}

// TestConvertProviderMetadataMultiVendor verifies the function
// preserves multiple vendor keys.
func TestConvertProviderMetadataMultiVendor(t *testing.T) {
	raw := map[string]any{
		"openai":    map[string]any{"a": 1.0},
		"anthropic": map[string]any{"b": 2.0},
		"google":    map[string]any{"c": 3.0},
	}
	pm, err := convertProviderMetadata(raw)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	for _, k := range []string{"openai", "anthropic", "google"} {
		if _, ok := pm[k]; !ok {
			t.Errorf("missing %s", k)
		}
	}
}

// TestConvertProviderMetadataEmpty verifies the empty case.
func TestConvertProviderMetadataEmpty(t *testing.T) {
	pm, err := convertProviderMetadata(map[string]any{})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if pm != nil {
		t.Errorf("expected nil, got %v", pm)
	}
}

// TestParseChatResponseToolCallArguments verifies the tool-call
// converter's argument JSON handling.
func TestParseChatResponseToolCallArguments(t *testing.T) {
	tc := map[string]any{
		"id":   "call_1",
		"type": "function",
		"function": map[string]any{
			"name":      "f",
			"arguments": `{"key":"val"}`,
		},
	}
	got, err := convertChatToolCallFromResponse(tc)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if string(got.Input) != `{"key":"val"}` {
		t.Errorf("input: %q", got.Input)
	}
	if !json.Valid(got.Input) {
		t.Errorf("input not valid JSON: %s", got.Input)
	}
}

// TestParseChatResponseAnnotations verifies the parseChatResponse
// handling of url_citation annotations.
func TestParseChatResponseAnnotations(t *testing.T) {
	body := []byte(`{
		"id":"r1",
		"created":1700000000,
		"model":"gpt-4o",
		"choices":[{
			"message":{
				"role":"assistant",
				"content":"ok",
				"annotations":[{
					"type":"url_citation",
					"url_citation":{
						"uuid":"u1",
						"url":"https://example.com",
						"title":"Example"
					}
				}]
			},
			"finish_reason":"stop"
		}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	result, err := m.parseChatResponse(body, nil, nil)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(result.Content) < 2 {
		t.Fatalf("content: %v", result.Content)
	}
	found := false
	for _, c := range result.Content {
		if sc, ok := c.(SourceContent); ok {
			if sc.URL == "https://example.com" && sc.Title == "Example" {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("expected source content, got %+v", result.Content)
	}
	if result.Response.ID != "r1" {
		t.Errorf("id: %q", result.Response.ID)
	}
}

// TestParseChatResponseLogprobsAndAcceptedTokens verifies the
// providerMetadata logprobs + completion_tokens_details handling.
func TestParseChatResponseLogprobsAndAcceptedTokens(t *testing.T) {
	body := []byte(`{
		"id":"r1",
		"created":1,
		"model":"gpt-4o",
		"choices":[{"logprobs":{"content":[{"token":"a"}]},"message":{"role":"assistant","content":"a"},"finish_reason":"stop"}],
		"usage":{
			"prompt_tokens":1,
			"completion_tokens":1,
			"total_tokens":2,
			"completion_tokens_details":{"accepted_prediction_tokens":3,"rejected_prediction_tokens":1},
			"prompt_tokens_details":{"cached_tokens":0}
		}
	}`)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	result, err := m.parseChatResponse(body, nil, nil)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	pm, ok := result.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("openai provider metadata: %v", result.ProviderMetadata)
	}
	if pm["acceptedPredictionTokens"] != 3 {
		t.Errorf("acceptedPredictionTokens: %v", pm["acceptedPredictionTokens"])
	}
	if pm["rejectedPredictionTokens"] != 1 {
		t.Errorf("rejectedPredictionTokens: %v", pm["rejectedPredictionTokens"])
	}
	if _, has := pm["logprobs"]; !has {
		t.Errorf("logprobs missing: %v", pm)
	}
}

// TestParseChatResponseInvalidJSON verifies malformed JSON throws
// InvalidResponseDataError.
func TestParseChatResponseInvalidJSON(t *testing.T) {
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	_, err := m.parseChatResponse([]byte("not-json"), nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T", err)
	}
}

// TestParseChatResponseCreatedInt64 verifies int64 created timestamp
// is supported.
func TestParseChatResponseCreatedInt64(t *testing.T) {
	body := []byte(`{"id":"r","created":1700000000,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"a"},"finish_reason":"stop"}],"usage":{}}`)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	result, err := m.parseChatResponse(body, nil, nil)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if result.Response.Timestamp == nil {
		t.Errorf("timestamp not set")
	}
}

// TestParseChatResponseCreatedFloat verifies float64 created timestamp.
func TestParseChatResponseCreatedFloat(t *testing.T) {
	body := []byte(`{"id":"r","created":1700000000.0,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"a"},"finish_reason":"stop"}],"usage":{}}`)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	result, err := m.parseChatResponse(body, nil, nil)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if result.Response.Timestamp == nil {
		t.Errorf("timestamp not set")
	}
}

// TestParseChatResponseReasoningContent verifies reasoning_content
// is mapped to ReasoningContent.
func TestParseChatResponseReasoningContent(t *testing.T) {
	body := []byte(`{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"a","reasoning_content":"thinking"},"finish_reason":"stop"}],"usage":{}}`)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	result, err := m.parseChatResponse(body, nil, nil)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	found := false
	for _, c := range result.Content {
		if rc, ok := c.(ReasoningContent); ok {
			if rc.Text == "thinking" {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("reasoning content missing: %v", result.Content)
	}
}

// TestParseChatResponseToolCallInvalidObject verifies that a
// non-map tool call element is silently skipped.
func TestParseChatResponseToolCallInvalidObject(t *testing.T) {
	body := []byte(`{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"a","tool_calls":["nope"]},"finish_reason":"stop"}],"usage":{}}`)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	_, err := m.parseChatResponse(body, nil, nil)
	if err != nil {
		t.Errorf("expected no error for non-map tool call, got %v", err)
	}
}

// TestParseChatResponseToolCallMissingName verifies the parseChatResponse
// error path when the tool call is missing the function name.
func TestParseChatResponseToolCallMissingName(t *testing.T) {
	body := []byte(`{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"a","tool_calls":[{"id":"c1","type":"function","function":{"arguments":"{}"}}]},"finish_reason":"stop"}],"usage":{}}`)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	_, err := m.parseChatResponse(body, nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T", err)
	}
}

// TestStringValue verifies the helper.
func TestStringValue(t *testing.T) {
	if got := stringValue("hi"); got != "hi" {
		t.Errorf("string: %q", got)
	}
	if got := stringValue(123); got != "" {
		t.Errorf("non-string: %q", got)
	}
	if got := stringValue(nil); got != "" {
		t.Errorf("nil: %q", got)
	}
}
