package openai

import (
	"context"
	"net/http"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestChatStructuredOutputProducesJSONSchema verifies that a chat
// completion with opts.StructuredOutput produces a response_format
// payload of type "json_schema" with schema, name, strict, and
// description fields.
func TestChatStructuredOutputProducesJSONSchema(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
	}
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages:         []openaicompatible.Message{openaicompatible.UserMessage{Content: []openaicompatible.UserContent{openaicompatible.TextContent{Text: "hi"}}}},
		StructuredOutput: &openaicompatible.StructuredOutput{Name: "City", Description: "A city", Schema: schema},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format: %v", body["response_format"])
	}
	if rf["type"] != "json_schema" {
		t.Errorf("type: %v", rf["type"])
	}
	js, _ := rf["json_schema"].(map[string]any)
	if js == nil {
		t.Fatalf("json_schema: %v", rf["json_schema"])
	}
	if js["name"] != "City" {
		t.Errorf("name: %v", js["name"])
	}
	if js["strict"] != true {
		t.Errorf("strict: %v", js["strict"])
	}
	if js["description"] != "A city" {
		t.Errorf("description: %v", js["description"])
	}
	if _, has := js["schema"]; !has {
		t.Errorf("schema missing: %v", js)
	}
}

// TestChatStructuredOutputDefaultName verifies that when no name is
// given, the default name "response" is used.
func TestChatStructuredOutputDefaultName(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages:         []openaicompatible.Message{openaicompatible.UserMessage{Content: []openaicompatible.UserContent{openaicompatible.TextContent{Text: "hi"}}}},
		StructuredOutput: &openaicompatible.StructuredOutput{Schema: map[string]any{"type": "object"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	rf, _ := body["response_format"].(map[string]any)
	js, _ := rf["json_schema"].(map[string]any)
	if js["name"] != "response" {
		t.Errorf("name: %v, want response", js["name"])
	}
}

// TestChatStructuredOutputStrictFalse verifies that the strictJsonSchema
// provider option can disable strict mode.
func TestChatStructuredOutputStrictFalse(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{openaicompatible.UserMessage{Content: []openaicompatible.UserContent{openaicompatible.TextContent{Text: "hi"}}}},
		StructuredOutput: &openaicompatible.StructuredOutput{
			Name:   "X",
			Schema: map[string]any{"type": "object"},
		},
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"strictJsonSchema": false},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	rf, _ := body["response_format"].(map[string]any)
	js, _ := rf["json_schema"].(map[string]any)
	if js["strict"] != false {
		t.Errorf("strict: %v, want false", js["strict"])
	}
}

// TestChatResponseFormatJSONType verifies that ResponseFormat with
// Type: "json" produces a json_object response_format.
func TestChatResponseFormatJSONType(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages:       []openaicompatible.Message{openaicompatible.UserMessage{Content: []openaicompatible.UserContent{openaicompatible.TextContent{Text: "hi"}}}},
		ResponseFormat: &openaicompatible.ResponseFormat{Type: "json"},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	rf, _ := body["response_format"].(map[string]any)
	if rf["type"] != "json_object" {
		t.Errorf("type: %v", rf["type"])
	}
}

// TestChatStructuredOutputTakesPrecedenceOverResponseFormat verifies
// that when both StructuredOutput and ResponseFormat are supplied,
// StructuredOutput wins and a warning is emitted.
func TestChatStructuredOutputTakesPrecedenceOverResponseFormat(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages:       []openaicompatible.Message{openaicompatible.UserMessage{Content: []openaicompatible.UserContent{openaicompatible.TextContent{Text: "hi"}}}},
		ResponseFormat: &openaicompatible.ResponseFormat{Type: "json", Schema: map[string]any{"type": "object"}, Name: "Resp"},
		StructuredOutput: &openaicompatible.StructuredOutput{
			Name:   "Structured",
			Schema: map[string]any{"type": "object"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(result.Warnings) == 0 {
		t.Errorf("expected precedence warning, got none")
	}
	body := decodeRequestBody(t, result.Request.Body)
	rf, _ := body["response_format"].(map[string]any)
	js, _ := rf["json_schema"].(map[string]any)
	if js["name"] != "Structured" {
		t.Errorf("name: %v, want Structured (StructuredOutput wins)", js["name"])
	}
}

// TestChatNoResponseFormatWhenNotSet verifies that no response_format
// is set when neither ResponseFormat nor StructuredOutput is supplied.
func TestChatNoResponseFormatWhenNotSet(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages: []openaicompatible.Message{openaicompatible.UserMessage{Content: []openaicompatible.UserContent{openaicompatible.TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, has := body["response_format"]; has {
		t.Errorf("response_format should not be set: %v", body["response_format"])
	}
}

// TestChatResponseFormatTextTypeIgnored verifies that ResponseFormat
// Type "text" is treated as no format (only "json" triggers output).
func TestChatResponseFormatTextTypeIgnored(t *testing.T) {
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), openaicompatible.GenerateOptions{
		Messages:       []openaicompatible.Message{openaicompatible.UserMessage{Content: []openaicompatible.UserContent{openaicompatible.TextContent{Text: "hi"}}}},
		ResponseFormat: &openaicompatible.ResponseFormat{Type: "text"},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, has := body["response_format"]; has {
		t.Errorf("response_format should not be set for type=text: %v", body["response_format"])
	}
}
