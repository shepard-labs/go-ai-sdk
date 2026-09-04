package openaicompatible

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
)

func assistantWithThoughtSignature(t *testing.T, requestOptions ProviderOptions, toolOptions ProviderMetadata) map[string]any {
	t.Helper()
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme-provider", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{AssistantMessage{Content: []AssistantContent{
			ToolCallContent{ToolCallID: "call-1", ToolName: "weather", Input: json.RawMessage(`{}`), ProviderOptions: toolOptions},
		}}},
		ProviderOptions: requestOptions,
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBodyFromFetcher(t, f)
	toolCalls := body["messages"].([]any)[0].(map[string]any)["tool_calls"].([]any)
	return toolCalls[0].(map[string]any)
}

func thoughtSignatureFromWire(t *testing.T, toolCall map[string]any) string {
	t.Helper()
	extra, ok := toolCall["extra_content"].(map[string]any)
	if !ok {
		t.Fatalf("extra_content missing: %#v", toolCall)
	}
	google, ok := extra["google"].(map[string]any)
	if !ok {
		t.Fatalf("google envelope missing: %#v", extra)
	}
	sig, ok := google["thought_signature"].(string)
	if !ok {
		t.Fatalf("thought_signature missing: %#v", google)
	}
	return sig
}

func TestAssistantToolCallThoughtSignatureFromGoogleKey(t *testing.T) {
	toolCall := assistantWithThoughtSignature(t, nil, ProviderMetadata{"google": map[string]any{"thoughtSignature": "sig-google"}})
	if got := thoughtSignatureFromWire(t, toolCall); got != "sig-google" {
		t.Fatalf("thought_signature = %q, want sig-google", got)
	}
}

func TestAssistantToolCallThoughtSignatureFallsBackToProviderKey(t *testing.T) {
	toolCall := assistantWithThoughtSignature(t, nil, ProviderMetadata{"acme-provider": map[string]any{"thoughtSignature": "sig-provider"}})
	if got := thoughtSignatureFromWire(t, toolCall); got != "sig-provider" {
		t.Fatalf("thought_signature = %q, want sig-provider", got)
	}
}

func TestAssistantToolCallThoughtSignatureGoogleKeyWins(t *testing.T) {
	toolCall := assistantWithThoughtSignature(t, nil, ProviderMetadata{
		"google":        map[string]any{"thoughtSignature": "sig-google"},
		"acme-provider": map[string]any{"thoughtSignature": "sig-provider"},
	})
	if got := thoughtSignatureFromWire(t, toolCall); got != "sig-google" {
		t.Fatalf("thought_signature = %q, want sig-google", got)
	}
}

func TestAssistantToolCallThoughtSignatureCamelCaseProviderKey(t *testing.T) {
	toolCall := assistantWithThoughtSignature(t,
		ProviderOptions{"acmeProvider": map[string]any{}},
		ProviderMetadata{"acmeProvider": map[string]any{"thoughtSignature": "sig-camel"}})
	if got := thoughtSignatureFromWire(t, toolCall); got != "sig-camel" {
		t.Fatalf("thought_signature = %q, want sig-camel", got)
	}
}
