package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
)

// TestResponsesFileSearchCallParsesQueriesAndResults verifies that a
// file_search_call output item is converted to a ToolCallContent +
// ToolResultContent with the queries and results carried in the result.
func TestResponsesFileSearchCallParsesQueriesAndResults(t *testing.T) {
	respBody := `{
		"id":"r1","created_at":1,"model":"gpt-4o","status":"completed",
		"output":[{
			"type":"file_search_call","id":"fs-1",
			"queries":["what is X?","explain Y"],
			"results":[
				{"file_id":"file-abc","filename":"doc.md","score":0.92,"text":"snippet A"},
				{"file_id":"file-def","filename":"doc2.md","score":0.81,"text":"snippet B"}
			]
		}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
	}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	var toolCall *ToolCallContent
	var toolResult *ToolResultContent
	for i, c := range result.Content {
		if tc, ok := c.(ToolCallContent); ok && tc.ToolName == "file_search" {
			tcCopy := tc
			toolCall = &tcCopy
		}
		if tr, ok := c.(ToolResultContent); ok {
			trCopy := tr
			toolResult = &trCopy
		}
		_ = i
	}
	if toolCall == nil {
		t.Fatal("expected ToolCallContent for file_search, not found")
	}
	if !toolCall.ProviderExecuted {
		t.Errorf("ProviderExecuted should be true for file_search")
	}
	if toolResult == nil {
		t.Fatal("expected ToolResultContent for file_search, not found")
	}
	if toolResult.ToolCallID != "fs-1" {
		t.Errorf("ToolCallID = %q, want fs-1", toolResult.ToolCallID)
	}
	// Result output should contain both queries and results.
	if toolResult.Output.Type != "json" {
		t.Errorf("Output.Type = %q, want json", toolResult.Output.Type)
	}
	combined := decodeCombinedQueriesResults(t, asBytes(t, toolResult.Output.Value))
	if len(combined["queries"].([]any)) != 2 {
		t.Errorf("queries: %v", combined["queries"])
	}
	if len(combined["results"].([]any)) != 2 {
		t.Errorf("results: %v", combined["results"])
	}
}

// TestResponsesMCPCallIncludesApprovalResolution verifies that an
// mcp_call output item with approval_request_id is converted to
// ToolCallContent (Dynamic, ProviderExecuted) plus a ToolResultContent.
func TestResponsesMCPCallIncludesApprovalResolution(t *testing.T) {
	respBody := `{
		"id":"r1","created_at":1,"model":"gpt-4o","status":"completed",
		"output":[{
			"type":"mcp_call","id":"mc-1","name":"search","server_label":"my-mcp",
			"arguments":"{\"q\":\"hello\"}","status":"completed",
			"output":"search result text",
			"approval_request_id":"apr_123"
		}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
	}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	var toolCall *ToolCallContent
	var toolResult *ToolResultContent
	for _, c := range result.Content {
		if tc, ok := c.(ToolCallContent); ok && tc.ToolName == "mcp" {
			tcCopy := tc
			toolCall = &tcCopy
		}
		if tr, ok := c.(ToolResultContent); ok {
			trCopy := tr
			toolResult = &trCopy
		}
	}
	if toolCall == nil {
		t.Fatal("expected mcp ToolCallContent")
	}
	if !toolCall.Dynamic {
		t.Errorf("Dynamic should be true for mcp")
	}
	if !toolCall.ProviderExecuted {
		t.Errorf("ProviderExecuted should be true for mcp")
	}
	if toolResult == nil {
		t.Fatal("expected mcp ToolResultContent")
	}
	if toolResult.ToolCallID != "mc-1" {
		t.Errorf("ToolCallID = %q, want mc-1", toolResult.ToolCallID)
	}
}

// TestResponsesShellCallIncludesItemIDMetadata verifies that shell_call
// output items propagate the item ID into the ToolCallContent's
// ProviderMetadata.
func TestResponsesShellCallIncludesItemIDMetadata(t *testing.T) {
	respBody := `{
		"id":"r1","created_at":1,"model":"gpt-4o","status":"completed",
		"output":[{
			"type":"shell_call","id":"sh-1","call_id":"shc-1","status":"completed",
			"action":{"type":"exec","command":"ls -la"}
		}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
	}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	var toolCall *ToolCallContent
	for _, c := range result.Content {
		if tc, ok := c.(ToolCallContent); ok && tc.ToolName == "shell" {
			tcCopy := tc
			toolCall = &tcCopy
		}
	}
	if toolCall == nil {
		t.Fatal("expected shell ToolCallContent")
	}
	if toolCall.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata should be set on shell call")
	}
	om, ok := toolCall.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("expected openai metadata, got: %v", toolCall.ProviderMetadata)
	}
	if v := om["itemId"]; v != "sh-1" {
		t.Errorf("itemId = %v, want sh-1", om["itemId"])
	}
}

// decodeCombinedQueriesResults is a small helper for file_search tests.
func decodeCombinedQueriesResults(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return out
}

// asBytes converts an `any` to []byte if possible, otherwise fails the test.
func asBytes(t *testing.T, v any) []byte {
	t.Helper()
	if b, ok := v.([]byte); ok {
		return b
	}
	if s, ok := v.(string); ok {
		return []byte(s)
	}
	t.Fatalf("cannot convert %T to []byte", v)
	return nil
}
