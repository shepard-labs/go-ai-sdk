package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestResponsesAssistantMessageApprovalResponse verifies that an
// AssistantMessage carrying a ToolApprovalResponse is serialized as an
// mcp_approval_response input item.
func TestResponsesAssistantMessageApprovalResponse(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			AssistantMessage{Content: []AssistantContent{
				ToolApprovalResponse{
					ApprovalID: "appr-1",
					Approve:    true,
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("input: %v", body["input"])
	}
	if len(input) != 1 {
		t.Fatalf("input len = %d", len(input))
	}
	m, _ := input[0].(map[string]any)
	if m["type"] != "mcp_approval_response" {
		t.Errorf("type: %v", m)
	}
	if m["approval_request_id"] != "appr-1" {
		t.Errorf("approval_request_id: %v", m)
	}
	if m["approve"] != true {
		t.Errorf("approve: %v", m)
	}
	if _, has := m["reason"]; has {
		t.Errorf("reason should not be set when empty")
	}
}

// TestResponsesAssistantMessageApprovalResponseDenyWithReason verifies that
// deny + reason is also forwarded.
func TestResponsesAssistantMessageApprovalResponseDenyWithReason(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{
			AssistantMessage{Content: []AssistantContent{
				ToolApprovalResponse{
					ApprovalID: "appr-2",
					Approve:    false,
					Reason:     "policy",
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	input, _ := body["input"].([]any)
	m, _ := input[0].(map[string]any)
	if m["approve"] != false {
		t.Errorf("approve: %v", m)
	}
	if m["reason"] != "policy" {
		t.Errorf("reason: %v", m)
	}
}
