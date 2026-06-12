package openai

import (
	"encoding/json"
	"testing"
)

// TestResponsesStreamMCPApprovalRequest verifies that when an
// mcp_approval_request output_item.added event arrives, the SDK emits
// a StreamToolCall (with ProviderExecuted=true, Dynamic=true and the
// approvalRequestId in ProviderMetadata) AND a StreamToolApprovalRequest.
func TestResponsesStreamMCPApprovalRequest(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"mcp_approval_request","id":"apr_123","name":"search","server_label":"my-mcp","arguments":"{\"q\":\"hello\"}"}}`,
	}
	parts := driveResponsesEvents(t, m, events)

	var toolCall *StreamToolCall
	var approvalReq *StreamToolApprovalRequest
	for i := range parts {
		switch p := parts[i].(type) {
		case StreamToolCall:
			toolCall = &p
		case StreamToolApprovalRequest:
			approvalReq = &p
		}
	}
	if toolCall == nil {
		t.Fatal("expected StreamToolCall, not found")
	}
	if !toolCall.ToolCallContent.ProviderExecuted {
		t.Errorf("ProviderExecuted should be true")
	}
	if !toolCall.ToolCallContent.Dynamic {
		t.Errorf("Dynamic should be true")
	}
	if toolCall.ToolCallContent.ToolCallID != "mcp-approval-apr_123" {
		t.Errorf("ToolCallID = %q", toolCall.ToolCallContent.ToolCallID)
	}
	pm, _ := toolCall.ToolCallContent.ProviderMetadata["openai"].(map[string]any)
	if pm == nil || pm["approvalRequestId"] != "apr_123" {
		t.Errorf("approvalRequestId not in ProviderMetadata: %v", toolCall.ToolCallContent.ProviderMetadata)
	}
	// Verify the input JSON carries name/arguments/server_label
	var input map[string]any
	if err := json.Unmarshal(toolCall.ToolCallContent.Input, &input); err != nil {
		t.Fatalf("unmarshal input: %v", err)
	}
	if input["name"] != "search" {
		t.Errorf("input.name = %v", input["name"])
	}

	if approvalReq == nil {
		t.Fatal("expected StreamToolApprovalRequest, not found")
	}
	if approvalReq.ApprovalID != "apr_123" {
		t.Errorf("ApprovalID = %q", approvalReq.ApprovalID)
	}
	if approvalReq.ToolCallID != "mcp-approval-apr_123" {
		t.Errorf("ToolCallID = %q", approvalReq.ToolCallID)
	}
}

// TestResponsesStreamMCPApprovalRequestDoneCleans verifies that the
// mcp_approval_request done event clears state without crashing.
func TestResponsesStreamMCPApprovalRequestDoneCleans(t *testing.T) {
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	events := []string{
		`{"type":"response.output_item.added","item":{"type":"mcp_approval_request","id":"apr_456","name":"action","server_label":"srv","arguments":"{}"}}`,
		`{"type":"response.output_item.done","item":{"type":"mcp_approval_request","id":"apr_456"}}`,
	}
	parts := driveResponsesEvents(t, m, events)
	// Should have emitted StreamToolCall + StreamToolApprovalRequest from added,
	// nothing extra from done.
	toolCalls := 0
	approvals := 0
	for _, p := range parts {
		switch p.(type) {
		case StreamToolCall:
			toolCalls++
		case StreamToolApprovalRequest:
			approvals++
		}
	}
	if toolCalls != 1 {
		t.Errorf("toolCalls = %d, want 1", toolCalls)
	}
	if approvals != 1 {
		t.Errorf("approvals = %d, want 1", approvals)
	}
}
