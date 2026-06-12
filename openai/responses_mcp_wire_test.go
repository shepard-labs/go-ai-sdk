package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestResponsesMCPToolWireFormat verifies that an openai.mcp provider
// tool is converted to the wire format with snake_case keys.
func TestResponsesMCPToolWireFormat(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	url := "https://mcp.example.com"
	auth := "Bearer xyz"
	connector := "conn-1"
	desc := "A test MCP server"
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []Tool{
			{Type: "provider", ID: "openai.mcp", Args: MCPArgs{
				ServerLabel:       "my-server",
				ServerURL:         &url,
				Authorization:     &auth,
				RequireApproval:   &MCPRequireApproval{Always: boolPtrP(true)},
				ConnectorID:       &connector,
				ServerDescription: &desc,
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	mcp, _ := tools[0].(map[string]any)
	if mcp["type"] != "mcp" {
		t.Errorf("type: %v", mcp["type"])
	}
	if mcp["server_label"] != "my-server" {
		t.Errorf("server_label: %v", mcp["server_label"])
	}
	if mcp["server_url"] != url {
		t.Errorf("server_url: %v", mcp["server_url"])
	}
	if mcp["authorization"] != auth {
		t.Errorf("authorization: %v", mcp["authorization"])
	}
	if mcp["connector_id"] != connector {
		t.Errorf("connector_id: %v", mcp["connector_id"])
	}
	if mcp["server_description"] != desc {
		t.Errorf("server_description: %v", mcp["server_description"])
	}
	// require_approval is a nested object with always/never/tool_names
	if _, ok := mcp["require_approval"]; !ok {
		t.Errorf("require_approval missing: %v", mcp)
	}
}

// TestResponsesMCPToolWithAllowedTools verifies the allowed_tools field
// is forwarded in the wire format.
func TestResponsesMCPToolWithAllowedTools(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	url := "https://mcp.example.com"
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []Tool{
			{Type: "provider", ID: "openai.mcp", Args: MCPArgs{
				ServerLabel:  "my-server",
				ServerURL:    &url,
				AllowedTools: []string{"tool_a", "tool_b"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	tools, _ := body["tools"].([]any)
	mcp, _ := tools[0].(map[string]any)
	at, _ := mcp["allowed_tools"].([]any)
	if len(at) != 2 {
		t.Fatalf("allowed_tools: %d", len(at))
	}
	if at[0] != "tool_a" || at[1] != "tool_b" {
		t.Errorf("allowed_tools: %v", at)
	}
}

// TestResponsesMCPToolWithHeaders verifies the headers field is
// forwarded (typically a map[string]string).
func TestResponsesMCPToolWithHeaders(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	url := "https://mcp.example.com"
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools: []Tool{
			{Type: "provider", ID: "openai.mcp", Args: MCPArgs{
				ServerLabel: "my-server",
				ServerURL:   &url,
				Headers:     map[string]string{"X-Api-Key": "abc"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	tools, _ := body["tools"].([]any)
	mcp, _ := tools[0].(map[string]any)
	hdr, ok := mcp["headers"].(map[string]any)
	if !ok || hdr["X-Api-Key"] != "abc" {
		t.Errorf("headers: %v", mcp["headers"])
	}
}

func boolPtrP(b bool) *bool { return &b }
