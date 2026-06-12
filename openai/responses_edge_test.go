package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestResponsesParseUnknownOutputTypeWarns verifies an unrecognized
// output type emits a warning rather than panicking or silently dropping.
func TestResponsesParseUnknownOutputTypeWarns(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"type":"future_unknown_type","foo":"bar"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected warning for unknown output type, got 0")
	}
}

// TestResponsesParseWebSearchCall verifies a web_search_call output
// surfaces both a tool call and tool result.
func TestResponsesParseWebSearchCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"ws-1","type":"web_search_call","action":{"type":"search","query":"weather"}}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) != 2 {
		t.Fatalf("content: %d", len(res.Content))
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || tc.ToolName != "web_search" {
		t.Errorf("expected web_search tool call, got: %T", res.Content[0])
	}
	if _, ok := res.Content[1].(ToolResultContent); !ok {
		t.Errorf("expected tool result, got: %T", res.Content[1])
	}
}

// TestResponsesParseFileSearchCall verifies file_search_call surfaces
// queries and results into a JSON tool result.
func TestResponsesParseFileSearchCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"fs-1","type":"file_search_call","queries":["q1"],"results":[{"file_id":"f-1","score":0.9}]}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 2 {
		t.Fatalf("content: %d", len(res.Content))
	}
	tr, ok := res.Content[1].(ToolResultContent)
	if !ok {
		t.Fatalf("expected tool result, got: %T", res.Content[1])
	}
	if tr.Output.Type != "json" {
		t.Errorf("output type: %q", tr.Output.Type)
	}
}

// TestResponsesParseCodeInterpreterCall verifies code_interpreter_call
// surfaces both call and result, with output JSON.
func TestResponsesParseCodeInterpreterCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"ci-1","type":"code_interpreter_call","code":"print(1)","container_id":"c-1","outputs":[{"type":"logs","logs":"1\n"}]}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 2 {
		t.Fatalf("content: %d", len(res.Content))
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || tc.ToolName != "code_interpreter" {
		t.Errorf("expected code_interpreter call, got: %T", res.Content[0])
	}
}

// TestResponsesParseImageGenerationCall verifies image_generation_call
// surfaces a base64 result.
func TestResponsesParseImageGenerationCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"ig-1","type":"image_generation_call","result":"aGVsbG8="}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 2 {
		t.Fatalf("content: %d", len(res.Content))
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || tc.ToolName != "image_generation" {
		t.Errorf("expected image_generation, got: %T", res.Content[0])
	}
}

// TestResponsesParseMcpCall verifies mcp_call surfaces both call and result.
func TestResponsesParseMcpCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"mcp-1","type":"mcp_call","name":"lookup","server_label":"srv","arguments":"{\"x\":1}","output":"ok"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 2 {
		t.Fatalf("content: %d", len(res.Content))
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || tc.ToolName != "mcp" {
		t.Errorf("expected mcp call, got: %T", res.Content[0])
	}
}

// TestResponsesParseMcpApprovalRequest verifies mcp_approval_request
// surfaces a tool call with the approval data.
func TestResponsesParseMcpApprovalRequest(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"mar-1","type":"mcp_approval_request","name":"lookup","server_label":"srv","arguments":"{\"x\":1}"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 1 {
		t.Fatalf("content: %d", len(res.Content))
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || tc.ToolName != "mcp" {
		t.Errorf("expected mcp call, got: %T", res.Content[0])
	}
}

// TestResponsesParseLocalShellCall verifies local_shell_call surfaces
// a tool call with the action as input.
func TestResponsesParseLocalShellCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"call_id":"ls-1","type":"local_shell_call","action":{"command":["ls"]}}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 1 {
		t.Fatalf("content: %d", len(res.Content))
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || tc.ToolName != "local_shell" {
		t.Errorf("expected local_shell, got: %T", res.Content[0])
	}
}

// TestResponsesParseShellCallClientAction verifies a shell_call without
// a container_id is not provider-executed.
func TestResponsesParseShellCallClientAction(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"sc-1","call_id":"sc-call","type":"shell_call","action":{"command":["ls"]}}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || tc.ProviderExecuted {
		t.Errorf("expected non-provider-executed shell call, got: %+v", res.Content[0])
	}
}

// TestResponsesParseShellCallServerAction verifies a shell_call with a
// container_id is provider-executed.
func TestResponsesParseShellCallServerAction(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"sc-1","call_id":"sc-call","type":"shell_call","action":{"command":["ls"],"container_id":"c-1"}}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || !tc.ProviderExecuted {
		t.Errorf("expected provider-executed shell call, got: %+v", res.Content[0])
	}
}

// TestResponsesParseApplyPatchCall verifies apply_patch_call surfaces
// operation as the input.
func TestResponsesParseApplyPatchCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"call_id":"ap-1","type":"apply_patch_call","operation":{"op":"add","path":"/a.txt","value":"hi"}}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 1 {
		t.Fatalf("content: %d", len(res.Content))
	}
	tc, ok := res.Content[0].(ToolCallContent)
	if !ok || tc.ToolName != "apply_patch" {
		t.Fatalf("expected apply_patch, got: %T", res.Content[0])
	}
	if !json.Valid(tc.Input) {
		t.Errorf("input not valid JSON: %s", tc.Input)
	}
}

// TestResponsesParseToolSearchServer verifies tool_search_call with
// execution=server is provider-executed.
func TestResponsesParseToolSearchServer(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"call_id":"ts-1","type":"tool_search_call","execution":"server","arguments":"{}"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if tc, ok := res.Content[0].(ToolCallContent); !ok || !tc.ProviderExecuted {
		t.Errorf("expected provider-executed tool search, got: %+v", res.Content[0])
	}
}

// TestResponsesParseCompaction verifies a compaction item surfaces as
// CompactionContent.
func TestResponsesParseCompaction(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"cp-1","type":"compaction","encrypted_content":"abc"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 1 {
		t.Fatalf("content: %d", len(res.Content))
	}
	if cp, ok := res.Content[0].(CompactionContent); !ok || cp.ItemID != "cp-1" {
		t.Errorf("expected CompactionContent, got: %T", res.Content[0])
	}
}

// TestResponsesInvalidJSON verifies malformed JSON throws
// InvalidResponseDataError.
func TestResponsesInvalidJSON(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, "not json")}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T: %v", err, err)
	}
}

// TestResponsesProviderError verifies provider setup error is returned.
func TestResponsesProviderError(t *testing.T) {
	f := &recordingFetcher{}
	p := CreateOpenAI(ProviderSettings{})
	p.(*openaiProvider).fetch = f
	_, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if err != ErrMissingAPIKey {
		t.Errorf("err = %v, want ErrMissingAPIKey", err)
	}
	if f.calls != 0 {
		t.Errorf("expected 0 calls, got %d", f.calls)
	}
}

// TestResponsesParseEmptyOutput verifies an empty output array yields
// no content and no warnings.
func TestResponsesParseEmptyOutput(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) != 0 {
		t.Errorf("expected 0 content, got %d", len(res.Content))
	}
	if len(res.Warnings) != 0 {
		t.Errorf("expected 0 warnings, got %v", res.Warnings)
	}
}

// TestResponsesParseNonMapOutputSkipped verifies a non-object output
// element is silently skipped.
func TestResponsesParseNonMapOutputSkipped(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":["not-an-object",{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) != 1 {
		t.Errorf("content: %d", len(res.Content))
	}
}

// TestResponsesServiceTierMetadata verifies service_tier is captured in
// provider metadata.
func TestResponsesServiceTierMetadata(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","service_tier":"priority","output":[]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	pm, _ := res.ProviderMetadata["openai"].(map[string]any)
	if pm["serviceTier"] != "priority" {
		t.Errorf("serviceTier: %v", pm["serviceTier"])
	}
	if pm["responseId"] != "resp-1" {
		t.Errorf("responseId: %v", pm["responseId"])
	}
}

// TestResponsesParseFunctionCall verifies a function_call output
// surfaces as ToolCallContent with name, args, and call_id.
func TestResponsesParseFunctionCall(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"call_id":"fc-1","type":"function_call","name":"lookup","arguments":"{\"x\":1}"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 1 {
		t.Fatalf("content: %d", len(res.Content))
	}
	tc, ok := res.Content[0].(ToolCallContent)
	if !ok {
		t.Fatalf("expected ToolCallContent, got: %T", res.Content[0])
	}
	if tc.ToolName != "lookup" {
		t.Errorf("ToolName: %q", tc.ToolName)
	}
	if tc.ToolCallID != "fc-1" {
		t.Errorf("ToolCallID: %q", tc.ToolCallID)
	}
}

// TestResponsesParseReasoning verifies a reasoning output surfaces
// summary text, item id, and encrypted content.
func TestResponsesParseReasoning(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"type":"reasoning","id":"r-1","summary":[{"text":"thinking"}],"encrypted_content":"enc"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 1 {
		t.Fatalf("content: %d", len(res.Content))
	}
	rc, ok := res.Content[0].(ReasoningContent)
	if !ok {
		t.Fatalf("expected ReasoningContent, got: %T", res.Content[0])
	}
	if rc.Text != "thinking" {
		t.Errorf("Text: %q", rc.Text)
	}
	if rc.EncryptedContent != "enc" {
		t.Errorf("EncryptedContent: %q", rc.EncryptedContent)
	}
}

// TestResponsesParseMessageWithAnnotations verifies a message with
// annotations propagates them to provider metadata.
func TestResponsesParseMessageWithAnnotations(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[{"id":"msg-1","type":"message","phase":"final","annotations":[{"type":"url_citation","url":"https://x"}],"content":[{"type":"output_text","text":"hi"}]}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) < 1 {
		t.Fatalf("content: %d", len(res.Content))
	}
	tc, ok := res.Content[0].(TextContent)
	if !ok {
		t.Fatalf("expected TextContent, got: %T", res.Content[0])
	}
	pm, _ := tc.ProviderOptions["openai"].(map[string]any)
	if pm["itemId"] != "msg-1" {
		t.Errorf("itemId: %v", pm["itemId"])
	}
	if pm["phase"] != "final" {
		t.Errorf("phase: %v", pm["phase"])
	}
	annotations, _ := pm["annotations"].([]map[string]any)
	if len(annotations) != 1 {
		t.Errorf("annotations: %v", pm["annotations"])
	}
}

// TestResponsesUsageCachedTokens verifies cached_tokens is captured.
func TestResponsesUsageCachedTokens(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":7}}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Usage.InputTokens.CacheRead == nil || *res.Usage.InputTokens.CacheRead != 7 {
		t.Errorf("CacheRead: %+v", res.Usage.InputTokens.CacheRead)
	}
}

// TestResponsesUsageReasoningTokens verifies reasoning_tokens is captured.
func TestResponsesUsageReasoningTokens(t *testing.T) {
	respBody := `{"id":"resp-1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"output_tokens_details":{"reasoning_tokens":3}}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Usage.OutputTokens.Reasoning == nil || *res.Usage.OutputTokens.Reasoning != 3 {
		t.Errorf("Reasoning: %+v", res.Usage.OutputTokens.Reasoning)
	}
}

// TestResponsesCreatedAt verifies created_at maps to Timestamp.
func TestResponsesCreatedAt(t *testing.T) {
	respBody := `{"id":"resp-1","created_at":1700000000,"status":"completed","output":[]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Response.Timestamp == nil {
		t.Errorf("Timestamp should not be nil")
	}
}

// Reference openaicompatible import to keep go vet happy.
var _ = openaicompatible.ToolResultOutput{}
