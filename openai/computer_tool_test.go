package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestComputerFactory verifies the computer provider tool constructor.
func TestComputerFactory(t *testing.T) {
	got := Computer(ComputerArgs{DisplayWidth: 1024, DisplayHeight: 768, Environment: strP("browser")})
	if got.Type != "provider" {
		t.Errorf("type: %q", got.Type)
	}
	if got.ID != "openai.computer" {
		t.Errorf("id: %q", got.ID)
	}
	args, ok := got.Args.(ComputerArgs)
	if !ok {
		t.Fatalf("args type: %T", got.Args)
	}
	if args.DisplayWidth != 1024 || args.DisplayHeight != 768 {
		t.Errorf("dimensions: %+v", args)
	}
	if args.Environment == nil || *args.Environment != "browser" {
		t.Errorf("environment: %+v", args.Environment)
	}
}

// TestResponsesComputerToolWire verifies the computer tool serializes to
// {type: "computer"} with display dimensions and environment.
func TestResponsesComputerToolWire(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		Computer(ComputerArgs{DisplayWidth: 1024, DisplayHeight: 768, Environment: strP("browser")}),
	})
	tools := extractTools(t, body)
	c, ok := tools["computer"]
	if !ok {
		t.Fatalf("no computer in tools: %v", tools)
	}
	if v, _ := c["display_width"].(float64); v != 1024 {
		t.Errorf("display_width: %v", c["display_width"])
	}
	if v, _ := c["display_height"].(float64); v != 768 {
		t.Errorf("display_height: %v", c["display_height"])
	}
	if c["environment"] != "browser" {
		t.Errorf("environment: %v", c["environment"])
	}
}

// runResponsesRaw runs a responses request with the given messages, tools,
// and stub response body, returning the result and request body.
func runResponsesRaw(t *testing.T, model string, opts ResponsesGenerateOptions, respBody string) (*ResponsesGenerateResult, map[string]any) {
	t.Helper()
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses(model).DoGenerate(context.Background(), opts)
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	return result, decodeRequestBody(t, result.Request.Body)
}

// TestResponsesComputerCallDecode verifies a computer_call output item
// decodes to a provider-executed computer tool call with SDK-shape input.
func TestResponsesComputerCallDecode(t *testing.T) {
	respBody := `{"id":"r","model":"gpt-5.6","status":"completed","output":[
		{"type":"computer_call","id":"comp_1","call_id":"call_1","status":"in_progress",
		 "actions":[{"type":"click","button":"left","x":100,"y":200},{"type":"scroll","x":0,"y":0,"scroll_x":0,"scroll_y":-500}],
		 "pending_safety_checks":[{"id":"sc_1","code":"malicious","message":"careful"}]},
		{"type":"message","content":[{"type":"output_text","text":"ok"}]}
	],"usage":{"input_tokens":1,"output_tokens":1}}`
	result, _ := runResponsesRaw(t, "gpt-5.6", ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools:    []Tool{Computer(ComputerArgs{DisplayWidth: 1024, DisplayHeight: 768})},
	}, respBody)
	var call *ToolCallContent
	for _, c := range result.Content {
		if tc, ok := c.(ToolCallContent); ok && tc.ToolName == "computer" {
			cp := tc
			call = &cp
		}
	}
	if call == nil {
		t.Fatalf("no computer tool call in %+v", result.Content)
	}
	if call.ToolCallID != "call_1" {
		t.Errorf("call id: %q", call.ToolCallID)
	}
	if !call.ProviderExecuted {
		t.Error("computer call should be provider-executed")
	}
	var input map[string]any
	if err := json.Unmarshal(call.Input, &input); err != nil {
		t.Fatalf("input JSON: %v", err)
	}
	if input["status"] != "in_progress" {
		t.Errorf("status: %v", input["status"])
	}
	actions, _ := input["actions"].([]any)
	if len(actions) != 2 {
		t.Fatalf("actions: %v", input["actions"])
	}
	scroll, _ := actions[1].(map[string]any)
	if scroll["scrollX"] != float64(0) || scroll["scrollY"] != float64(-500) {
		t.Errorf("scroll mapped: %v", scroll)
	}
	if _, has := scroll["scroll_x"]; has {
		t.Errorf("snake_case scroll_x leaked to SDK shape: %v", scroll)
	}
	checks, _ := input["pendingSafetyChecks"].([]any)
	if len(checks) != 1 {
		t.Fatalf("pendingSafetyChecks: %v", input["pendingSafetyChecks"])
	}
	if id, _ := call.ProviderMetadata["openai"].(map[string]any)["itemId"]; id != "comp_1" {
		t.Errorf("itemId: %v", call.ProviderMetadata)
	}
}

// TestResponsesComputerRoundTrip verifies a computer tool call and its
// screenshot result encode back to computer_call / computer_call_output
// input items on a follow-up request.
func TestResponsesComputerRoundTrip(t *testing.T) {
	callInput, _ := json.Marshal(map[string]any{
		"actions":             []any{map[string]any{"type": "click", "button": "left", "x": 100, "y": 200}},
		"pendingSafetyChecks": []any{map[string]any{"id": "sc_1"}},
		"status":              "in_progress",
	})
	resultValue := map[string]any{
		"output": map[string]any{"type": "computer_screenshot", "imageUrl": "img-1"},
		"acknowledgedSafetyChecks": []any{
			map[string]any{"id": "sc_1"},
		},
	}
	respBody := `{"id":"r","model":"gpt-5.6","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`
	_, body := runResponsesRaw(t, "gpt-5.6", ResponsesGenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{
					ToolCallContentEmbed: ToolCallContentEmbed{
						ToolCallID:       "call_1",
						ToolName:         "computer",
						Input:            callInput,
						ProviderMetadata: ProviderMetadata{"openai": map[string]any{"itemId": "comp_1"}},
					},
					ProviderExecuted: true,
				},
			}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: "call_1",
						Output:     ToolResultOutput{Type: "json", Value: resultValue},
					},
					ToolName: "computer",
				},
			}},
		},
		Tools: []Tool{Computer(ComputerArgs{DisplayWidth: 1024, DisplayHeight: 768})},
	}, respBody)
	input, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("input: %v", body["input"])
	}
	byType := map[string]map[string]any{}
	for _, item := range input {
		if m, ok := item.(map[string]any); ok {
			if typ, ok := m["type"].(string); ok {
				byType[typ] = m
			}
		}
	}
	call, ok := byType["computer_call"]
	if !ok {
		t.Fatalf("no computer_call in input: %v", input)
	}
	if call["call_id"] != "call_1" || call["id"] != "comp_1" {
		t.Errorf("call ids: %v", call)
	}
	if call["status"] != "in_progress" {
		t.Errorf("status: %v", call["status"])
	}
	actions, _ := call["actions"].([]any)
	if len(actions) != 1 {
		t.Errorf("actions: %v", call["actions"])
	}
	checks, _ := call["pending_safety_checks"].([]any)
	if len(checks) != 1 {
		t.Errorf("pending_safety_checks: %v", call["pending_safety_checks"])
	}
	output, ok := byType["computer_call_output"]
	if !ok {
		t.Fatalf("no computer_call_output in input: %v", input)
	}
	if output["call_id"] != "call_1" {
		t.Errorf("output call_id: %v", output["call_id"])
	}
	shot, _ := output["output"].(map[string]any)
	if shot["type"] != "computer_screenshot" || shot["image_url"] != "img-1" {
		t.Errorf("screenshot: %v", output["output"])
	}
	acks, _ := output["acknowledged_safety_checks"].([]any)
	if len(acks) != 1 {
		t.Errorf("acknowledged_safety_checks: %v", output["acknowledged_safety_checks"])
	}
}
