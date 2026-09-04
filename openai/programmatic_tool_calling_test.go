package openai

import (
	"encoding/json"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// TestProgrammaticToolCallingFactory verifies the
// programmatic_tool_calling provider tool constructor.
func TestProgrammaticToolCallingFactory(t *testing.T) {
	got := ProgrammaticToolCalling(ProgrammaticToolCallingArgs{
		AllowedCallers: []string{"programmatic"},
	})
	if got.Type != "provider" {
		t.Errorf("type: %q", got.Type)
	}
	if got.ID != "openai.programmatic_tool_calling" {
		t.Errorf("id: %q", got.ID)
	}
	args, ok := got.Args.(ProgrammaticToolCallingArgs)
	if !ok {
		t.Fatalf("args type: %T", got.Args)
	}
	if len(args.AllowedCallers) != 1 || args.AllowedCallers[0] != "programmatic" {
		t.Errorf("allowedCallers: %+v", args)
	}
}

// TestResponsesProgrammaticToolCallingWire verifies the tool serializes
// to the bare {type: "programmatic_tool_calling"} payload.
func TestResponsesProgrammaticToolCallingWire(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		ProgrammaticToolCalling(ProgrammaticToolCallingArgs{AllowedCallers: []string{"programmatic"}}),
	})
	tools := extractTools(t, body)
	ptc, ok := tools["programmatic_tool_calling"]
	if !ok {
		t.Fatalf("no programmatic_tool_calling in tools: %v", tools)
	}
	if ptc["type"] != "programmatic_tool_calling" {
		t.Errorf("type: %v", ptc["type"])
	}
}

// TestResponsesProgramDecode verifies program/program_output items decode
// to a provider-executed tool call and its result.
func TestResponsesProgramDecode(t *testing.T) {
	respBody := `{"id":"r","model":"gpt-5.6-sol","status":"completed","output":[
		{"type":"program","id":"pg_1","call_id":"call_p1","code":"text(\"hi\")","fingerprint":"fp_1"},
		{"type":"program_output","id":"pgo_1","call_id":"call_p1","result":"hi","status":"completed"},
		{"type":"message","content":[{"type":"output_text","text":"ok"}]}
	],"usage":{"input_tokens":1,"output_tokens":1}}`
	result, _ := runResponsesRaw(t, "gpt-5.6-sol", ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools:    []Tool{ProgrammaticToolCalling(ProgrammaticToolCallingArgs{})},
	}, respBody)
	var call *ToolCallContent
	var res *ToolResultContent
	for _, c := range result.Content {
		switch p := c.(type) {
		case ToolCallContent:
			if p.ToolName == "programmatic_tool_calling" {
				cp := p
				call = &cp
			}
		case ToolResultContent:
			if p.ToolName == "programmatic_tool_calling" {
				rp := p
				res = &rp
			}
		}
	}
	if call == nil {
		t.Fatalf("no program tool call in %+v", result.Content)
	}
	if call.ToolCallID != "call_p1" || !call.ProviderExecuted {
		t.Errorf("call: %+v", call)
	}
	var input map[string]any
	if err := json.Unmarshal(call.Input, &input); err != nil {
		t.Fatalf("input JSON: %v", err)
	}
	if input["code"] != `text("hi")` || input["fingerprint"] != "fp_1" {
		t.Errorf("input: %v", input)
	}
	openaiMeta, ok := call.ProviderMetadata["openai"].(map[string]any)
	if !ok || openaiMeta["itemId"] != "pg_1" {
		t.Errorf("call itemId: %v", call.ProviderMetadata)
	}
	if res == nil {
		t.Fatalf("no program tool result in %+v", result.Content)
	}
	if res.ToolCallID != "call_p1" {
		t.Errorf("result call id: %q", res.ToolCallID)
	}
	var value map[string]any
	raw, ok := res.Output.Value.([]byte)
	if !ok {
		t.Fatalf("result value type: %T", res.Output.Value)
	}
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("result value JSON: %v", err)
	}
	if value["result"] != "hi" || value["status"] != "completed" {
		t.Errorf("result value: %v", value)
	}
}

// TestResponsesProgramRoundTrip verifies program calls and outputs encode
// back to program/program_output input items on a follow-up request.
func TestResponsesProgramRoundTrip(t *testing.T) {
	callInput, _ := json.Marshal(map[string]any{"code": `text("hi")`, "fingerprint": "fp_1"})
	respBody := `{"id":"r","model":"gpt-5.6-sol","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`
	storeFalse := false
	_, body := runResponsesRaw(t, "gpt-5.6-sol", ResponsesGenerateOptions{
		Store: &storeFalse,
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{
					ToolCallContentEmbed: ToolCallContentEmbed{
						ToolCallID:       "call_p1",
						ToolName:         "programmatic_tool_calling",
						Input:            callInput,
						ProviderMetadata: ProviderMetadata{"openai": map[string]any{"itemId": "pg_1"}},
					},
					ProviderExecuted: true,
				},
			}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: "call_p1",
						Output:     ToolResultOutput{Type: "json", Value: map[string]any{"result": "hi", "status": "completed"}},
					},
					ToolName: "programmatic_tool_calling",
				},
			}},
		},
		Tools: []Tool{ProgrammaticToolCalling(ProgrammaticToolCallingArgs{})},
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
	program, ok := byType["program"]
	if !ok {
		t.Fatalf("no program in input: %v", input)
	}
	if program["call_id"] != "call_p1" || program["id"] != "pg_1" {
		t.Errorf("program ids: %v", program)
	}
	if program["code"] != `text("hi")` || program["fingerprint"] != "fp_1" {
		t.Errorf("program payload: %v", program)
	}
	output, ok := byType["program_output"]
	if !ok {
		t.Fatalf("no program_output in input: %v", input)
	}
	if output["call_id"] != "call_p1" || output["result"] != "hi" || output["status"] != "completed" {
		t.Errorf("program_output: %v", output)
	}
}

// TestResponsesFunctionCallerCorrelation verifies the caller link between
// a program-initiated function call and its result: decode carries the
// wire caller into provider metadata, and encode writes it back.
func TestResponsesFunctionCallerCorrelation(t *testing.T) {
	respBody := `{"id":"r","model":"gpt-5.6-sol","status":"completed","output":[
		{"type":"function_call","id":"fc_1","call_id":"call_f1","name":"getInventory","arguments":"{\"sku\":\"s1\"}",
		 "caller":{"type":"program","caller_id":"call_p1"}},
		{"type":"message","content":[{"type":"output_text","text":"ok"}]}
	],"usage":{"input_tokens":1,"output_tokens":1}}`
	result, _ := runResponsesRaw(t, "gpt-5.6-sol", ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	}, respBody)
	var call *ToolCallContent
	for _, c := range result.Content {
		if tc, ok := c.(ToolCallContent); ok && tc.ToolName == "getInventory" {
			cp := tc
			call = &cp
		}
	}
	if call == nil {
		t.Fatalf("no function call in %+v", result.Content)
	}
	caller, _ := call.ProviderMetadata["openai"].(map[string]any)["caller"].(map[string]any)
	if caller["type"] != "program" || caller["callerId"] != "call_p1" {
		t.Fatalf("decoded caller: %v", call.ProviderMetadata)
	}

	// Encode the result back with the caller attached.
	respBody2 := `{"id":"r","model":"gpt-5.6-sol","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`
	_, body := runResponsesRaw(t, "gpt-5.6-sol", ResponsesGenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
			AssistantMessage{Content: []AssistantContent{*call}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: "call_f1",
						Output:     ToolResultOutput{Type: "text", Value: "ok"},
						ProviderOptions: ProviderMetadata{
							"openai": map[string]any{"caller": map[string]any{"type": "program", "callerId": "call_p1"}},
						},
					},
					ToolName: "getInventory",
				},
			}},
		},
	}, respBody2)
	input, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("input: %v", body["input"])
	}
	found := false
	for _, item := range input {
		m, ok := item.(map[string]any)
		if !ok || m["type"] != "function_call_output" {
			continue
		}
		found = true
		wireCaller, _ := m["caller"].(map[string]any)
		if wireCaller["type"] != "program" || wireCaller["caller_id"] != "call_p1" {
			t.Errorf("encoded caller: %v", m["caller"])
		}
	}
	if !found {
		t.Errorf("no function_call_output in input: %v", input)
	}
}
