package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"
)

func TestThinkingEnabledRequest(t *testing.T) {
	temp := 0.7
	topK := 5
	topP := 0.9
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{Thinking: &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 2048}}).(*anthropicLanguageModel)
	req, warnings := model.buildRequest(GenerateOptions{MaxTokens: 100, Temperature: &temp, TopK: &topK, TopP: &topP}, false)
	if req.Thinking == nil || req.Thinking.Type != "enabled" || req.Thinking.BudgetTokens != 2048 || req.MaxTokens != 2148 {
		t.Fatalf("request = %#v", req)
	}
	if req.Temperature != nil || req.TopK != nil || req.TopP != nil || len(warnings) != 4 {
		t.Fatalf("request = %#v warnings = %#v", req, warnings)
	}
}

func TestThinkingEnabledDefaultBudget(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{Thinking: &ThinkingConfig{Type: ThinkingTypeEnabled}}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{MaxTokens: 1}, false)
	if req.Thinking.BudgetTokens != 1024 || req.MaxTokens != 1025 {
		t.Fatalf("request = %#v", req)
	}
}

func TestThinkingAdaptiveRequest(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{Thinking: &ThinkingConfig{Type: ThinkingTypeAdaptive, Display: "summarized"}}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{}, false)
	if req.Thinking == nil || req.Thinking.Type != "adaptive" || req.Thinking.Display != "summarized" {
		t.Fatalf("request = %#v", req)
	}
}

func TestThinkingDisabledRequest(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{Thinking: &ThinkingConfig{Type: ThinkingTypeDisabled}}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{}, false)
	if req.Thinking == nil || req.Thinking.Type != "disabled" {
		t.Fatalf("request = %#v", req)
	}
}

func TestPerRequestThinkingOverridesModelDefault(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{Thinking: &ThinkingConfig{Type: ThinkingTypeEnabled, BudgetTokens: 2048}}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{Thinking: &ThinkingConfig{Type: ThinkingTypeDisabled}}, false)
	if req.Thinking == nil || req.Thinking.Type != "disabled" || req.Thinking.BudgetTokens != 0 {
		t.Fatalf("request = %#v", req)
	}
}

func TestStructuredOutputOutputFormatMode(t *testing.T) {
	schema := map[string]any{"type": "object", "default": map[string]any{"bad": true}, "properties": map[string]any{"name": map[string]any{"type": "string"}}}
	model := CreateAnthropic(ProviderSettings{}).Model("claude-sonnet-4-5", ModelOptions{StructuredOutputMode: StructuredOutputModeOutputFormat}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{StructuredOutput: &StructuredOutput{Name: "out", Schema: schema}}, false)
	if req.OutputConfig == nil || req.OutputConfig.Format.Type != "json_schema" || len(req.Tools) != 0 {
		t.Fatalf("request = %#v", req)
	}
	sanitized := req.OutputConfig.Format.Schema.(map[string]any)
	if _, ok := sanitized["default"]; ok {
		t.Fatalf("schema was not sanitized: %#v", sanitized)
	}
}

func TestStructuredOutputJsonToolMode(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{StructuredOutputMode: StructuredOutputModeJSONTool}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{StructuredOutput: &StructuredOutput{Schema: map[string]any{"type": "object"}}}, false)
	if len(req.Tools) != 1 || req.Tools[0].Name != "json" || req.ToolChoice.Type != "tool" || req.ToolChoice.Name != "json" || !req.ToolChoice.DisableParallelToolUse || !req.DisableParallelToolUse {
		t.Fatalf("request = %#v", req)
	}
}

func TestStructuredOutputAutoMode(t *testing.T) {
	supported := CreateAnthropic(ProviderSettings{}).Model("claude-sonnet-4-5", ModelOptions{StructuredOutputMode: StructuredOutputModeAuto}).(*anthropicLanguageModel)
	req, _ := supported.buildRequest(GenerateOptions{StructuredOutput: &StructuredOutput{Schema: map[string]any{"type": "object"}}}, false)
	if req.OutputConfig == nil || len(req.Tools) != 0 {
		t.Fatalf("supported request = %#v", req)
	}

	unsupported := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{StructuredOutputMode: StructuredOutputModeAuto}).(*anthropicLanguageModel)
	req, _ = unsupported.buildRequest(GenerateOptions{StructuredOutput: &StructuredOutput{Schema: map[string]any{"type": "object"}}}, false)
	if req.OutputConfig == nil || req.OutputConfig.Format != nil || len(req.Tools) != 1 || req.Tools[0].Name != "json" {
		t.Fatalf("unsupported request = %#v", req)
	}
}

func TestSchemaSanitization(t *testing.T) {
	schema := map[string]any{"$schema": "x", "type": "object", "properties": map[string]any{"x": map[string]any{"type": "string", "default": "bad", "examples": []any{"bad"}}, "nested": map[string]any{"type": "object"}}}
	sanitized := SanitizeSchema(schema).(map[string]any)
	if _, ok := sanitized["$schema"]; ok {
		t.Fatalf("schema = %#v", sanitized)
	}
	if sanitized["additionalProperties"] != false {
		t.Fatalf("schema = %#v", sanitized)
	}
	property := sanitized["properties"].(map[string]any)["x"].(map[string]any)
	if _, ok := property["default"]; ok {
		t.Fatalf("property = %#v", property)
	}
	nested := sanitized["properties"].(map[string]any)["nested"].(map[string]any)
	if nested["additionalProperties"] != false {
		t.Fatalf("nested = %#v", nested)
	}
}

func TestCacheControlInRequest(t *testing.T) {
	cache := &CacheControl{Type: "ephemeral", TTL: "5m"}
	system, messages := convertPrompt([]Message{
		SystemMessage{Content: "system", CacheControl: cache},
		UserMessage{Content: []UserContent{CacheControlledTextContent{Text: "hello", CacheControl: cache}}},
		AssistantMessage{Content: []AssistantContent{CacheControlledTextContent{Text: "answer", CacheControl: cache}}},
	})
	if system[0].CacheControl != cache || messages[0].Content[0].CacheControl != cache || messages[1].Content[0].CacheControl != cache {
		t.Fatalf("system = %#v messages = %#v", system, messages)
	}
}

func TestMCPToolStreaming(t *testing.T) {
	sse := "event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"mcp_tool_use\",\"id\":\"mcp\",\"name\":\"tool\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"mcp_tool_use_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":0}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var parts []StreamPart
	for part := range result.Stream {
		parts = append(parts, part)
	}
	if parts[0].(StreamToolCall).Type != "mcp-tool-call" || parts[1].(StreamToolInputDelta).Delta.(InputJSONDelta).PartialJSON == "" || parts[2].(StreamToolInputEnd).ID != "0" {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestMCPServerContainerAndContextRequest(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{
		MCPServers: []MCPServer{{Type: "url", Name: "server", URL: "https://mcp", ToolConfiguration: &ToolConfiguration{Enabled: true, AllowedTools: []string{"tool"}}}},
		Container:  &Container{ID: "container", Skills: []Skill{{Type: "skill", SkillID: "s", Version: "1"}}},
		ContextManagement: &ContextManagement{Edits: []ContextManagementEdit{
			ClearToolUsesEdit{Type: "clear_tool_uses", Trigger: InputTokensTrigger{Value: 100}, Keep: ToolUsesCount{Value: 1}, ClearAtLeast: ToolUsesCount{Value: 2}, ClearToolInputs: true, ExcludeTools: []string{"keep"}},
			ClearThinkingEdit{Type: "clear_thinking", Keep: ThinkingTurnsCount{Value: 1}},
			CompactEdit{Type: "compact", Trigger: ToolUsesTrigger{Value: 3}, PauseAfterCompaction: true, Instructions: "compact"},
		}},
	}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{}, false)
	if req.MCPServers[0].ToolConfiguration.AllowedTools[0] != "tool" || req.Container.Skills[0].SkillID != "s" || len(req.ContextManagement.Edits) != 3 {
		t.Fatalf("request = %#v", req)
	}
	if len(req.Tools) != 1 || req.Tools[0].Type != "mcp_toolset" || req.Tools[0].MCPServerName != "server" || req.Tools[0].DefaultConfig == nil || req.Tools[0].DefaultConfig.Enabled == nil || *req.Tools[0].DefaultConfig.Enabled || req.Tools[0].Configs["tool"].Enabled == nil || !*req.Tools[0].Configs["tool"].Enabled {
		t.Fatalf("request = %#v", req)
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(data) {
		t.Fatalf("invalid json: %s", data)
	}
	if bytes.Contains(data, []byte("tool_configuration")) {
		t.Fatalf("deprecated tool_configuration serialized: %s", data)
	}
}

func TestContextManagementResponseParsing(t *testing.T) {
	body := []byte(`{"content":[],"context_management":{"applied_edits":[{"type":"clear_tool_uses","cleared_tool_uses":2},{"type":"clear_thinking","cleared_thinking_turns":1},{"type":"compact","compacted":true}]}}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	ctx := result.MessageMetadata["contextManagement"].(ContextManagementResponse)
	if len(ctx.AppliedEdits) != 3 || ctx.AppliedEdits[0].(ClearToolUsesResponse).ClearedToolUses != 2 || ctx.AppliedEdits[1].(ClearThinkingResponse).ClearedThinkingTurns != 1 || !ctx.AppliedEdits[2].(CompactResponse).Compacted {
		t.Fatalf("context = %#v", ctx)
	}
}

func TestJsonToolResponseConvertedToText(t *testing.T) {
	body := []byte(`{"content":[{"type":"tool_use","id":"json","name":"json","input":{"ok":true}}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content[0].(TextContent).Text != `{"ok":true}` {
		t.Fatalf("content = %#v", result.Content)
	}
}
