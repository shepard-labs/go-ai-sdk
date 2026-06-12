package openai

import (
	"context"
	"net/http"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// str returns a pointer to s.
func strP(s string) *string { return &s }
func intP(i int) *int       { return &i }
func boolP(b bool) *bool    { return &b }

// runChatWithTools runs a chat request with the given tools and returns
// the request body.
func runChatWithTools(t *testing.T, tools []Tool) map[string]any {
	t.Helper()
	respBody := `{"id":"r","created":1,"model":"gpt-4o","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	// Convert to openaicompatible.Tool so the type matches GenerateOptions.Tools.
	compatTools := make([]openaicompatible.Tool, 0, len(tools))
	for _, t := range tools {
		compatTools = append(compatTools, openaicompatible.Tool{
			Type:        t.Type,
			ID:          t.ID,
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
			Strict:      t.Strict,
			Args:        t.Args,
		})
	}
	result, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools:    compatTools,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	return decodeRequestBody(t, result.Request.Body)
}

// TestChatProviderToolWebSearch verifies that openai.webSearch is
// serialized as {type: "web_search"} in the chat completions body.
func TestChatProviderToolWebSearch(t *testing.T) {
	body := runChatWithTools(t, []Tool{
		{Type: "provider", ID: "openai.webSearch", Args: WebSearchArgs{
			SearchContextSize: strP("high"),
		}},
	})
	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools: %v", body["tools"])
	}
	m, _ := tools[0].(map[string]any)
	if m["type"] != "web_search" {
		t.Errorf("type: %v", m)
	}
}

// TestChatProviderToolFileSearch verifies that openai.fileSearch is
// serialized as {type: "file_search"} in the chat completions body.
func TestChatProviderToolFileSearch(t *testing.T) {
	body := runChatWithTools(t, []Tool{
		{Type: "provider", ID: "openai.fileSearch", Args: FileSearchArgs{
			VectorStoreIDs: []string{"vs_1"},
		}},
	})
	tools, _ := body["tools"].([]any)
	m, _ := tools[0].(map[string]any)
	if m["type"] != "file_search" {
		t.Errorf("type: %v", m)
	}
}

// runResponsesWithTools runs a responses request with the given tools and
// returns the request body.
func runResponsesWithTools(t *testing.T, tools []Tool) map[string]any {
	t.Helper()
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	return decodeRequestBody(t, result.Request.Body)
}

// extractTools pulls the tools array out of the request body and returns
// it as a map keyed by tool "type".
func extractTools(t *testing.T, body map[string]any) map[string]map[string]any {
	t.Helper()
	tools, ok := body["tools"].([]any)
	if !ok {
		t.Fatalf("tools: %v", body["tools"])
	}
	out := map[string]map[string]any{}
	for _, ti := range tools {
		m, ok := ti.(map[string]any)
		if !ok {
			t.Fatalf("tool entry: %T", ti)
		}
		typ, _ := m["type"].(string)
		out[typ] = m
	}
	return out
}

// TestProviderToolFileSearch verifies file_search Args serialize to
// vector_store_ids, max_num_results, ranking_options, filters.
func TestProviderToolFileSearch(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.fileSearch", Args: FileSearchArgs{
			VectorStoreIDs: []string{"vs_1", "vs_2"},
			MaxNumResults:  intP(5),
			Ranking: &FileSearchRanking{
				Ranker:         strP("default"),
				ScoreThreshold: nil,
			},
			Filters: map[string]any{"type": "eq", "key": "x"},
		}},
	})
	tools := extractTools(t, body)
	fs, ok := tools["file_search"]
	if !ok {
		t.Fatalf("no file_search in tools: %v", tools)
	}
	vs, _ := fs["vector_store_ids"].([]any)
	if len(vs) != 2 {
		t.Errorf("vector_store_ids: %v", fs["vector_store_ids"])
	}
	if v, _ := fs["max_num_results"].(float64); v != 5 {
		t.Errorf("max_num_results: %v", fs["max_num_results"])
	}
	if _, has := fs["ranking_options"]; !has {
		t.Errorf("ranking_options missing: %v", fs)
	}
	if _, has := fs["filters"]; !has {
		t.Errorf("filters missing: %v", fs)
	}
}

// TestProviderToolWebSearch verifies web_search Args serialize to
// external_web_access, filters, search_context_size, user_location.
func TestProviderToolWebSearch(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.webSearch", Args: WebSearchArgs{
			ExternalWebAccess: boolP(true),
			SearchContextSize: strP("high"),
			Filters:           &WebSearchFilters{AllowedDomains: []string{"x.com"}},
			UserLocation: &WebSearchUserLocation{
				Type:     "approximate",
				Country:  "US",
				Timezone: "America/Los_Angeles",
			},
		}},
	})
	tools := extractTools(t, body)
	ws, ok := tools["web_search"]
	if !ok {
		t.Fatalf("no web_search in tools: %v", tools)
	}
	if ws["external_web_access"] != true {
		t.Errorf("external_web_access: %v", ws["external_web_access"])
	}
	if ws["search_context_size"] != "high" {
		t.Errorf("search_context_size: %v", ws["search_context_size"])
	}
	if _, has := ws["user_location"]; !has {
		t.Errorf("user_location missing: %v", ws)
	}
}

// TestProviderToolCodeInterpreter verifies code_interpreter Args pass
// through container configuration.
func TestProviderToolCodeInterpreter(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.codeInterpreter", Args: CodeInterpreterArgs{
			Container: &CodeInterpreterContainer{Type: "auto"},
		}},
	})
	tools := extractTools(t, body)
	ci, ok := tools["code_interpreter"]
	if !ok {
		t.Fatalf("no code_interpreter: %v", tools)
	}
	if ci["container"] == nil {
		t.Errorf("container missing: %v", ci)
	}
}

// TestProviderToolImageGeneration verifies image_generation Args pass
// through background, quality, size.
func TestProviderToolImageGeneration(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.imageGeneration", Args: ImageGenerationArgs{
			Background: strP("auto"),
			Quality:    strP("high"),
			Size:       strP("1024x1024"),
		}},
	})
	tools := extractTools(t, body)
	ig, ok := tools["image_generation"]
	if !ok {
		t.Fatalf("no image_generation: %v", tools)
	}
	if ig["background"] != "auto" {
		t.Errorf("background: %v", ig["background"])
	}
	if ig["quality"] != "high" {
		t.Errorf("quality: %v", ig["quality"])
	}
	if ig["size"] != "1024x1024" {
		t.Errorf("size: %v", ig["size"])
	}
}

// TestProviderToolMCP verifies MCP Args include server_label, server_url,
// and convert allowedTools/authorization/etc to snake_case.
func TestProviderToolMCP(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.mcp", Args: MCPArgs{
			ServerLabel:       "my-server",
			ServerURL:         strP("https://mcp.example.com"),
			AllowedTools:      []string{"read", "write"},
			Authorization:     strP("Bearer x"),
			Headers:           map[string]string{"X-Custom": "v"},
			ServerDescription: strP("a test server"),
		}},
	})
	tools := extractTools(t, body)
	mcp, ok := tools["mcp"]
	if !ok {
		t.Fatalf("no mcp: %v", tools)
	}
	if mcp["server_label"] != "my-server" {
		t.Errorf("server_label: %v", mcp["server_label"])
	}
	if mcp["server_url"] != "https://mcp.example.com" {
		t.Errorf("server_url: %v", mcp["server_url"])
	}
	if mcp["authorization"] != "Bearer x" {
		t.Errorf("authorization: %v", mcp["authorization"])
	}
	if mcp["server_description"] != "a test server" {
		t.Errorf("server_description: %v", mcp["server_description"])
	}
	hdrs, ok := mcp["headers"].(map[string]any)
	if !ok || hdrs["X-Custom"] != "v" {
		t.Errorf("headers: %v", mcp["headers"])
	}
}

// TestProviderToolMCPRequireApproval verifies that the MCP tool's
// require_approval field is converted to snake_case.
func TestProviderToolMCPRequireApproval(t *testing.T) {
	never := true
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.mcp", Args: MCPArgs{
			ServerLabel:     "my-server",
			ServerURL:       strP("https://mcp.example.com"),
			RequireApproval: &MCPRequireApproval{Never: &never},
		}},
	})
	tools := extractTools(t, body)
	mcp, ok := tools["mcp"]
	if !ok {
		t.Fatalf("no mcp: %v", tools)
	}
	ra, ok := mcp["require_approval"].(map[string]any)
	if !ok {
		t.Fatalf("require_approval: %v", mcp["require_approval"])
	}
	if ra["never"] != true {
		t.Errorf("never: %v", ra["never"])
	}
}

// TestProviderToolMCPRequireApprovalToolNames verifies that the
// MCPRequireApproval.ToolNames field is serialized as tool_names (snake_case).
func TestProviderToolMCPRequireApprovalToolNames(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.mcp", Args: MCPArgs{
			ServerLabel: "s",
			RequireApproval: &MCPRequireApproval{
				ToolNames: []string{"read", "write"},
			},
		}},
	})
	tools := extractTools(t, body)
	mcp := tools["mcp"]
	ra, ok := mcp["require_approval"].(map[string]any)
	if !ok {
		t.Fatalf("require_approval: %v", mcp["require_approval"])
	}
	names, ok := ra["tool_names"].([]any)
	if !ok {
		t.Fatalf("tool_names: %v", ra)
	}
	if len(names) != 2 {
		t.Errorf("tool_names: %v", names)
	}
}

func TestProviderToolShell(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.shell", Args: ShellArgs{
			Environment: &ShellEnvironment{Type: "local"},
		}},
	})
	tools := extractTools(t, body)
	sh, ok := tools["shell"]
	if !ok {
		t.Fatalf("no shell: %v", tools)
	}
	env, ok := sh["environment"].(map[string]any)
	if !ok || env["type"] != "local" {
		t.Errorf("environment: %v", sh["environment"])
	}
}

// TestProviderToolLocalShell verifies local_shell has no args.
func TestProviderToolLocalShell(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.localShell", Args: LocalShellArgs{}},
	})
	tools := extractTools(t, body)
	ls, ok := tools["local_shell"]
	if !ok {
		t.Fatalf("no local_shell: %v", tools)
	}
	if ls["type"] != "local_shell" {
		t.Errorf("type: %v", ls["type"])
	}
}

// TestProviderToolApplyPatch verifies apply_patch has no args.
func TestProviderToolApplyPatch(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.applyPatch", Args: ApplyPatchArgs{}},
	})
	tools := extractTools(t, body)
	ap, ok := tools["apply_patch"]
	if !ok {
		t.Fatalf("no apply_patch: %v", tools)
	}
	if ap["type"] != "apply_patch" {
		t.Errorf("type: %v", ap["type"])
	}
}

// TestProviderToolToolSearch verifies tool_search Args include execution,
// description, parameters.
func TestProviderToolToolSearch(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.toolSearch", Args: ToolSearchArgs{
			Execution:   strP("server"),
			Description: strP("find me a tool"),
			Parameters:  map[string]any{"type": "object"},
		}},
	})
	tools := extractTools(t, body)
	ts, ok := tools["tool_search"]
	if !ok {
		t.Fatalf("no tool_search: %v", tools)
	}
	if ts["execution"] != "server" {
		t.Errorf("execution: %v", ts["execution"])
	}
	if ts["description"] != "find me a tool" {
		t.Errorf("description: %v", ts["description"])
	}
	if _, has := ts["parameters"]; !has {
		t.Errorf("parameters missing: %v", ts)
	}
}

// TestProviderToolCustom verifies custom Args pass through description and
// format.
func TestProviderToolCustom(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.custom", Args: CustomToolArgs{
			Description: "analyze code",
			Format: &CustomToolFormat{
				Type:       "grammar",
				Syntax:     "regex",
				Definition: "[a-z]+",
			},
		}},
	})
	tools := extractTools(t, body)
	cu, ok := tools["custom"]
	if !ok {
		t.Fatalf("no custom: %v", tools)
	}
	if cu["description"] != "analyze code" {
		t.Errorf("description: %v", cu["description"])
	}
	format, ok := cu["format"].(map[string]any)
	if !ok {
		t.Fatalf("format: %v", cu["format"])
	}
	if format["type"] != "grammar" {
		t.Errorf("format type: %v", format["type"])
	}
	if format["syntax"] != "regex" {
		t.Errorf("format syntax: %v", format["syntax"])
	}
	if format["definition"] != "[a-z]+" {
		t.Errorf("format definition: %v", format["definition"])
	}
}

// TestProviderToolWebSearchPreview verifies web_search_preview Args include
// search_context_size and user_location.
func TestProviderToolWebSearchPreview(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.webSearchPreview", Args: WebSearchPreviewArgs{
			SearchContextSize: strP("medium"),
			UserLocation: &WebSearchUserLocation{
				Type:    "approximate",
				Country: "US",
			},
		}},
	})
	tools := extractTools(t, body)
	ws, ok := tools["web_search_preview"]
	if !ok {
		t.Fatalf("no web_search_preview: %v", tools)
	}
	if ws["search_context_size"] != "medium" {
		t.Errorf("search_context_size: %v", ws["search_context_size"])
	}
	if _, has := ws["user_location"]; !has {
		t.Errorf("user_location missing: %v", ws)
	}
}

// TestCamelToSnake verifies the helper for the MCP tool key conversion.
func TestCamelToSnake(t *testing.T) {
	cases := map[string]string{
		"serverLabel":       "server_label",
		"serverUrl":         "server_url",
		"allowedTools":      "allowed_tools",
		"connectorId":       "connector_id",
		"requireApproval":   "require_approval",
		"serverDescription": "server_description",
		"foo":               "foo",
		"":                  "",
	}
	for in, want := range cases {
		if got := camelToSnake(in); got != want {
			t.Errorf("camelToSnake(%q) = %q, want %q", in, got, want)
		}
	}
}
