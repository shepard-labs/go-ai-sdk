package anthropic

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

type spoofPreSendError struct{ cause error }

func (e spoofPreSendError) Error() string { return e.cause.Error() }
func (e spoofPreSendError) Unwrap() error { return e.cause }
func (e spoofPreSendError) As(target any) bool {
	marked, ok := target.(**preSendTransportError)
	if ok {
		*marked = &preSendTransportError{cause: e.cause}
	}
	return ok
}

func TestPreSendMarkerCannotBeSpoofedByAs(t *testing.T) {
	if isPreSendTransportError(spoofPreSendError{cause: errors.New("spoof")}) {
		t.Fatal("spoofed As method was accepted as package marker")
	}
	if !isPreSendTransportError(fmt.Errorf("outer: %w", NewPreSendTransportError(errors.New("dial")))) {
		t.Fatal("real marker was not found through ordinary wrapping")
	}
}

type tracedRoundTripper struct {
	wrote bool
	err   error
}

func (t tracedRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.wrote {
		if trace := httptrace.ContextClientTrace(req.Context()); trace != nil && trace.WroteRequest != nil {
			trace.WroteRequest(httptrace.WroteRequestInfo{})
		}
	}
	return nil, t.err
}

func TestPreSendMarkingTransportBoundary(t *testing.T) {
	tests := []struct {
		name  string
		wrote bool
		err   error
		want  bool
	}{
		{name: "dns before write", err: &net.DNSError{Err: "no such host", Name: "example.test"}, want: true},
		{name: "dial refusal before write", err: &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}, want: true},
		{name: "tls handshake before write", err: tls.RecordHeaderError{}, want: true},
		{name: "reset after write", wrote: true, err: syscall.ECONNRESET},
		{name: "eof after write", wrote: true, err: io.EOF},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodPost, "https://example.test", nil)
			if err != nil {
				t.Fatal(err)
			}
			_, gotErr := (preSendMarkingTransport{base: tracedRoundTripper{wrote: tt.wrote, err: tt.err}}).RoundTrip(req)
			if got := isPreSendTransportError(gotErr); got != tt.want {
				t.Fatalf("isPreSendTransportError() = %v, want %v; err = %v", got, tt.want, gotErr)
			}
		})
	}
}

func TestDefaultHTTPClientDoesNotFollowRedirect(t *testing.T) {
	var redirectRequests atomic.Int32
	redirectTarget := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirectRequests.Add(1)
		if got := r.Header.Get("x-api-key"); got != "" {
			t.Errorf("redirect target received x-api-key %q", got)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer redirectTarget.Close()

	var initialRequests atomic.Int32
	initialServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		initialRequests.Add(1)
		w.Header().Set("Location", redirectTarget.URL)
		w.WriteHeader(http.StatusTemporaryRedirect)
	}))
	defer initialServer.Close()

	req, err := http.NewRequest(http.MethodPost, initialServer.URL, strings.NewReader(`{"message":"hello"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("x-api-key", "secret")
	resp, err := defaultHTTPClient().Do(req)
	if err != nil {
		t.Fatalf("Do returned transport error for redirect response: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusTemporaryRedirect)
	}
	if got := initialRequests.Load(); got != 1 {
		t.Fatalf("initial requests = %d, want 1", got)
	}
	if got := redirectRequests.Load(); got != 0 {
		t.Fatalf("redirect target requests = %d, want 0", got)
	}
}

// Expected declarations are copied from @ai-sdk/anthropic@3.0.115
// src/anthropic-prepare-tools.ts:140-366 and checked against 4.0.46:137-365.
func TestProviderToolsUseAuthoritativeWireDeclarations(t *testing.T) {
	one, tokens, width, height, display := 1, 2048, 800, 600, 0
	zoom := false
	citations := &CitationsConfig{Enabled: true}
	location := &UserLocation{Type: "approximate", City: "San Francisco", Region: "California", Country: "US", Timezone: "America/Los_Angeles"}
	tests := []struct {
		tool Tool
		want string
	}{
		{Tool{ID: "anthropic.bash_20241022"}, `{"name":"bash","type":"bash_20241022"}`},
		{Tool{ID: "anthropic.bash_20250124"}, `{"name":"bash","type":"bash_20250124"}`},
		{Tool{ID: "anthropic.code_execution_20250522"}, `{"name":"code_execution","type":"code_execution_20250522"}`},
		{Tool{ID: "anthropic.code_execution_20250825"}, `{"name":"code_execution","type":"code_execution_20250825"}`},
		{Tool{ID: "anthropic.code_execution_20260120"}, `{"name":"code_execution","type":"code_execution_20260120"}`},
		{Tool{ID: "anthropic.computer_20251124", DisplayWidthPx: &width, DisplayHeightPx: &height, DisplayNumber: &display, EnableZoom: &zoom}, `{"name":"computer","type":"computer_20251124","display_width_px":800,"display_height_px":600,"display_number":0,"enable_zoom":false}`},
		{Tool{ID: "anthropic.text_editor_20250728", MaxCharacters: &one}, `{"name":"str_replace_based_edit_tool","type":"text_editor_20250728","max_characters":1}`},
		{Tool{ID: "anthropic.memory_20250818"}, `{"name":"memory","type":"memory_20250818"}`},
		{Tool{ID: "anthropic.web_fetch_20250910", MaxUses: &one, AllowedDomains: []string{"allowed.example"}, BlockedDomains: []string{"blocked.example"}, Citations: citations, MaxContentTokens: &tokens}, `{"name":"web_fetch","type":"web_fetch_20250910","max_uses":1,"allowed_domains":["allowed.example"],"blocked_domains":["blocked.example"],"citations":{"enabled":true},"max_content_tokens":2048}`},
		{Tool{ID: "anthropic.web_fetch_20260209", MaxUses: &one, AllowedDomains: []string{"allowed.example"}, BlockedDomains: []string{"blocked.example"}, Citations: citations, MaxContentTokens: &tokens}, `{"name":"web_fetch","type":"web_fetch_20260209","max_uses":1,"allowed_domains":["allowed.example"],"blocked_domains":["blocked.example"],"citations":{"enabled":true},"max_content_tokens":2048}`},
		{Tool{ID: "anthropic.web_search_20250305", MaxUses: &one, AllowedDomains: []string{"allowed.example"}, BlockedDomains: []string{"blocked.example"}, UserLocation: location}, `{"name":"web_search","type":"web_search_20250305","max_uses":1,"allowed_domains":["allowed.example"],"blocked_domains":["blocked.example"],"user_location":{"type":"approximate","city":"San Francisco","region":"California","country":"US","timezone":"America/Los_Angeles"}}`},
		{Tool{ID: "anthropic.web_search_20260209", MaxUses: &one, AllowedDomains: []string{"allowed.example"}, BlockedDomains: []string{"blocked.example"}, UserLocation: location}, `{"name":"web_search","type":"web_search_20260209","max_uses":1,"allowed_domains":["allowed.example"],"blocked_domains":["blocked.example"],"user_location":{"type":"approximate","city":"San Francisco","region":"California","country":"US","timezone":"America/Los_Angeles"}}`},
		{Tool{ID: "anthropic.tool_search_regex_20251119"}, `{"name":"tool_search_tool_regex","type":"tool_search_tool_regex_20251119"}`},
		{Tool{ID: "anthropic.tool_search_bm25_20251119"}, `{"name":"tool_search_tool_bm25","type":"tool_search_tool_bm25_20251119"}`},
		{Tool{ID: "anthropic.advisor_20260301", Name: "sdk-name", Description: "sdk description", InputSchema: "sdk input", ArgsSchema: "sdk args", OutputSchema: "sdk output", Dynamic: true, ProviderExecuted: true, AdvisorModel: "claude-opus-4-8", AdvisorMaxUses: &one, AdvisorCaching: &CachingConfig{true}}, `{"name":"advisor","type":"advisor_20260301","max_uses":1,"model":"claude-opus-4-8","caching":{"type":"ephemeral","ttl":"5m"}}`},
	}
	for _, tt := range tests {
		wire, replaySafe, err := toolsToWire([]Tool{tt.tool})
		if err != nil {
			t.Fatalf("%s: %v", tt.tool.ID, err)
		}
		if replaySafe {
			t.Fatalf("%s marked replay safe", tt.tool.ID)
		}
		got, err := json.Marshal(wire[0])
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != tt.want {
			t.Fatalf("%s = %s, want %s", tt.tool.ID, got, tt.want)
		}
	}
}

func TestOrdinaryClientToolExactWireDeclaration(t *testing.T) {
	trueValue := true
	tool := Tool{ID: "sdk-only", Name: "lookup", Description: "Look up a record", InputSchema: map[string]any{"type": "object"}, ArgsSchema: "sdk args", OutputSchema: "sdk output", CacheControl: &CacheControl{Type: "ephemeral", TTL: "5m"}, EagerInputStreaming: &trueValue, Strict: &trueValue, DeferLoading: &trueValue, AllowedCallers: []ToolCallCaller{"direct"}, InputExamples: []any{map[string]any{"id": "123"}}, Dynamic: true, ProviderExecuted: true}
	tool.ID = ""
	wire, replaySafe, err := toolsToWire([]Tool{tool})
	if err != nil || !replaySafe {
		t.Fatalf("wire conversion: replaySafe=%v err=%v", replaySafe, err)
	}
	got, err := json.Marshal(wire[0])
	if err != nil {
		t.Fatal(err)
	}
	want := `{"name":"lookup","description":"Look up a record","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral","ttl":"5m"},"eager_input_streaming":true,"strict":true,"defer_loading":true,"allowed_callers":["direct"],"input_examples":[{"id":"123"}]}`
	if string(got) != want {
		t.Fatalf("tool = %s, want %s", got, want)
	}
}

func TestToolWireConstructionFailsClosedBeforeTransport(t *testing.T) {
	var attempts atomic.Int32
	model := CreateAnthropic(ProviderSettings{Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { attempts.Add(1); return nil, errors.New("unexpected") }}}).Model("claude-3-haiku").(*anthropicLanguageModel)
	zero := 0
	for index, tool := range []Tool{{ID: "anthropic.unknown"}, {ID: "anthropic.computer_20251124"}, {ID: "anthropic.web_fetch_20260209", MaxUses: &zero}} {
		_, err := model.DoGenerate(context.Background(), GenerateOptions{Tools: []Tool{tool}})
		var constructionErr *RequestConstructionError
		if !errors.As(err, &constructionErr) {
			t.Fatalf("error = %T %v", err, err)
		}
		if constructionErr.ToolIndex != 0 || constructionErr.ToolID != tool.ID || constructionErr.Message == "" {
			t.Fatalf("tool %d error = %#v", index, constructionErr)
		}
	}
	if attempts.Load() != 0 {
		t.Fatalf("transport attempts = %d", attempts.Load())
	}
}

func TestToolRequestsRetrySafety(t *testing.T) {
	for _, tt := range []struct {
		name         string
		modelOptions ModelOptions
		tools        []Tool
		wantAttempts int32
	}{
		{"provider tool", ModelOptions{}, []Tool{{ID: "anthropic.code_execution_20250522"}}, 1},
		{"server tool", ModelOptions{}, []Tool{{ID: "anthropic.web_search_20260209"}}, 1},
		{"MCP toolset", ModelOptions{MCPServers: []MCPServer{{Name: "server"}}}, nil, 1},
		{"ordinary client tool", ModelOptions{}, []Tool{{Name: "lookup"}}, 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var attempts atomic.Int32
			model := CreateAnthropic(ProviderSettings{Retry: &RetryOptions{MaxRetries: 2, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}, Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
				attempts.Add(1)
				return jsonResponse(500, `{"error":{"type":"api_error","message":"failed"}}`), nil
			}}}).Model("claude-3-haiku", tt.modelOptions)
			_, err := model.DoGenerate(context.Background(), GenerateOptions{Tools: tt.tools})
			if err == nil {
				t.Fatal("DoGenerate error = nil")
			}
			if attempts.Load() != tt.wantAttempts {
				t.Fatalf("attempts = %d, want %d", attempts.Load(), tt.wantAttempts)
			}
		})
	}
}

type roundTripFetcher struct {
	fn func(*http.Request) (*http.Response, error)
}

func (f roundTripFetcher) Do(req *http.Request) (*http.Response, error) { return f.fn(req) }

func TestPromptConvertTextOnly(t *testing.T) {
	system, messages := convertPrompt([]Message{
		SystemMessage{Content: "system"},
		UserMessage{Content: []UserContent{TextContent{Text: "hello"}}},
	})
	if len(system) != 1 || system[0].Text != "system" {
		t.Fatalf("system = %#v", system)
	}
	if len(messages) != 1 || messages[0].Role != "user" || messages[0].Content[0].Text != "hello" {
		t.Fatalf("messages = %#v", messages)
	}
}

func TestPromptConvertWithImages(t *testing.T) {
	_, messages := convertPrompt([]Message{UserMessage{Content: []UserContent{ImageContent{Source: ImageSource{MediaType: "image/png", Data: "abc"}}}}})
	block := messages[0].Content[0]
	if block.Type != "image" || block.Source.Type != "base64" || block.Source.MediaType != "image/png" || block.Source.Data != "abc" {
		t.Fatalf("image block = %#v", block)
	}
}

func TestPromptConvertWithDocuments(t *testing.T) {
	_, messages := convertPrompt([]Message{UserMessage{Content: []UserContent{DocumentContent{Source: DocumentSource{MediaType: "application/pdf", Data: "abc"}}}}})
	block := messages[0].Content[0]
	if block.Type != "document" || block.Source.Type != "base64" || block.Source.MediaType != "application/pdf" || block.Source.Data != "abc" {
		t.Fatalf("document block = %#v", block)
	}
}

func TestPromptConvertWithAssistantContent(t *testing.T) {
	_, messages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{
		TextContent{Text: "answer"},
		ThinkingContent{Thinking: "think", Signature: "sig"},
		RedactedThinkingContent{Data: "redacted"},
		CompactionContent{Text: "compact"},
	}}})
	blocks := messages[0].Content
	if blocks[0].Type != "text" || blocks[0].Text != "answer" {
		t.Fatalf("text block = %#v", blocks[0])
	}
	if blocks[1].Type != "thinking" || blocks[1].Thinking != "think" || blocks[1].Signature != "sig" {
		t.Fatalf("thinking block = %#v", blocks[1])
	}
	if blocks[2].Type != "redacted_thinking" || blocks[2].Data != "redacted" {
		t.Fatalf("redacted block = %#v", blocks[2])
	}
	if blocks[3].Type != "compaction" || blocks[3].Text != "compact" {
		t.Fatalf("compaction block = %#v", blocks[3])
	}
}

func TestBuildRequest(t *testing.T) {
	temp := 0.5
	topK := 3
	topP := 0.8
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	req, warnings := model.buildRequest(GenerateOptions{
		Messages:      []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}},
		MaxTokens:     100,
		Temperature:   &temp,
		TopK:          &topK,
		TopP:          &topP,
		StopSequences: []string{"stop"},
	}, false)
	if req.Model != "claude-3-haiku-20240307" || req.MaxTokens != 100 || req.Stream {
		t.Fatalf("request = %#v", req)
	}
	if *req.Temperature != 0.5 || *req.TopK != 3 || req.TopP != nil || req.StopSequences[0] != "stop" {
		t.Fatalf("request params = %#v", req)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestBuildRequestWithModelOptions(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-opus-4-6", ModelOptions{
		Metadata:     &Metadata{UserID: "user"},
		Speed:        "fast",
		InferenceGeo: "us",
		Container:    &Container{ID: "container"},
		MCPServers:   []MCPServer{{Name: "mcp"}},
	}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{}, false)
	if req.Metadata.UserID != "user" || req.Speed != "fast" || req.InferenceGeo != "us" || req.Container.ID != "container" || req.MCPServers[0].Name != "mcp" {
		t.Fatalf("request = %#v", req)
	}
}

func TestBuildRequestWithToolsAndToolChoice(t *testing.T) {
	deferLoading := true
	eager := true
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307").(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{
		Tools:       []Tool{{Name: "weather", Description: "Get weather", InputSchema: map[string]any{"type": "object"}}},
		ToolOptions: ToolOptions{DeferLoading: &deferLoading, EagerInputStreaming: &eager, AllowedCallers: []ToolCallCaller{"code"}},
		ToolChoice:  &ToolChoice{Type: "required", Name: "weather", DisableParallelToolUse: true},
	}, false)
	if len(req.Tools) != 1 || req.Tools[0].Name != "weather" || req.Tools[0].DeferLoading == nil || req.Tools[0].AllowedCallers[0] != "code" {
		t.Fatalf("tools = %#v", req.Tools)
	}
	if req.ToolChoice.Type != "tool" || req.ToolChoice.Name != "weather" || !req.ToolChoice.DisableParallelToolUse || !req.DisableParallelToolUse {
		t.Fatalf("request = %#v", req)
	}
}

func TestBuildRequestWithProviderToolsAndModelOptions(t *testing.T) {
	budget := 100
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307", ModelOptions{SendReasoning: true, DisableParallelToolUse: true, Effort: "xhigh", TaskBudget: &budget}).(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{Tools: []Tool{{ID: "anthropic.code_execution_20250522", Name: "code_execution_20250522", Type: "anthropic.code_execution_20250522", ProviderExecuted: true}}}, false)
	if len(req.Tools) != 1 || req.Tools[0].ID != "anthropic.code_execution_20250522" || req.ToolChoice.Type != "auto" || !req.ToolChoice.DisableParallelToolUse || !req.DisableParallelToolUse || req.OutputConfig.Effort != "xhigh" || req.OutputConfig.TaskBudget.Total != 100 {
		t.Fatalf("request = %#v", req)
	}
}

func TestBuildRequestWithStructuredOutputUsesOutputConfigJSONSchema(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-sonnet-4-6").(*anthropicLanguageModel)
	req, _ := model.buildRequest(GenerateOptions{StructuredOutput: &StructuredOutput{
		Name:   "summary",
		Schema: map[string]any{"type": "object", "properties": map[string]any{"title": map[string]any{"type": "string"}}},
	}}, false)
	if req.OutputConfig == nil || req.OutputConfig.Format == nil || req.OutputConfig.Format.Type != "json_schema" {
		t.Fatalf("output_config = %#v", req.OutputConfig)
	}
	if len(req.Tools) != 0 || req.ToolChoice != nil {
		t.Fatalf("request = %#v", req)
	}
}

func TestBetaHeadersProviderTools(t *testing.T) {
	p := CreateAnthropic(ProviderSettings{}).(*anthropicProvider)
	headers := p.headersForOptions(ModelOptions{RequestTools: []Tool{{ID: "anthropic.code_execution_20250522"}, {ID: "anthropic.web_fetch_20260209"}}})
	values := headers.Values("anthropic-beta")
	if len(values) != 2 || values[0] != "code-execution-2025-05-22" || values[1] != "code-execution-web-tools-2026-02-09" {
		t.Fatalf("headers = %#v", values)
	}
}

func TestPromptConvertToolResultsAndAssistantToolCalls(t *testing.T) {
	_, userMessages := convertPrompt([]Message{UserMessage{Content: []UserContent{ToolResultContent{ToolCallID: "call", IsError: true, Result: []ToolResultPart{ToolResultText{Text: "ok"}, ToolResultReference{ID: "ref"}}}}}})
	result := userMessages[0].Content[0]
	if result.Type != "tool_result" || result.ToolUseID != "call" || !result.IsError {
		t.Fatalf("tool result = %#v", result)
	}

	_, assistantMessages := convertPrompt([]Message{AssistantMessage{Content: []AssistantContent{
		ToolCallContent{ToolCallID: "call", ToolName: "weather", Input: json.RawMessage(`{"city":"SF"}`)},
		ServerToolUseContent{ID: "server", Name: "web_search", Input: json.RawMessage(`{"query":"x"}`)},
		MCPToolUseContent{ID: "mcp", Name: "tool", ServerName: "server", Input: json.RawMessage(`{"a":1}`)},
	}}})
	blocks := assistantMessages[0].Content
	if blocks[0].Type != "server_tool_use" || blocks[1].Type != "mcp_tool_use" || blocks[1].ServerName != "server" || blocks[2].Type != "tool_use" || blocks[2].Name != "weather" {
		t.Fatalf("blocks = %#v", blocks)
	}
}

func TestTemperatureClamping(t *testing.T) {
	high := 2.0
	low := -1.0
	got, warnings := normalizeTemperature(&high)
	if *got != 1.0 || len(warnings) != 1 {
		t.Fatalf("high = %v %#v", *got, warnings)
	}
	got, warnings = normalizeTemperature(&low)
	if *got != 0 || len(warnings) != 1 {
		t.Fatalf("low = %v %#v", *got, warnings)
	}
}

func TestUnsupportedParameterWarnings(t *testing.T) {
	value := 1.0
	seed := 1
	warnings := unsupportedWarnings(GenerateOptions{FrequencyPenalty: &value, PresencePenalty: &value, Seed: &seed, TopP: &value, Temperature: &value}, ModelCapabilities{})
	if len(warnings) != 3 {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestBuildRequestOmitsTopPWhenTemperatureSet(t *testing.T) {
	temperature := 0.2
	topP := 0.9
	model := CreateAnthropic(ProviderSettings{}).Model("claude-sonnet-4-5").(*anthropicLanguageModel)
	req, warnings := model.buildRequest(GenerateOptions{Temperature: &temperature, TopP: &topP}, false)
	if req.Temperature == nil || *req.Temperature != temperature || req.TopP != nil {
		t.Fatalf("request = %#v", req)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestBuildRequestOmitsSamplingForModelsThatRejectIt(t *testing.T) {
	temperature := 0.2
	topP := 0.9
	topK := 40
	model := CreateAnthropic(ProviderSettings{}).Model("claude-opus-4-8").(*anthropicLanguageModel)
	req, warnings := model.buildRequest(GenerateOptions{Temperature: &temperature, TopP: &topP, TopK: &topK}, false)
	if req.Temperature != nil || req.TopP != nil || req.TopK != nil {
		t.Fatalf("request = %#v", req)
	}
	if len(warnings) != 3 {
		t.Fatalf("warnings = %#v", warnings)
	}
}

func TestUsageConversionWithCacheAndIterations(t *testing.T) {
	usage := UsageFromResponseUsage(ResponseUsage{InputTokens: 10, OutputTokens: 5, CacheCreationInputTokens: 3, CacheReadInputTokens: 2, Iterations: []UsageIteration{{InputTokens: 1, OutputTokens: 1}}})
	if usage.InputTokens.Total != 10 || usage.InputTokens.CacheWrite != 3 || usage.InputTokens.CacheRead != 2 || usage.OutputTokens.Total != 5 || usage.TotalTokens != 15 || len(usage.Iterations) != 1 {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestDoGenerateTextOnly(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		if req.URL.Path != "/messages" {
			t.Fatalf("path = %q", req.URL.Path)
		}
		if req.Header.Get("x-api-key") != "key" || req.Header.Get("anthropic-version") == "" {
			t.Fatalf("headers = %#v", req.Header)
		}
		body, _ := io.ReadAll(req.Body)
		var decoded apiMessagesRequest
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Messages[0].Content[0].Text != "hello" {
			t.Fatalf("body = %s", body)
		}
		return jsonResponse(200, `{"id":"msg_1","model":"claude-3-haiku-20240307","role":"assistant","content":[{"type":"text","text":"answer"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content[0].(TextContent).Text != "answer" || result.FinishReason != FinishReasonStop || result.Usage.InputTokens.Total != 2 || result.Usage.OutputTokens.Total != 3 {
		t.Fatalf("result = %#v", result)
	}
}

func TestDoGenerateAPIError(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		resp := jsonResponse(400, `{"error":{"type":"invalid_request_error","message":"bad"}}`)
		resp.Header.Set("x-request-id", "req_123")
		return resp, nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || apiErr.Status != 400 || apiErr.Message != "bad" || apiErr.Type != "invalid_request_error" || apiErr.RequestID != "req_123" || apiErr.Retryable {
		t.Fatalf("err = %#v", err)
	}
}

func TestDoGenerateAPIErrorBodyLimit(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		return jsonResponse(500, strings.Repeat("x", 64)), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 0}, MaxErrorResponseBytes: 8}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || !apiErr.Truncated || len(apiErr.Body) != 8 || !apiErr.Retryable {
		t.Fatalf("err = %#v", err)
	}
}

func TestDoGenerateSuccessBodyLimit(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		return jsonResponse(200, `{"content":[{"type":"text","text":"too large"}]}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, MaxResponseBodyBytes: 8}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || !apiErr.Truncated || apiErr.Status != 200 || apiErr.Retryable {
		t.Fatalf("err = %#v", err)
	}
}

func TestDoStreamTextOnly(t *testing.T) {
	sse := "event: message_start\ndata: {\"message\":{\"id\":\"msg_1\",\"model\":\"claude-provider-model\"}}\n\n" +
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	var parts []StreamPart
	terminalReceived := false
	for part := range result.Stream {
		parts = append(parts, part)
		_, terminalReceived = part.(StreamFinish)
	}
	if !terminalReceived {
		t.Fatal("stream closed without terminal event")
	}
	if len(parts) != 6 || parts[0].(StreamStart) != (StreamStart{}) || parts[1].(StreamResponseMetadata).ID != "msg_1" || parts[1].(StreamResponseMetadata).ModelID != "claude-provider-model" || parts[3].(StreamTextDelta).Text != "hi" || parts[5].(StreamFinish).Usage.OutputTokens.Total != 1 || result.Response.ID != "msg_1" || result.Response.ModelID != "claude-provider-model" {
		t.Fatalf("parts = %#v response = %#v", parts, result.Response)
	}
}

type blockingStreamBody struct {
	closed chan struct{}
	once   sync.Once
}

func (b *blockingStreamBody) Read([]byte) (int, error) {
	<-b.closed
	return 0, io.ErrClosedPipe
}

func (b *blockingStreamBody) Close() error {
	b.once.Do(func() { close(b.closed) })
	return nil
}

func TestDoStreamCancelInterruptsBlockedBodyRead(t *testing.T) {
	body := &blockingStreamBody{closed: make(chan struct{})}
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: body, Request: req}, nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	result.Cancel()
	select {
	case part, open := <-result.Stream:
		if !open {
			t.Fatal("stream closed without cancellation error")
		}
		streamErr, ok := part.(StreamError)
		if !ok || !errors.Is(streamErr.Err, context.Canceled) {
			t.Fatalf("terminal part = %#v, want StreamError{context.Canceled}", part)
		}
	case <-time.After(time.Second):
		t.Fatal("stream did not emit cancellation error")
	}
	select {
	case _, open := <-result.Stream:
		if open {
			t.Fatal("stream emitted output after cancellation error")
		}
	case <-time.After(time.Second):
		t.Fatal("stream did not close after cancellation error")
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("stream body was not closed after cancellation")
	}
}

type fullStreamBufferBody struct {
	events      []string
	next        int
	closed      chan struct{}
	readBlocked chan struct{}
	readExited  chan struct{}
	once        sync.Once
}

func (b *fullStreamBufferBody) Read(p []byte) (int, error) {
	if b.next < len(b.events) {
		event := b.events[b.next]
		b.next++
		return copy(p, event), nil
	}
	close(b.readBlocked)
	<-b.closed
	close(b.readExited)
	return 0, io.ErrClosedPipe
}

func (b *fullStreamBufferBody) Close() error {
	b.once.Do(func() { close(b.closed) })
	return nil
}

func TestDoStreamCancelReplacesFullOutputBufferWithCancellationError(t *testing.T) {
	events := []string{event("message_start", `{"message":{"id":"m","model":"claude"}}`)}
	for i := 0; i < 5; i++ {
		events = append(events,
			event("content_block_start", fmt.Sprintf(`{"index":%d,"content_block":{"type":"text"}}`, i)),
			event("content_block_delta", fmt.Sprintf(`{"index":%d,"delta":{"type":"text_delta","text":"x"}}`, i)),
			event("content_block_stop", fmt.Sprintf(`{"index":%d}`, i)),
		)
	}
	body := &fullStreamBufferBody{events: events, closed: make(chan struct{}), readBlocked: make(chan struct{}), readExited: make(chan struct{})}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: body, Request: req}, nil
	}}}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-body.readBlocked:
	case <-time.After(time.Second):
		t.Fatal("stream did not fill its output buffer and block the body read")
	}
	deadline := time.After(time.Second)
	for len(result.Stream) != cap(result.Stream) {
		select {
		case <-deadline:
			t.Fatalf("stream buffer = %d/%d, want full", len(result.Stream), cap(result.Stream))
		default:
			time.Sleep(time.Millisecond)
		}
	}

	result.Cancel()
	select {
	case <-body.closed:
	case <-time.After(time.Second):
		t.Fatal("stream body was not closed after cancellation")
	}
	select {
	case <-body.readExited:
	case <-time.After(time.Second):
		t.Fatal("blocked body read did not exit after cancellation")
	}

	var cancellationErrors, finishes int
	stream := result.Stream
	for stream != nil {
		select {
		case part, open := <-result.Stream:
			if !open {
				stream = nil
				continue
			}
			switch part := part.(type) {
			case StreamError:
				if !errors.Is(part.Err, context.Canceled) {
					t.Fatalf("StreamError = %v, want context.Canceled", part.Err)
				}
				cancellationErrors++
			case StreamFinish:
				finishes++
			}
		case <-time.After(time.Second):
			t.Fatal("stream did not close after cancellation")
		}
	}
	if cancellationErrors != 1 || finishes != 0 {
		t.Fatalf("terminal events: StreamError=%d StreamFinish=%d, want 1 and 0", cancellationErrors, finishes)
	}
}

func TestModelConcurrentGenerateReuse(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(req.Body)
		var decoded apiMessagesRequest
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Messages[0].Content[0].Text == "" {
			t.Fatalf("body = %s", body)
		}
		return jsonResponse(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := model.DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, Tools: []Tool{{Name: "tool", InputSchema: map[string]any{"type": "object"}}}, MaxTokens: 10})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}

func TestDoStreamLargeDataLine(t *testing.T) {
	large := strings.Repeat("a", 128*1024)
	sse := "event: message_start\ndata: {\"message\":{\"id\":\"msg\",\"model\":\"claude\"}}\n\n" +
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"" + large + "\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":0}\n\nevent: message_stop\ndata: {}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	<-result.Stream
	<-result.Stream
	<-result.Stream
	part := <-result.Stream
	delta, ok := part.(StreamTextDelta)
	if !ok || len(delta.Text) != len(large) {
		t.Fatalf("part = %#v", part)
	}
	for part := range result.Stream {
		if err, ok := part.(StreamError); ok {
			t.Fatalf("stream error: %v", err.Err)
		}
	}
}

func TestDoGenerateRetriesRetryableStatus(t *testing.T) {
	var calls int32
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		if atomic.AddInt32(&calls, 1) == 1 {
			return jsonResponse(429, `{"error":{"type":"rate_limit_error","message":"limited"}}`), nil
		}
		return jsonResponse(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 1, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}}).Model("claude-3-haiku-20240307")
	result, err := model.DoGenerate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content[0].(TextContent).Text != "ok" || atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("result = %#v calls = %d", result, calls)
	}
}

func TestDoGenerateRetriesCanBeDisabled(t *testing.T) {
	var calls int32
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		atomic.AddInt32(&calls, 1)
		return jsonResponse(500, `{"error":{"type":"server_error","message":"bad"}}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 0}}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	if err == nil || atomic.LoadInt32(&calls) != 1 {
		t.Fatalf("err = %v calls = %d", err, calls)
	}
}

func TestDoGenerateNilResponseFetcher(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return nil, nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Retry: &RetryOptions{MaxRetries: 0}}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	apiErr, ok := err.(*APICallError)
	if !ok || apiErr.Message == "" {
		t.Fatalf("err = %#v", err)
	}
}

func TestResponseContentParsing(t *testing.T) {
	body := []byte(`{"content":[{"type":"thinking","thinking":"think","signature":"sig"},{"type":"redacted_thinking","data":"hidden"},{"type":"compaction","text":"compact"},{"type":"source","id":"src","url":"https://example.com","title":"title","media_type":"text/html","filename":"a.html"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":2}}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != FinishReasonLength || result.Content[0].(ReasoningContent).Type != "reasoning" || result.Content[1].(ReasoningContent).Type != "redacted_reasoning" || result.Content[2].(CompactionContent).Text != "compact" || result.Content[3].(SourceContent).Filename != "a.html" {
		t.Fatalf("result = %#v", result)
	}
}

func TestResponseParseToolCallsAndResults(t *testing.T) {
	body := []byte(`{"content":[{"type":"tool_use","id":"call","name":"weather","input":{"city":"SF"},"dynamic":true,"provider_metadata":{"a":1}},{"type":"server_tool_use","id":"server","name":"web_search","input":{"query":"x"}},{"type":"mcp_tool_result","tool_use_id":"mcp","content":[{"type":"text","text":"done"}],"is_error":true,"provider_executed":true}],"stop_reason":"tool_use"}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	call := result.Content[0].(ToolCallContent)
	server := result.Content[1].(ServerToolUseContent)
	toolResult := result.Content[2].(ToolResultContent)
	if result.FinishReason != FinishReasonToolCalls || call.Type != "tool-call" || !call.Dynamic || server.ID != "server" || server.Name != "web_search" || string(server.Input) != `{"query":"x"}` || toolResult.Type != "mcp_tool_result" || toolResult.ToolCallID != "mcp" || !toolResult.IsError || !toolResult.ProviderExecuted || toolResult.Result[0].(ToolResultText).Text != "done" {
		t.Fatalf("result = %#v", result)
	}
}

func TestStreamToolEvents(t *testing.T) {
	sse := "event: message_start\ndata: {\"message\":{\"id\":\"msg\",\"model\":\"claude\"}}\n\n" +
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"input_json\",\"name\":\"weather\",\"dynamic\":true}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":0}\n\n" +
		"event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call\",\"name\":\"weather\",\"input\":{\"city\":\"SF\"},\"provider_metadata\":{\"x\":1}}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":1}\n\n" +
		"event: content_block_start\ndata: {\"index\":2,\"content_block\":{\"type\":\"tool_result\",\"tool_use_id\":\"call\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}],\"provider_executed\":true}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":2}\n\n" +
		"event: content_block_start\ndata: {\"index\":3,\"content_block\":{\"type\":\"source\",\"id\":\"src\",\"url\":\"https://example.com\",\"title\":\"Example\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":3}\n\n" +
		"event: message_stop\ndata: {}\n\n"
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
	if parts[2].(StreamToolInputStart).ToolName != "weather" || !parts[2].(StreamToolInputStart).Dynamic || parts[3].(StreamToolInputDelta).Delta.(InputJSONDelta).PartialJSON == "" || parts[4].(StreamToolInputEnd).ID != "0" || parts[7].(StreamToolCall).ToolName != "weather" || parts[8].(StreamToolResult).Result[0].(ToolResultText).Text != "ok" || parts[9].(StreamSource).URL != "https://example.com" {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestSupportURLs(t *testing.T) {
	model := CreateAnthropic(ProviderSettings{}).Model("claude-3-haiku-20240307")
	urls := model.SupportURLs()
	if len(urls["anthropic"]) == 0 || !urls["anthropic"][0].MatchString("https://docs.anthropic.com/en/docs") {
		t.Fatalf("urls = %#v", urls)
	}
	if model.Provider() != "anthropic" {
		t.Fatalf("provider = %q", model.Provider())
	}
}

func jsonResponse(status int, body string) *http.Response {
	resp := textResponse(status, body)
	resp.Header.Set("Content-Type", "application/json")
	return resp
}

func textResponse(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}
