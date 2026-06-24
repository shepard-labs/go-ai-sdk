package google

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// ---- helpers ----

func newTestProvider(t *testing.T, handler http.Handler) *googleProvider {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	var counter int64
	return &googleProvider{
		baseURL:                        server.URL,
		name:                           "google.generative-ai",
		headers:                        http.Header{"User-Agent": []string{"ai-sdk-go/google/0.1.0"}},
		logger:                         testLogger{t: t},
		fetch:                          defaultHTTPClient(),
		generateID:                     func() string { return fmt.Sprintf("test-%d", atomicAdd(&counter)) },
		retry:                          RetryOptions{MaxRetries: 0, BaseDelay: 0, MaxDelay: 0, Jitter: false},
		maxResponseBodyBytes:           32 << 20,
		maxErrorResponseBytes:          1 << 20,
		maxEmbeddingsPerCall:           2048,
		supportsParallelEmbeddingCalls: true,
	}
}

func atomicAdd(p *int64) int64 {
	return atomic.AddInt64(p, 1)
}

type testLogger struct{ t *testing.T }

func (l testLogger) Debug(msg string, args ...any) { l.t.Log("DEBUG:", msg, args) }
func (l testLogger) Info(msg string, args ...any)  { l.t.Log("INFO:", msg, args) }
func (l testLogger) Warn(msg string, args ...any)  { l.t.Log("WARN:", msg, args) }
func (l testLogger) Error(msg string, args ...any) { l.t.Log("ERROR:", msg, args) }

// captureRequest returns a handler that decodes the request body and
// invokes respond with the response body.
func captureRequest(t *testing.T, respond func(body []byte) (int, []byte)) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		status, respBody := respond(body)
		w.WriteHeader(status)
		_, _ = w.Write(respBody)
	})
}

func decodeBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	return out
}

// ---- DoGenerate end-to-end ----

func TestDoGenerate_TextOnly(t *testing.T) {
	p := newTestProvider(t, captureRequest(t, func(body []byte) (int, []byte) {
		// Echo the prompt text back.
		req := decodeBody(t, body)
		contents, _ := req["contents"].([]any)
		first := contents[0].(map[string]any)
		parts := first["parts"].([]any)
		userText := parts[0].(map[string]any)["text"].(string)
		return 200, []byte(fmt.Sprintf(`{
			"candidates": [{
				"index": 0,
				"content": {"role": "model", "parts": [{"text": "echo: %s"}]},
				"finishReason": "STOP"
			}],
			"usageMetadata": {
				"promptTokenCount": 5,
				"candidatesTokenCount": 3,
				"thoughtsTokenCount": 0,
				"totalTokenCount": 8
			}
		}`, userText))
	}))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	res, err := lm.DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(res.Content))
	}
	if tc, ok := res.Content[0].(TextContent); !ok || tc.Text != "echo: hi" {
		t.Errorf("content[0] = %+v", res.Content[0])
	}
	if res.FinishReason.Unified != "stop" {
		t.Errorf("finish = %q, want stop", res.FinishReason.Unified)
	}
	if res.Usage.InputTokens.Total == nil || *res.Usage.InputTokens.Total != 5 {
		t.Errorf("input tokens: %+v", res.Usage.InputTokens)
	}
}

func TestDoGenerate_StopWithToolCalls_ReportsToolCalls(t *testing.T) {
	p := newTestProvider(t, captureRequest(t, func(body []byte) (int, []byte) {
		return 200, []byte(`{
			"candidates": [{
				"index": 0,
				"content": {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "sf"}}}]},
				"finishReason": "STOP"
			}]
		}`)
	}))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	res, err := lm.DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "weather?"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason.Unified != "tool-calls" {
		t.Errorf("finish = %q, want tool-calls (STOP+hasToolCalls)", res.FinishReason.Unified)
	}
}

func TestDoGenerate_FinishReasonMapping(t *testing.T) {
	cases := []struct {
		raw  string
		want string
	}{
		{"STOP", "stop"},
		{"MAX_TOKENS", "length"},
		{"IMAGE_SAFETY", "content-filter"},
		{"RECITATION", "content-filter"},
		{"SAFETY", "content-filter"},
		{"BLOCKLIST", "content-filter"},
		{"PROHIBITED_CONTENT", "content-filter"},
		{"SPII", "content-filter"},
		{"MALFORMED_FUNCTION_CALL", "error"},
		{"FINISH_REASON_UNSPECIFIED", "other"},
		{"OTHER", "other"},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			fr := mapGoogleFinishReason(tc.raw, false)
			if fr.Unified != tc.want {
				t.Errorf("Unified = %q, want %q", fr.Unified, tc.want)
			}
			if fr.Raw != tc.raw {
				t.Errorf("Raw = %q, want %q", fr.Raw, tc.raw)
			}
		})
	}
}

func TestDoGenerate_UsageWithThoughts(t *testing.T) {
	p := newTestProvider(t, captureRequest(t, func(body []byte) (int, []byte) {
		return 200, []byte(`{
			"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "x"}]}}],
			"usageMetadata": {
				"promptTokenCount": 10,
				"cachedContentTokenCount": 4,
				"candidatesTokenCount": 5,
				"thoughtsTokenCount": 7,
				"totalTokenCount": 26
			}
		}`)
	}))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	res, err := lm.DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.InputTokens.CacheRead == nil || *res.Usage.InputTokens.CacheRead != 4 {
		t.Errorf("cacheRead = %+v", res.Usage.InputTokens.CacheRead)
	}
	if res.Usage.InputTokens.NoCache == nil || *res.Usage.InputTokens.NoCache != 6 {
		t.Errorf("noCache = %+v, want 6", res.Usage.InputTokens.NoCache)
	}
	if res.Usage.OutputTokens.Reasoning == nil || *res.Usage.OutputTokens.Reasoning != 7 {
		t.Errorf("reasoning = %+v", res.Usage.OutputTokens.Reasoning)
	}
	if res.Usage.OutputTokens.Total == nil || *res.Usage.OutputTokens.Total != 12 {
		t.Errorf("output total = %+v, want 12", res.Usage.OutputTokens.Total)
	}
}

// ---- buildChatRequest golden tests ----

func TestBuildChatRequest_Passthrough(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	args := getArgsResult{
		Options: GoogleOptions{},
		Passthrough: map[string]any{
			"customField": "customValue",
		},
	}
	opts := GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, err := lm.buildChatRequest(args, opts, contents, system)
	if err != nil {
		t.Fatal(err)
	}
	if body["customField"] != "customValue" {
		t.Errorf("passthrough missing: %+v", body)
	}
}

func TestBuildChatRequest_ResponseFormatJSON_Schema(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"x": map[string]any{"type": "string"}},
	}
	opts := GenerateOptions{
		ResponseFormat: &ResponseFormat{Type: "json", Schema: schema, Name: "x"},
		Messages:       []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	}
	args := getArgsResult{Options: GoogleOptions{}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, err := lm.buildChatRequest(args, opts, contents, system)
	if err != nil {
		t.Fatal(err)
	}
	gc, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatal("generationConfig missing")
	}
	if gc["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v, want application/json", gc["responseMimeType"])
	}
	rs, ok := gc["responseSchema"].(map[string]any)
	if !ok {
		t.Fatalf("responseSchema: %T", gc["responseSchema"])
	}
	if rs["type"] != "object" {
		t.Errorf("responseSchema.type = %v", rs["type"])
	}
	if rs["title"] != "x" {
		t.Errorf("responseSchema.title = %v, want x", rs["title"])
	}
}

func TestBuildChatRequest_ThinkingBudget_Gemini25(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	budget := 2048
	args := getArgsResult{Options: GoogleOptions{
		ThinkingConfig: &ThinkingConfig{ThinkingBudget: &budget},
	}}
	opts := GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	gc := body["generationConfig"].(map[string]any)
	tc := gc["thinkingConfig"].(map[string]any)
	if tc["thinkingBudget"].(int) != 2048 {
		t.Errorf("thinkingBudget = %v, want 2048", tc["thinkingBudget"])
	}
}

func TestBuildChatRequest_ThinkingLevel_Gemini3(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini3ProPreview}
	args := getArgsResult{Options: GoogleOptions{
		ThinkingConfig: &ThinkingConfig{ThinkingLevel: "high"},
	}}
	opts := GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	gc := body["generationConfig"].(map[string]any)
	tc := gc["thinkingConfig"].(map[string]any)
	if tc["thinkingLevel"] != "high" {
		t.Errorf("thinkingLevel = %v, want high", tc["thinkingLevel"])
	}
}

func TestBuildChatRequest_Reasoning_NoneOnGemini3(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini3ProPreview}
	opts := GenerateOptions{
		Reasoning: "none",
		Messages:  []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
	}
	args, _ := lm.getArgs(opts)
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	gc := body["generationConfig"].(map[string]any)
	tc := gc["thinkingConfig"].(map[string]any)
	if tc["thinkingLevel"] != "minimal" {
		t.Errorf("thinkingLevel = %v, want minimal (none→minimal)", tc["thinkingLevel"])
	}
}

func TestBuildChatRequest_SafetySettings_ThresholdDefault(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	opts := GenerateOptions{
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"threshold": "BLOCK_NONE",
				"safetySettings": []any{
					map[string]any{"category": "HARM_CATEGORY_HATE_SPEECH"},
					map[string]any{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
				},
			},
		},
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
	}
	args, _ := lm.getArgs(opts)
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	ss, ok := body["safetySettings"].([]map[string]any)
	if !ok {
		t.Fatalf("safetySettings: %+v", body["safetySettings"])
	}
	if len(ss) != 2 {
		t.Fatalf("safetySettings = %d, want 2", len(ss))
	}
	if ss[0]["threshold"] != "BLOCK_NONE" {
		t.Errorf("ss[0].threshold = %v, want BLOCK_NONE (default applied)", ss[0]["threshold"])
	}
	if ss[1]["threshold"] != "BLOCK_LOW_AND_ABOVE" {
		t.Errorf("ss[1].threshold = %v, want BLOCK_LOW_AND_ABOVE (preserved)", ss[1]["threshold"])
	}
}

func TestBuildChatRequest_ImageConfig(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	args := getArgsResult{Options: GoogleOptions{
		ImageConfig: &ImageConfig{AspectRatio: "16:9", ImageSize: "1K"},
	}}
	opts := GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	gc := body["generationConfig"].(map[string]any)
	ic, ok := gc["imageConfig"].(map[string]any)
	if !ok {
		t.Fatal("imageConfig missing")
	}
	if ic["aspectRatio"] != "16:9" {
		t.Errorf("aspectRatio = %v", ic["aspectRatio"])
	}
	if ic["imageSize"] != "1K" {
		t.Errorf("imageSize = %v", ic["imageSize"])
	}
}

func TestBuildChatRequest_CachedContent(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	args := getArgsResult{Options: GoogleOptions{CachedContent: "cachedContents/abc"}}
	opts := GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	if body["cachedContent"] != "cachedContents/abc" {
		t.Errorf("cachedContent = %v", body["cachedContent"])
	}
}

func TestBuildChatRequest_ServiceTier_OmittedOnVertex(t *testing.T) {
	p := newTestProvider(t, nil)
	p.isVertex = true
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	args := getArgsResult{Options: GoogleOptions{ServiceTier: "flex"}, IsVertex: true}
	opts := GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	if _, ok := body["serviceTier"]; ok {
		t.Errorf("serviceTier should be omitted on Vertex, got %v", body["serviceTier"])
	}
}

func TestBuildChatRequest_ServiceTier_PresentOnGemini(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	args := getArgsResult{Options: GoogleOptions{ServiceTier: "flex"}, IsVertex: false}
	opts := GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	if body["serviceTier"] != "flex" {
		t.Errorf("serviceTier = %v, want flex", body["serviceTier"])
	}
}

func TestBuildChatRequest_VertexHeaders(t *testing.T) {
	p := newTestProvider(t, nil)
	p.isVertex = true
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	args := getArgsResult{
		Options:  GoogleOptions{SharedRequestType: "dedicated", RequestType: "shared"},
		IsVertex: true,
	}
	opts := GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}}}
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	_, headers, _, _ := lm.buildChatRequest(args, opts, contents, system)
	if headers.Get("X-Vertex-AI-LLM-Shared-Request-Type") != "dedicated" {
		t.Errorf("shared header = %q", headers.Get("X-Vertex-AI-LLM-Shared-Request-Type"))
	}
	if headers.Get("X-Vertex-AI-LLM-Request-Type") != "shared" {
		t.Errorf("request-type header = %q", headers.Get("X-Vertex-AI-LLM-Request-Type"))
	}
}

func TestBuildChatRequest_VertexHeaders_NonVertex_Warning(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	opts := GenerateOptions{
		ProviderOptions: ProviderOptions{
			"google": map[string]any{
				"sharedRequestType": "dedicated",
			},
		},
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
	}
	args, _ := lm.getArgs(opts)
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	_, headers, _, _ := lm.buildChatRequest(args, opts, contents, system)
	if headers.Get("X-Vertex-AI-LLM-Shared-Request-Type") != "" {
		t.Errorf("non-vertex: header should be empty, got %q", headers.Get("X-Vertex-AI-LLM-Shared-Request-Type"))
	}
	found := false
	for _, w := range args.Warnings {
		if w.Feature == "sharedRequestType/requestType" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected sharedRequestType warning, got %+v", args.Warnings)
	}
}

func TestBuildChatRequest_ToolChoice(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	tool := Tool{Type: "function", Name: "f"}
	opts := GenerateOptions{
		Tools:      []Tool{tool},
		ToolChoice: &ToolChoice{Type: "required"},
		Messages:   []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
	}
	args, _ := lm.getArgs(opts)
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	tc, ok := body["toolConfig"].(map[string]any)
	if !ok {
		t.Fatal("toolConfig missing")
	}
	fcc := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("mode = %v, want ANY", fcc["mode"])
	}
}

func TestBuildChatRequest_ToolChoice_ToolName(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	tool := Tool{Type: "function", Name: "f"}
	opts := GenerateOptions{
		Tools:      []Tool{tool},
		ToolChoice: &ToolChoice{Type: "tool", ToolName: "f"},
		Messages:   []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
	}
	args, _ := lm.getArgs(opts)
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	tc := body["toolConfig"].(map[string]any)
	fcc := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("mode = %v", fcc["mode"])
	}
	allowed := fcc["allowedFunctionNames"].([]string)
	if len(allowed) != 1 || allowed[0] != "f" {
		t.Errorf("allowedFunctionNames = %v", allowed)
	}
}

func TestBuildChatRequest_MixedTools_Gemini3_IncludeServerSide(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini3ProPreview}
	opts := GenerateOptions{
		Tools: []Tool{
			{Type: "provider", ID: "google.google_search", Name: "google_search"},
			{Type: "function", Name: "f"},
		},
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
	}
	args, _ := lm.getArgs(opts)
	contents, system, _, _ := ConvertPrompt(lm.modelID, opts)
	body, _, _, _ := lm.buildChatRequest(args, opts, contents, system)
	tc := body["toolConfig"].(map[string]any)
	if tc["includeServerSideToolInvocations"] != true {
		t.Errorf("includeServerSideToolInvocations = %v, want true", tc["includeServerSideToolInvocations"])
	}
}

// ---- parseGenerateResponse tests ----

func TestParseGenerateResponse_UnknownPart(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [
				{"text": "ok"},
				{"totallyNewField": {"foo": "bar"}}
			]},
			"finishReason": "STOP"
		}]
	}`)
	res, warnings, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Content) != 1 {
		t.Errorf("content = %d, want 1 (only the text)", len(res.Content))
	}
	pm, ok := res.ProviderMetadata["google"].(map[string]any)
	if !ok {
		t.Fatal("no google metadata")
	}
	unknowns, ok := pm["unknownParts"].([]map[string]any)
	if !ok || len(unknowns) != 1 {
		t.Errorf("unknownParts = %+v", pm["unknownParts"])
	}
	if !hasUnknownPartsWarning(warnings) {
		t.Errorf("expected unknown-part warning, got %+v", warnings)
	}
}

func TestParseGenerateResponse_GroundingSources(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [{"text": "x"}]},
			"finishReason": "STOP",
			"groundingMetadata": {
				"webSearchQueries": ["q1"],
				"groundingChunks": [
					{"web": {"uri": "https://example.com/a", "title": "A"}},
					{"web": {"uri": "https://example.com/a", "title": "A"}},
					{"image": {"sourceUri": "https://img.example.com/b", "title": "B"}},
					{"retrievedContext": {"uri": "https://r.example.com/c", "title": "C"}},
					{"retrievedContext": {"uri": "gs://bucket/file.pdf", "title": "doc"}},
					{"maps": {"uri": "https://maps.example.com/d", "title": "D"}}
				]
			}
		}]
	}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	pm := res.ProviderMetadata["google"].(map[string]any)
	sources, ok := pm["sources"].([]map[string]any)
	if !ok {
		t.Fatal("sources missing")
	}
	if len(sources) != 5 { // dedup'd by URL
		t.Errorf("sources = %d, want 5 (deduped)", len(sources))
	}
	// gs:// URL should be a "document" type.
	var foundDoc bool
	for _, s := range sources {
		if s["type"] == "document" {
			foundDoc = true
			if s["mediaType"] != "application/pdf" {
				t.Errorf("doc mediaType = %v", s["mediaType"])
			}
		}
	}
	if !foundDoc {
		t.Errorf("expected a document source for gs:// URI, got %+v", sources)
	}
}

func TestParseGenerateResponse_PromptFeedback(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [{"text": "ok"}]},
			"finishReason": "STOP"
		}],
		"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH"}]}
	}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	pm := res.ProviderMetadata["google"].(map[string]any)
	pf, ok := pm["promptFeedback"].(map[string]any)
	if !ok {
		t.Fatal("promptFeedback missing")
	}
	if pf["blockReason"] != "SAFETY" {
		t.Errorf("blockReason = %v", pf["blockReason"])
	}
}

func TestParseGenerateResponse_ServerToolCall(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini3ProPreview}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [
				{"toolCall": {"toolType": "google_search", "id": "srv-1", "args": {"q": "weather"}}}
			]},
			"finishReason": "STOP"
		}]
	}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Content) != 1 {
		t.Fatalf("content = %d, want 1", len(res.Content))
	}
	tc, ok := res.Content[0].(ToolCallContent)
	if !ok {
		t.Fatalf("content[0] = %T, want ToolCallContent", res.Content[0])
	}
	if tc.ToolName != "google_search" {
		t.Errorf("ToolName = %q, want google_search", tc.ToolName)
	}
	pm := tc.ProviderMetadata["google"].(map[string]any)
	if pm["serverToolType"] != "google_search" {
		t.Errorf("serverToolType = %v", pm["serverToolType"])
	}
}

func TestParseGenerateResponse_CodeExecutionPair(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [
				{"executableCode": {"language": "PYTHON", "code": "print(1)"}},
				{"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1"}}
			]},
			"finishReason": "STOP"
		}]
	}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Content) != 2 {
		t.Fatalf("content = %d, want 2", len(res.Content))
	}
	if ec, ok := res.Content[0].(ExecutableCodeContent); !ok || ec.Code != "print(1)" {
		t.Errorf("content[0] = %+v", res.Content[0])
	}
	if cr, ok := res.Content[1].(CodeExecutionResultContent); !ok || cr.Output != "1" {
		t.Errorf("content[1] = %+v", res.Content[1])
	}
}

func TestParseGenerateResponse_InlineDataAndFileData(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [
				{"inlineData": {"mimeType": "image/png", "data": "PNG-DATA"}},
				{"fileData": {"mimeType": "image/jpeg", "fileUri": "https://x/y.jpg"}}
			]},
			"finishReason": "STOP"
		}]
	}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Content) != 2 {
		t.Fatalf("content = %d, want 2", len(res.Content))
	}
	if img, ok := res.Content[0].(ImageContent); !ok || img.Source.Data != "PNG-DATA" {
		t.Errorf("content[0] = %+v", res.Content[0])
	}
	if img, ok := res.Content[1].(ImageContent); !ok || img.Source.URL != "https://x/y.jpg" {
		t.Errorf("content[1] = %+v", res.Content[1])
	}
}

func TestParseGenerateResponse_FunctionCallWithArgs(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [
				{"functionCall": {"id": "c-1", "name": "f", "args": {"x": 1}}}
			]},
			"finishReason": "STOP"
		}]
	}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	tc, ok := res.Content[0].(ToolCallContent)
	if !ok {
		t.Fatalf("content[0] = %T", res.Content[0])
	}
	if tc.ToolCallID != "c-1" || tc.ToolName != "f" {
		t.Errorf("tc = %+v", tc)
	}
}

func TestParseGenerateResponse_EmptyCandidates(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{"candidates": []}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Content) != 0 {
		t.Errorf("content = %d, want 0", len(res.Content))
	}
}

// ---- Reasoning to wire / wire to Reasoning roundtrip ----

func TestConvertPrompt_Reasoning_ThoughtTrue(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			AssistantMessage{Content: []AssistantContent{
				ReasoningContent{Text: "I think therefore I am", Signature: "s1"},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatal(err)
	}
	parts := contents[0].Parts
	if len(parts) != 1 {
		t.Fatalf("parts = %d", len(parts))
	}
	if parts[0].Thought == nil || !*parts[0].Thought {
		t.Error("thought flag not set")
	}
}

func TestParseGenerateResponse_ReasoningFromWire(t *testing.T) {
	p := newTestProvider(t, nil)
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	body := []byte(`{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [
				{"text": "thinking deeply", "thought": true, "thoughtSignature": "sig-1"},
				{"text": "the answer"}
			]},
			"finishReason": "STOP"
		}]
	}`)
	res, _, err := lm.parseGenerateResponse(body, nil, http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Content) != 2 {
		t.Fatalf("content = %d, want 2", len(res.Content))
	}
	rc, ok := res.Content[0].(ReasoningContent)
	if !ok {
		t.Fatalf("content[0] = %T", res.Content[0])
	}
	if rc.Text != "thinking deeply" || rc.Signature != "sig-1" {
		t.Errorf("rc = %+v", rc)
	}
	if _, ok := res.Content[1].(TextContent); !ok {
		t.Errorf("content[1] = %T, want TextContent", res.Content[1])
	}
}

// ---- getArgs provider options parsing ----

func TestGetArgs_VertexNamespace(t *testing.T) {
	p := newTestProvider(t, nil)
	p.isVertex = true
	p.useVertexAIHeaders = true
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	opts := GenerateOptions{
		ProviderOptions: ProviderOptions{
			"googleVertex": map[string]any{"responseModalities": []any{"TEXT"}},
		},
	}
	args, _ := lm.getArgs(opts)
	if !args.IsVertexLike {
		t.Error("expected isVertexLike=true on Vertex provider with googleVertex options")
	}
}

// ---- helper tests ----

func TestPartToContent_Text(t *testing.T) {
	tr := true
	c, unk, w := partToContent(internal.APIPart{Text: "hello", Thought: &tr})
	if _, ok := c.(ReasoningContent); !ok {
		t.Errorf("thought=true: got %T, want ReasoningContent", c)
	}
	if unk != nil || w != nil {
		t.Errorf("unk=%v w=%v", unk, w)
	}
}

func TestHasClientSideToolCalls(t *testing.T) {
	if hasClientSideToolCalls([]Content{TextContent{Text: "x"}}) {
		t.Error("text content: should be false")
	}
	if !hasClientSideToolCalls([]Content{ToolCallContent{ToolName: "f"}}) {
		t.Error("plain tool call: should be true")
	}
	server := ToolCallContent{
		ToolName:         "f",
		ProviderMetadata: ProviderMetadata{"google": map[string]any{"serverToolType": "google_search"}},
	}
	if hasClientSideToolCalls([]Content{server}) {
		t.Error("server tool call: should be false")
	}
}

func TestExtensionToMediaType(t *testing.T) {
	cases := []struct {
		uri          string
		wantMedia    string
		wantFilename string
	}{
		{"gs://bucket/file.pdf", "application/pdf", "file.pdf"},
		{"gs://b/x.txt", "text/plain", "x.txt"},
		{"gs://b/y.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "y.docx"},
		{"gs://b/z.md", "text/markdown", "z.md"},
		{"gs://b/unk", "", "unk"},
	}
	for _, tc := range cases {
		mt, fn := extensionToMediaType(tc.uri)
		if mt != tc.wantMedia || fn != tc.wantFilename {
			t.Errorf("%s: got (%q,%q), want (%q,%q)", tc.uri, mt, fn, tc.wantMedia, tc.wantFilename)
		}
	}
}

func TestMapReasoningForModel(t *testing.T) {
	// Gemini 3 uses thinkingLevel.
	g3 := mapReasoningForModel(ModelGemini3ProPreview, "high", nil, nil)
	if g3.ThinkingLevel != "high" {
		t.Errorf("g3 high: %+v", g3)
	}
	g3x := mapReasoningForModel(ModelGemini3ProPreview, "xhigh", nil, nil)
	if g3x.ThinkingLevel != "high" {
		t.Errorf("g3 xhigh: %+v, want high", g3x)
	}
	g3none := mapReasoningForModel(ModelGemini3ProPreview, "none", nil, nil)
	if g3none.ThinkingLevel != "minimal" {
		t.Errorf("g3 none: %+v, want minimal", g3none)
	}
}

// ---- public request body verification ----

func TestDoGenerate_EndToEnd_BodyShape(t *testing.T) {
	var captured map[string]any
	p := newTestProvider(t, captureRequest(t, func(body []byte) (int, []byte) {
		captured = decodeBody(t, body)
		return 200, []byte(`{
			"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "x"}]}, "finishReason": "STOP"}]
		}`)
	}))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini35Flash}
	_, err := lm.DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	contents, ok := captured["contents"].([]any)
	if !ok || len(contents) == 0 {
		t.Fatalf("contents = %+v", captured["contents"])
	}
	role := contents[0].(map[string]any)["role"]
	if role != "user" {
		t.Errorf("role = %v, want user", role)
	}
}

func TestProvider_Defaults_PassthroughShape(t *testing.T) {
	// Sanity: ProviderSettings includes the fields used by the chat path.
	s := ProviderSettings{}
	if s.BaseURL != "" {
		t.Errorf("BaseURL = %q, want empty default", s.BaseURL)
	}
	if s.Retry != nil {
		t.Errorf("Retry should default to nil")
	}
}

func TestIsVertexProvider_ByBaseURL(t *testing.T) {
	cases := []struct {
		base string
		want bool
	}{
		{"https://generativelanguage.googleapis.com/v1beta", false},
		{"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/x", true},
		{"https://custom.example.com", false},
	}
	for _, tc := range cases {
		t.Run(tc.base, func(t *testing.T) {
			if got := isVertexProvider(tc.base, false); got != tc.want {
				t.Errorf("isVertexProvider(%q) = %v, want %v", tc.base, got, tc.want)
			}
		})
	}
}
