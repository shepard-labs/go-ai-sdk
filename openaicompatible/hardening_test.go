package openaicompatible

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// ---------------------------------------------------------------------------
// Phase 6: Public export sweep (compile-time checks)
// ---------------------------------------------------------------------------

// TestPublicExportSweep verifies all required public names exist and that
// model ID types are plain strings. Failures are compile errors, not runtime
// panics, because the assignments below would not compile without the exports.
func TestPublicExportSweep(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "sweep"})

	// Required factory and interface names.
	var _ Provider = p
	var _ LanguageModel = p.Chat("m")
	var _ LanguageModel = p.ChatModel("m")
	var _ LanguageModel = p.Completion("m")
	var _ LanguageModel = p.CompletionModel("m")
	var _ EmbeddingModel = p.Embedding("m")
	var _ EmbeddingModel = p.EmbeddingModel("m")
	var _ EmbeddingModel = p.TextEmbeddingModel("m")
	var _ ImageModel = p.Image("m")
	var _ ImageModel = p.ImageModel("m")

	// Concrete type aliases.
	var _ *OpenAICompatibleChatLanguageModel = p.Chat("m").(*openAICompatibleChatLanguageModel)
	var _ *OpenAICompatibleCompletionLanguageModel = p.Completion("m").(*openAICompatibleCompletionLanguageModel)
	var _ *OpenAICompatibleEmbeddingModel = p.Embedding("m").(*openAICompatibleEmbeddingModel)
	var _ *OpenAICompatibleImageModel = p.Image("m").(*openAICompatibleImageModel)

	// Type aliases must exist.
	var _ ChatLanguageModel = p.Chat("m")
	var _ CompletionLanguageModel = p.Completion("m")

	// Option types.
	_ = ChatOptions{}
	_ = CompletionOptions{}
	_ = EmbeddingOptions{}

	// Error types.
	_ = ErrMissingBaseURL
	_ = ErrMissingName
	_ = APIError{}
	_ = &APICallError{}
	_ = UnsupportedFunctionalityError{}
	_ = InvalidPromptError{}
	_ = InvalidResponseDataError{}
	_ = TooManyEmbeddingValuesForCallError{}

	// Provider error structure.
	_ = ProviderErrorStructure{}

	// Metadata extractor interfaces.
	var _ MetadataExtractor = nil
	var _ StreamMetadataExtractor = nil

	// Version constant.
	if Version == "" {
		t.Fatal("Version is empty")
	}

	// Model ID types are free-form strings (plain string arguments).
	_ = p.Chat("any-string-model-id")
	_ = p.Completion("any-string-model-id")
	_ = p.Embedding("any-string-model-id")
	_ = p.Image("any-string-model-id")
}

// ---------------------------------------------------------------------------
// Phase 6: Named tests per phase checklist
// ---------------------------------------------------------------------------

// TestProviderRequiresBaseURLAndName verifies that missing BaseURL and Name
// surface as provider errors through Err() and that model calls fail before
// HTTP execution.
func TestProviderRequiresBaseURLAndName(t *testing.T) {
	t.Run("missing base URL fails Err and DoGenerate", func(t *testing.T) {
		p := CreateOpenAICompatible(ProviderSettings{Name: "acme"})
		if !errors.Is(p.Err(), ErrMissingBaseURL) {
			t.Fatalf("Err() = %v, want ErrMissingBaseURL", p.Err())
		}
		f := &recordingFetcher{}
		noFetchP := CreateOpenAICompatible(ProviderSettings{Name: "acme", Fetch: f})
		_, err := noFetchP.Chat("m").DoGenerate(context.Background(), GenerateOptions{})
		if !errors.Is(err, ErrMissingBaseURL) {
			t.Fatalf("DoGenerate err = %v, want ErrMissingBaseURL", err)
		}
		if f.calls != 0 {
			t.Fatalf("fetcher was called despite provider error")
		}
	})

	t.Run("missing name fails Err and DoEmbed", func(t *testing.T) {
		p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test"})
		if !errors.Is(p.Err(), ErrMissingName) {
			t.Fatalf("Err() = %v, want ErrMissingName", p.Err())
		}
		f := &recordingFetcher{}
		noFetchP := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Fetch: f})
		_, err := noFetchP.Embedding("m").DoEmbed(context.Background(), EmbedOptions{})
		if !errors.Is(err, ErrMissingName) {
			t.Fatalf("DoEmbed err = %v, want ErrMissingName", err)
		}
		if f.calls != 0 {
			t.Fatalf("fetcher was called despite provider error")
		}
	})

	t.Run("both missing uses errors.Join", func(t *testing.T) {
		p := CreateOpenAICompatible(ProviderSettings{})
		if !errors.Is(p.Err(), ErrMissingBaseURL) || !errors.Is(p.Err(), ErrMissingName) {
			t.Fatalf("Err() = %v, want both errors joined", p.Err())
		}
	})
}

// TestChatRequestMergesProviderOptions verifies that provider-options merge
// order is respected: openai-compatible < openaiCompatible < Name < camelCase(Name).
func TestChatRequestMergesProviderOptions(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "my-provider",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{
			"openai-compatible": {"user": "compat"},
			"openaiCompatible":  {"user": "camelCompat"},
			"my-provider":       {"user": "raw-name"},
			"myProvider":        {"user": "camel-name"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	// camel-name should win (last in merge order).
	if body["user"] != "camel-name" {
		t.Fatalf("user = %v, want camel-name", body["user"])
	}
	// deprecated key warning should be present.
	found := false
	for _, w := range result.Warnings {
		if w.Message == deprecatedProviderOptionsWarningMessage {
			found = true
		}
	}
	if !found {
		t.Fatalf("deprecated warning missing: %#v", result.Warnings)
	}
}

// TestChatStreamAccumulatesToolCallsByIndex verifies that streaming tool-call
// deltas are accumulated per index and emitted as StreamToolCall events.
func TestChatStreamAccumulatesToolCallsByIndex(t *testing.T) {
	// Build two tool calls delivered across multiple chunks each.
	chunks := sse().
		add(`{"id":"r1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc-0","function":{"name":"fn0","arguments":"{"}}]}}]}`).
		add(`{"id":"r1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\":1}"}}]}}]}`).
		add(`{"id":"r1","choices":[{"delta":{"tool_calls":[{"index":1,"id":"tc-1","function":{"name":"fn1","arguments":"{\"y\":2}"}}]}}]}`).
		done().build()

	f := &recordingFetcher{responses: []*http.Response{
		{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(chunks))},
	}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.Chat("m").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	var toolCalls []StreamToolCall
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			toolCalls = append(toolCalls, tc)
		}
	}
	if len(toolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2: %#v", len(toolCalls), parts)
	}
	if toolCalls[0].ToolCallID != "tc-0" || toolCalls[0].ToolName != "fn0" {
		t.Fatalf("tool[0] = %#v", toolCalls[0])
	}
	if toolCalls[1].ToolCallID != "tc-1" || toolCalls[1].ToolName != "fn1" {
		t.Fatalf("tool[1] = %#v", toolCalls[1])
	}
}

// TestCompletionPromptRejectsLaterSystemMessage verifies the improved Go error
// message for unexpected system messages in completion prompts.
func TestCompletionPromptRejectsLaterSystemMessage(t *testing.T) {
	msgs := []Message{
		UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
		SystemMessage{Content: "unexpected"},
	}
	_, _, err := flattenCompletionPrompt(msgs)
	if err == nil {
		t.Fatal("expected error")
	}
	inv, ok := err.(InvalidPromptError)
	if !ok {
		t.Fatalf("error type = %T", err)
	}
	want := "unexpected system message in completion prompt: unexpected"
	if inv.Message != want {
		t.Fatalf("message = %q, want %q", inv.Message, want)
	}
}

// TestEmbeddingSupportsCamelCaseProviderOptions verifies that the Go port's
// embedding model correctly reads camelCase provider-options keys (a Go-specific
// improvement over the upstream TypeScript which omits toCamelCase for embeddings).
func TestEmbeddingSupportsCamelCaseProviderOptions(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		response(200, `{"data":[{"embedding":[0.1,0.2]}]}`),
	}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "my-provider",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	dimensions := 512
	_, err := p.Embedding("text-embed").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"hello"},
		ProviderOptions: ProviderOptions{
			"myProvider": {"dimensions": &dimensions},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := f.requests[0]
	var body map[string]any
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		t.Fatalf("parse request: %v", err)
	}
	if body["dimensions"] == nil {
		t.Fatalf("dimensions missing from request body: %#v", body)
	}
}

// TestImageEditUsesRepeatedImageFields verifies that multiple image files in
// an edit request are encoded as repeated `image` multipart fields (not image[]).
func TestImageEditUsesRepeatedImageFields(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		response(200, `{"data":[{"b64_json":"abc"},{"b64_json":"def"}]}`),
	}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.Image("dall-e-2").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "test",
		N:      2,
		Files: []ImageFile{
			{Type: "bytes", Data: []byte("img1"), MediaType: "image/png"},
			{Type: "bytes", Data: []byte("img2"), MediaType: "image/png"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := f.requests[0]
	ct := req.Header.Get("Content-Type")
	if len(ct) < 9 || ct[:9] != "multipart" {
		t.Fatalf("Content-Type = %q, want multipart/form-data", ct)
	}
	// Body should contain the field name "image" twice but not "image[]".
	rawBody, _ := io.ReadAll(req.Body)
	bodyStr := string(rawBody)
	if strings.Contains(bodyStr, `name="image[]"`) {
		t.Fatalf("body contains image[], must use repeated image fields: %q", bodyStr[:min(200, len(bodyStr))])
	}
}

// ---------------------------------------------------------------------------
// Phase 6: Concurrency and immutability sweep
// ---------------------------------------------------------------------------

// TestProviderConcurrentUse verifies that a provider is safe for concurrent
// use by multiple goroutines calling model factories and DoGenerate/DoEmbed.
func TestProviderConcurrentUse(t *testing.T) {
	const goroutines = 20
	responses := make([]*http.Response, goroutines)
	for i := range responses {
		responses[i] = response(200, `{"choices":[{"message":{"content":"ok"}}]}`)
	}
	f := &recordingFetcher{responses: responses}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	var wg sync.WaitGroup
	var errCount int64
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
				Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
			})
			if err != nil {
				atomic.AddInt64(&errCount, 1)
			}
		}()
	}
	wg.Wait()
	if errCount != 0 {
		t.Fatalf("%d goroutines got errors during concurrent DoGenerate", errCount)
	}
}

// TestModelConcurrentUse verifies that a single model instance is safe for
// concurrent use.
func TestModelConcurrentUse(t *testing.T) {
	const goroutines = 10
	responses := make([]*http.Response, goroutines)
	for i := range responses {
		responses[i] = response(200, `{"choices":[{"message":{"content":"ok"}}]}`)
	}
	f := &recordingFetcher{responses: responses}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	model := p.Chat("gpt-4o")
	var wg sync.WaitGroup
	var errCount int64
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := model.DoGenerate(context.Background(), GenerateOptions{
				Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
			})
			if err != nil {
				atomic.AddInt64(&errCount, 1)
			}
		}()
	}
	wg.Wait()
	if errCount != 0 {
		t.Fatalf("%d goroutines got errors during concurrent model use", errCount)
	}
}

// TestProviderHeadersClonedAtCreation verifies that mutating the ProviderSettings
// Headers map after CreateOpenAICompatible does not affect provider state.
func TestProviderHeadersClonedAtCreation(t *testing.T) {
	original := http.Header{"X-Custom": []string{"original"}}
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
		Headers: original,
	})
	// Mutate original after creation.
	original.Set("X-Custom", "mutated")

	_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Provider should have used the original "original" value, not "mutated".
	if got := f.requests[0].Header.Get("X-Custom"); got != "original" {
		t.Fatalf("X-Custom = %q, want original (headers must be cloned at creation)", got)
	}
}

// TestProviderQueryParamsClonedAtCreation verifies QueryParams cloning.
func TestProviderQueryParamsClonedAtCreation(t *testing.T) {
	original := map[string]string{"version": "1"}
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL:     "https://example.test",
		Name:        "acme",
		Fetch:       f,
		Retry:       &RetryOptions{MaxRetries: 0},
		QueryParams: original,
	})
	original["version"] = "2"
	_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := f.requests[0].URL.Query().Get("version")
	if got != "1" {
		t.Fatalf("version query param = %q, want 1 (QueryParams must be cloned)", got)
	}
}

// TestTextEmbeddingModelDeprecatedAlias verifies TextEmbeddingModel is a
// deprecated alias for EmbeddingModel and behaves identically.
func TestTextEmbeddingModelDeprecatedAlias(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme"})
	emb := p.EmbeddingModel("text-embed-3")
	dep := p.TextEmbeddingModel("text-embed-3")
	if emb.Provider() != dep.Provider() {
		t.Fatalf("Provider mismatch: %q vs %q", emb.Provider(), dep.Provider())
	}
	if emb.ModelID() != dep.ModelID() {
		t.Fatalf("ModelID mismatch: %q vs %q", emb.ModelID(), dep.ModelID())
	}
	if emb.MaxEmbeddingsPerCall() != dep.MaxEmbeddingsPerCall() {
		t.Fatalf("MaxEmbeddingsPerCall mismatch: %d vs %d", emb.MaxEmbeddingsPerCall(), dep.MaxEmbeddingsPerCall())
	}
}

// ---------------------------------------------------------------------------
// Phase 6: Wire encoding verification
// ---------------------------------------------------------------------------

// TestChatAbsentOptionalFieldsOmitted verifies absent optional JSON fields
// are omitted from the request body.
func TestChatAbsentOptionalFieldsOmitted(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		response(200, `{"choices":[{"message":{"content":"ok"}}]}`),
	}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	for _, field := range []string{"user", "max_tokens", "temperature", "top_p", "stop", "seed",
		"frequency_penalty", "presence_penalty", "response_format", "tools", "tool_choice",
		"reasoning_effort", "verbosity"} {
		if _, ok := body[field]; ok {
			t.Fatalf("optional field %q should be absent but is present: %#v", field, body)
		}
	}
}

// TestAssistantContentNullWhenToolCallsOnly verifies JSON null is emitted
// for assistant content when only tool calls are present and text is empty.
func TestAssistantContentNullWhenToolCallsOnly(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		response(200, `{"choices":[{"message":{"content":"ok"}}]}`),
	}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{ToolCallID: "tc1", ToolName: "fn", Input: []byte(`{}`)},
			}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{ToolCallID: "tc1", Output: ToolResultOutput{Type: "text", Value: "ok"}},
			}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var rawBody map[string]any
	if err := json.NewDecoder(f.requests[0].Body).Decode(&rawBody); err != nil {
		t.Fatalf("parse request body: %v", err)
	}
	messages := rawBody["messages"].([]any)
	// First message is the assistant message.
	assistant := messages[0].(map[string]any)
	if assistant["content"] != nil {
		t.Fatalf("assistant content = %v, want null when tool calls present and no text", assistant["content"])
	}
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
