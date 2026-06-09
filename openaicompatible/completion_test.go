package openaicompatible

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Prompt flattening tests
// ---------------------------------------------------------------------------

func TestCompletionFlattenInitialSystemMessage(t *testing.T) {
	msgs := []Message{
		SystemMessage{Content: "sys-content"},
		UserMessage{Content: []UserContent{TextContent{Text: "hello"}}},
	}
	prompt, stop, err := flattenCompletionPrompt(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(prompt, "sys-content\n\n") {
		t.Fatalf("system not prepended: %q", prompt)
	}
	if stop != "\nuser:" {
		t.Fatalf("generated stop = %q", stop)
	}
}

func TestCompletionFlattenLaterSystemMessageErrors(t *testing.T) {
	msgs := []Message{
		UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
		SystemMessage{Content: "late-sys"},
	}
	_, _, err := flattenCompletionPrompt(msgs)
	if err == nil {
		t.Fatal("expected error for late system message")
	}
	inv, ok := err.(InvalidPromptError)
	if !ok {
		t.Fatalf("error type = %T, want InvalidPromptError", err)
	}
	want := "unexpected system message in completion prompt: late-sys"
	if inv.Message != want {
		t.Fatalf("error message = %q, want %q", inv.Message, want)
	}
}

func TestCompletionFlattenUserFormatting(t *testing.T) {
	msgs := []Message{
		UserMessage{Content: []UserContent{TextContent{Text: "hello world"}}},
	}
	prompt, _, err := flattenCompletionPrompt(msgs)
	if err != nil {
		t.Fatal(err)
	}
	// Must contain user:\nhello world\n\n
	if !strings.Contains(prompt, "user:\nhello world\n\n") {
		t.Fatalf("user formatting wrong: %q", prompt)
	}
}

func TestCompletionFlattenAssistantFormatting(t *testing.T) {
	msgs := []Message{
		AssistantMessage{Content: []AssistantContent{TextContent{Text: "answer"}}},
	}
	prompt, _, err := flattenCompletionPrompt(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "assistant:\nanswer\n\n") {
		t.Fatalf("assistant formatting wrong: %q", prompt)
	}
}

func TestCompletionFlattenFinalAssistantPrefixAppended(t *testing.T) {
	msgs := []Message{
		UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
	}
	prompt, _, err := flattenCompletionPrompt(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(prompt, "assistant:\n") {
		t.Fatalf("assistant prefix not at end: %q", prompt)
	}
}

func TestCompletionFlattenNonTextUserPartsIgnored(t *testing.T) {
	msgs := []Message{
		UserMessage{Content: []UserContent{
			TextContent{Text: "text part"},
			FileContent{MediaType: "image/png", Data: []byte("img")},
		}},
	}
	prompt, _, err := flattenCompletionPrompt(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "text part") {
		t.Fatalf("text part missing: %q", prompt)
	}
	// Non-text part should be silently ignored (no error).
}

func TestCompletionFlattenAssistantToolCallErrors(t *testing.T) {
	msgs := []Message{
		AssistantMessage{Content: []AssistantContent{
			ToolCallContent{ToolCallID: "id", ToolName: "fn", Input: json.RawMessage(`{}`)},
		}},
	}
	_, _, err := flattenCompletionPrompt(msgs)
	if err == nil {
		t.Fatal("expected error for assistant tool call")
	}
	unsup, ok := err.(UnsupportedFunctionalityError)
	if !ok {
		t.Fatalf("error type = %T", err)
	}
	if unsup.Functionality != "tool-call messages" {
		t.Fatalf("functionality = %q", unsup.Functionality)
	}
}

func TestCompletionFlattenToolMessageErrors(t *testing.T) {
	msgs := []Message{
		ToolMessage{Content: []ToolContent{
			ToolResultContent{ToolCallID: "id", Output: ToolResultOutput{Type: "text", Value: "result"}},
		}},
	}
	_, _, err := flattenCompletionPrompt(msgs)
	if err == nil {
		t.Fatal("expected error for tool message")
	}
	unsup, ok := err.(UnsupportedFunctionalityError)
	if !ok {
		t.Fatalf("error type = %T", err)
	}
	if unsup.Functionality != "tool messages" {
		t.Fatalf("functionality = %q", unsup.Functionality)
	}
}

func TestCompletionFlattenGeneratedStopSequence(t *testing.T) {
	_, stop, err := flattenCompletionPrompt([]Message{
		UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if stop != "\nuser:" {
		t.Fatalf("generated stop = %q, want \\nuser:", stop)
	}
}

// ---------------------------------------------------------------------------
// Completion request tests
// ---------------------------------------------------------------------------

func completionProvider(t *testing.T, f *recordingFetcher) Provider {
	t.Helper()
	return CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
}

func completionResponse200(text string) *http.Response {
	body := `{"id":"cmp-1","created":10,"model":"gpt-3","choices":[{"text":"` + text + `","finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5}}`
	return response(200, body)
}

func TestCompletionRequestUsesCompletionsEndpoint(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = result
	if len(f.requests) == 0 {
		t.Fatal("no request recorded")
	}
	if f.requests[0].URL.Path != "/completions" {
		t.Fatalf("path = %q, want /completions", f.requests[0].URL.Path)
	}
}

func TestCompletionRequestIncludesConvertedPrompt(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	prompt, ok := body["prompt"].(string)
	if !ok || !strings.Contains(prompt, "user:\nhello\n\n") {
		t.Fatalf("prompt = %q", prompt)
	}
}

func TestCompletionRequestStopSequenceOrder(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages:      []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		StopSequences: []string{"caller-stop"},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	stops, ok := body["stop"].([]any)
	if !ok || len(stops) < 2 {
		t.Fatalf("stop = %#v", body["stop"])
	}
	if stops[0].(string) != "\nuser:" {
		t.Fatalf("first stop = %q, want \\nuser:", stops[0])
	}
	if stops[1].(string) != "caller-stop" {
		t.Fatalf("second stop = %q, want caller-stop", stops[1])
	}
}

func TestCompletionRequestStandardOptionMapping(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	maxTokens := 20
	temp := 0.7
	topP := 0.9
	freqPen := 0.3
	presPen := 0.1
	seed := 42
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages:         []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		MaxOutputTokens:  &maxTokens,
		Temperature:      &temp,
		TopP:             &topP,
		FrequencyPenalty: &freqPen,
		PresencePenalty:  &presPen,
		Seed:             &seed,
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["max_tokens"].(float64) != 20 {
		t.Fatalf("max_tokens = %v", body["max_tokens"])
	}
	if body["temperature"].(float64) != 0.7 {
		t.Fatalf("temperature = %v", body["temperature"])
	}
	if body["top_p"].(float64) != 0.9 {
		t.Fatalf("top_p = %v", body["top_p"])
	}
	if body["frequency_penalty"].(float64) != 0.3 {
		t.Fatalf("frequency_penalty = %v", body["frequency_penalty"])
	}
	if body["presence_penalty"].(float64) != 0.1 {
		t.Fatalf("presence_penalty = %v", body["presence_penalty"])
	}
	if body["seed"].(float64) != 42 {
		t.Fatalf("seed = %v", body["seed"])
	}
}

func TestCompletionRequestTypedOptionsMapping(t *testing.T) {
	echoTrue := true
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{
			"acme": map[string]any{
				"echo":      &echoTrue,
				"logitBias": map[string]float64{"50256": -100},
				"suffix":    "extra",
				"user":      "user-xyz",
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["echo"] != true {
		t.Fatalf("echo = %v", body["echo"])
	}
	if body["logit_bias"] == nil {
		t.Fatalf("logit_bias missing")
	}
	if body["suffix"] != "extra" {
		t.Fatalf("suffix = %v", body["suffix"])
	}
	if body["user"] != "user-xyz" {
		t.Fatalf("user = %v", body["user"])
	}
}

func TestCompletionRequestProviderRawAndCamelCaseMerge(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "my-provider",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
	})
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ProviderOptions: ProviderOptions{
			"my-provider": map[string]any{"raw_field": "from-raw"},
			"myProvider":  map[string]any{"camel_field": "from-camel"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["raw_field"] != "from-raw" {
		t.Fatalf("raw_field = %v", body["raw_field"])
	}
	if body["camel_field"] != "from-camel" {
		t.Fatalf("camel_field = %v", body["camel_field"])
	}
}

func TestCompletionRequestTopKEmitsWarning(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	topK := 5
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		TopK:     &topK,
	})
	if err != nil {
		t.Fatal(err)
	}
	hasTopKWarn := false
	for _, w := range result.Warnings {
		if w.Feature == "topK" {
			hasTopKWarn = true
		}
	}
	if !hasTopKWarn {
		t.Fatalf("no topK warning: %#v", result.Warnings)
	}
}

func TestCompletionRequestToolsEmitsWarning(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Tools:    []Tool{{Type: "function", Name: "weather"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	hasToolsWarn := false
	for _, w := range result.Warnings {
		if w.Feature == "tools" {
			hasToolsWarn = true
		}
	}
	if !hasToolsWarn {
		t.Fatalf("no tools warning: %#v", result.Warnings)
	}
}

func TestCompletionRequestToolChoiceEmitsWarning(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages:   []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ToolChoice: &ToolChoice{Type: "auto"},
	})
	if err != nil {
		t.Fatal(err)
	}
	hasWarn := false
	for _, w := range result.Warnings {
		if w.Feature == "toolChoice" {
			hasWarn = true
		}
	}
	if !hasWarn {
		t.Fatalf("no toolChoice warning: %#v", result.Warnings)
	}
}

func TestCompletionRequestJSONResponseFormatEmitsWarning(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages:       []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		ResponseFormat: &ResponseFormat{Type: "json"},
	})
	if err != nil {
		t.Fatal(err)
	}
	hasWarn := false
	for _, w := range result.Warnings {
		if w.Feature == "responseFormat" && w.Details == "JSON response format is not supported." {
			hasWarn = true
		}
	}
	if !hasWarn {
		t.Fatalf("no responseFormat warning: %#v", result.Warnings)
	}
}

func TestCompletionRequestTransformRequestBodyNotCalled(t *testing.T) {
	transformCalled := false
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 0},
		TransformRequestBody: func(body map[string]any) map[string]any {
			transformCalled = true
			return body
		},
	})
	_, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if transformCalled {
		t.Fatal("TransformRequestBody was called for completion request (should not be)")
	}
}

// ---------------------------------------------------------------------------
// Non-streaming response tests
// ---------------------------------------------------------------------------

func TestCompletionResponseParsesFirstChoiceOnly(t *testing.T) {
	body := `{"choices":[{"text":"first","finish_reason":"stop"},{"text":"second","finish_reason":"stop"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, body)}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(result.Content))
	}
	if result.Content[0].(TextContent).Text != "first" {
		t.Fatalf("content = %q", result.Content[0].(TextContent).Text)
	}
}

func TestCompletionResponseNonEmptyTextReturnsTextContent(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("hello response")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content = %#v", result.Content)
	}
	if result.Content[0].(TextContent).Text != "hello response" {
		t.Fatalf("text = %q", result.Content[0].(TextContent).Text)
	}
}

func TestCompletionResponseEmptyTextReturnsNoContentPart(t *testing.T) {
	body := `{"choices":[{"text":"","finish_reason":"stop"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, body)}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 0 {
		t.Fatalf("content len = %d, want 0 for empty text", len(result.Content))
	}
}

func TestCompletionResponseFinishReasonMaps(t *testing.T) {
	cases := map[string]string{
		"stop":           "stop",
		"length":         "length",
		"content_filter": "content-filter",
		"custom-raw":     "other",
	}
	for raw, want := range cases {
		t.Run(raw, func(t *testing.T) {
			body := `{"choices":[{"text":"x","finish_reason":"` + raw + `"}]}`
			f := &recordingFetcher{responses: []*http.Response{response(200, body)}}
			p := completionProvider(t, f)
			result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
				Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if result.FinishReason.Unified != want {
				t.Fatalf("finish reason unified = %q, want %q", result.FinishReason.Unified, want)
			}
			if result.FinishReason.Raw != raw {
				t.Fatalf("finish reason raw = %q, want %q", result.FinishReason.Raw, raw)
			}
		})
	}
}

func TestCompletionResponseUsageMapsCorrectly(t *testing.T) {
	body := `{"choices":[{"text":"x","finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, body)}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	u := result.Usage
	if u.InputTokens.Total == nil || *u.InputTokens.Total != 10 {
		t.Fatalf("input total = %v", u.InputTokens.Total)
	}
	if u.InputTokens.NoCache == nil || *u.InputTokens.NoCache != 10 {
		t.Fatalf("input no-cache = %v", u.InputTokens.NoCache)
	}
	if u.InputTokens.CacheRead != nil {
		t.Fatalf("cache read should be nil: %v", u.InputTokens.CacheRead)
	}
	if u.OutputTokens.Total == nil || *u.OutputTokens.Total != 5 {
		t.Fatalf("output total = %v", u.OutputTokens.Total)
	}
	if u.OutputTokens.Text == nil || *u.OutputTokens.Text != 5 {
		t.Fatalf("output text = %v", u.OutputTokens.Text)
	}
	if u.OutputTokens.Reasoning != nil {
		t.Fatalf("reasoning should be nil: %v", u.OutputTokens.Reasoning)
	}
}

func TestCompletionResponseMissingUsageFieldsInsidePresentUsageBecomeZeroPointers(t *testing.T) {
	body := `{"choices":[{"text":"x","finish_reason":"stop"}],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, body)}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	u := result.Usage
	// present usage with missing fields → zero pointers
	if u.InputTokens.Total == nil || *u.InputTokens.Total != 0 {
		t.Fatalf("input total with missing field = %v", u.InputTokens.Total)
	}
	if u.OutputTokens.Total == nil || *u.OutputTokens.Total != 0 {
		t.Fatalf("output total with missing field = %v", u.OutputTokens.Total)
	}
}

func TestCompletionResponseMissingUsageIsNil(t *testing.T) {
	body := `{"choices":[{"text":"x","finish_reason":"stop"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, body)}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.InputTokens.Total != nil || result.Usage.OutputTokens.Total != nil {
		t.Fatalf("usage should be empty when absent: %#v", result.Usage)
	}
}

func TestCompletionResponseIDModelTimestampHeadersAndRawBody(t *testing.T) {
	body := `{"id":"cmp-id","created":1710000000,"model":"gpt-3.5","choices":[{"text":"ok","finish_reason":"stop"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, body)}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Response.ID != "cmp-id" {
		t.Fatalf("response ID = %q", result.Response.ID)
	}
	if result.Response.ModelID != "gpt-3.5" {
		t.Fatalf("response model = %q", result.Response.ModelID)
	}
	if result.Response.Timestamp == nil || !result.Response.Timestamp.Equal(time.Unix(1710000000, 0)) {
		t.Fatalf("response timestamp = %v", result.Response.Timestamp)
	}
	if result.Response.Headers.Get("X-Request-Id") == "" {
		t.Fatalf("response headers missing: %#v", result.Response.Headers)
	}
	if len(result.Response.Body) == 0 {
		t.Fatal("response raw body is empty")
	}
}

func TestCompletionResponseRequestBodyMetadataRetained(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{completionResponse200("ok")}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Request.Body) == 0 {
		t.Fatal("request body metadata is empty")
	}
	var body map[string]any
	if err := json.Unmarshal(result.Request.Body, &body); err != nil {
		t.Fatalf("request body not valid JSON: %v", err)
	}
	if body["model"] != "gpt-3" {
		t.Fatalf("request body model = %v", body["model"])
	}
}

// ---------------------------------------------------------------------------
// Streaming request tests
// ---------------------------------------------------------------------------

func completionChunk(text string) string {
	return `{"id":"cmp-1","created":10,"model":"gpt-3","choices":[{"text":"` + text + `","index":0}]}`
}

func completionChunkWithFinish(text, finishReason string) string {
	return `{"id":"cmp-1","created":10,"model":"gpt-3","choices":[{"text":"` + text + `","finish_reason":"` + finishReason + `","index":0}]}`
}

func completionSseResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"X-Request-Id": []string{"resp-id"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestCompletionStreamRequestIncludesStreamTrue(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	body := decodeRequestBody(t, result.Request.Body)
	if body["stream"] != true {
		t.Fatalf("stream = %v", body["stream"])
	}
}

func TestCompletionStreamIncludeUsageAddsStreamOptions(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL:      "https://example.test",
		Name:         "acme",
		Fetch:        f,
		Retry:        &RetryOptions{MaxRetries: 0},
		IncludeUsage: true,
	})
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	body := decodeRequestBody(t, result.Request.Body)
	so, ok := body["stream_options"].(map[string]any)
	if !ok || so["include_usage"] != true {
		t.Fatalf("stream_options = %#v", body["stream_options"])
	}
}

func TestCompletionStreamNoIncludeUsageWhenFalse(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	body := decodeRequestBody(t, result.Request.Body)
	if _, ok := body["stream_options"]; ok {
		t.Fatalf("stream_options present when IncludeUsage=false: %#v", body)
	}
}

func TestCompletionStreamStartIncludesWarnings(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	topK := 5
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{
			Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
			TopK:     &topK,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	if len(parts) == 0 {
		t.Fatal("no parts")
	}
	start, ok := parts[0].(StreamStart)
	if !ok {
		t.Fatalf("first part = %T", parts[0])
	}
	hasTopK := false
	for _, w := range start.Warnings {
		if w.Feature == "topK" {
			hasTopK = true
		}
	}
	if !hasTopK {
		t.Fatalf("no topK warning in StreamStart: %#v", start.Warnings)
	}
}

func TestCompletionStreamRawChunksEmitBeforeParsedEvents(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions:  GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
		IncludeRawChunks: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	hasRaw := false
	for _, p := range parts {
		if _, ok := p.(StreamRaw); ok {
			hasRaw = true
			break
		}
	}
	if !hasRaw {
		t.Fatal("no StreamRaw emitted when IncludeRawChunks=true")
	}
}

func TestCompletionStreamInvalidJSONWithIncludeRawChunksEmitsRawWithNilDecoded(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().addRaw("not-valid-json").done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions:  GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
		IncludeRawChunks: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	var rawPart *StreamRaw
	for i := range parts {
		if r, ok := parts[i].(StreamRaw); ok {
			rawPart = &r
			break
		}
	}
	if rawPart == nil {
		t.Fatal("no StreamRaw for invalid JSON")
	}
	if rawPart.Decoded != nil {
		t.Fatalf("Decoded should be nil for invalid JSON, got %#v", rawPart.Decoded)
	}
}

func TestCompletionStreamFirstValidChunkEmitsResponseMetadataThenTextStart(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	var events []string
	for _, p := range parts {
		switch p.(type) {
		case StreamResponseMetadata:
			events = append(events, "meta")
		case StreamTextStart:
			events = append(events, "text-start")
		}
	}
	// Meta must come before text-start.
	metaIdx, textIdx := -1, -1
	for i, e := range events {
		if e == "meta" {
			metaIdx = i
		}
		if e == "text-start" {
			textIdx = i
		}
	}
	if metaIdx < 0 || textIdx < 0 {
		t.Fatalf("events = %v", events)
	}
	if metaIdx > textIdx {
		t.Fatalf("meta after text-start: %v", events)
	}
}

func TestCompletionStreamTextDeltasUseID0(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	for _, p := range parts {
		switch part := p.(type) {
		case StreamTextStart:
			if part.ID != "0" {
				t.Fatalf("StreamTextStart.ID = %q", part.ID)
			}
		case StreamTextDelta:
			if part.ID != "0" {
				t.Fatalf("StreamTextDelta.ID = %q", part.ID)
			}
		case StreamTextEnd:
			if part.ID != "0" {
				t.Fatalf("StreamTextEnd.ID = %q", part.ID)
			}
		}
	}
}

func TestCompletionStreamTextEndEmitsOnFlushAfterValidChunks(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("a")).add(completionChunk("b")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	hasEnd := false
	for _, p := range parts {
		if _, ok := p.(StreamTextEnd); ok {
			hasEnd = true
		}
	}
	if !hasEnd {
		t.Fatal("no StreamTextEnd on flush")
	}
}

func TestCompletionStreamLatestUsageChunkConvertedAtFinish(t *testing.T) {
	usageChunk := `{"id":"cmp-1","created":10,"model":"gpt-3","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":4}}`
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).add(usageChunk).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil {
		t.Fatal("no StreamFinish")
	}
	if finish.Usage.InputTokens.Total == nil || *finish.Usage.InputTokens.Total != 8 {
		t.Fatalf("usage = %#v", finish.Usage)
	}
	if finish.Usage.OutputTokens.Total == nil || *finish.Usage.OutputTokens.Total != 4 {
		t.Fatalf("usage = %#v", finish.Usage)
	}
}

func TestCompletionStreamFinishProviderMetadataIsEmpty(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil {
		t.Fatal("no StreamFinish")
	}
	// ProviderMetadata must be zero/nil for completion streams.
	if len(finish.ProviderMetadata) != 0 {
		t.Fatalf("ProviderMetadata = %#v, want empty", finish.ProviderMetadata)
	}
}

func TestCompletionStreamFinishReasonUpdatesFromChunks(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).add(completionChunkWithFinish("", "length")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	var finish *StreamFinish
	for i := range parts {
		if f, ok := parts[i].(StreamFinish); ok {
			finish = &f
		}
	}
	if finish == nil {
		t.Fatal("no StreamFinish")
	}
	if finish.FinishReason.Unified != "length" {
		t.Fatalf("finish reason = %q", finish.FinishReason.Unified)
	}
}

func TestCompletionStreamErrorChunkEmitsStreamErrorThenFinish(t *testing.T) {
	errChunk := `{"error":{"message":"completion error","type":"server_error"}}`
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(errChunk).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	var streamErr *StreamError
	var finish *StreamFinish
	for i := range parts {
		switch p := parts[i].(type) {
		case StreamError:
			streamErr = &p
		case StreamFinish:
			finish = &p
		}
	}
	if streamErr == nil {
		t.Fatal("no StreamError for error chunk")
	}
	apiErr, ok := streamErr.Err.(APIError)
	if !ok || apiErr.Message != "completion error" {
		t.Fatalf("error = %v", streamErr.Err)
	}
	if finish == nil || finish.FinishReason.Unified != "error" {
		t.Fatalf("finish = %#v", finish)
	}
}

func TestCompletionStreamParseFailureEmitsStreamErrorThenFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().addRaw("not-json").done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	hasError, hasFinish := false, false
	for _, p := range parts {
		switch p.(type) {
		case StreamError:
			hasError = true
		case StreamFinish:
			hasFinish = true
		}
	}
	if !hasError {
		t.Fatal("no StreamError for parse failure")
	}
	if !hasFinish {
		t.Fatal("no StreamFinish for parse failure")
	}
}

func TestCompletionStreamSuccessfulStreamExactlyOneFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().add(completionChunk("hi")).done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	finishCount := 0
	for _, p := range parts {
		if _, ok := p.(StreamFinish); ok {
			finishCount++
		}
	}
	if finishCount != 1 {
		t.Fatalf("finish count = %d, want 1", finishCount)
	}
}

func TestCompletionStreamFatalStreamAtMostOneErrorExactlyOneFinish(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().addRaw("{invalid").done().build()),
	}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStream(result.Stream)
	errCount, finishCount := 0, 0
	for _, p := range parts {
		switch p.(type) {
		case StreamError:
			errCount++
		case StreamFinish:
			finishCount++
		}
	}
	if errCount > 1 {
		t.Fatalf("error count = %d, want at most 1", errCount)
	}
	if finishCount != 1 {
		t.Fatalf("finish count = %d, want 1", finishCount)
	}
}

func TestCompletionStreamPreStreamNon2xxReturnsAPICallError(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(400, `{"error":{"message":"bad request"}}`)}}
	p := completionProvider(t, f)
	_, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err == nil {
		t.Fatal("expected error for non-2xx")
	}
	apiErr := new(APICallError)
	if !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Fatalf("error = %v", err)
	}
}

func TestCompletionStreamMidStreamFailuresAreNotRetried(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{
		completionSseResponse(sse().addRaw("{invalid").done().build()),
	}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test",
		Name:    "acme",
		Fetch:   f,
		Retry:   &RetryOptions{MaxRetries: 3, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond},
	})
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	if f.calls != 1 {
		t.Fatalf("calls = %d, want 1 (mid-stream not retried)", f.calls)
	}
}

func TestCompletionStreamResponseBodyClosesOnNormalCompletion(t *testing.T) {
	var closed atomic.Bool
	type closingBody struct {
		io.Reader
	}
	body := &struct {
		io.Reader
		closed *atomic.Bool
	}{
		Reader: strings.NewReader(sse().add(completionChunk("hi")).done().build()),
		closed: &closed,
	}
	// Use trackedBody from chat_stream_test.go.
	tracked := &trackedBody{Reader: strings.NewReader(sse().add(completionChunk("hi")).done().build())}
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: tracked}
	_ = body // suppress unused
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	if !tracked.closed.Load() {
		t.Fatal("body not closed on normal completion")
	}
}

func TestCompletionStreamResponseBodyClosesOnFatalParseError(t *testing.T) {
	tracked := &trackedBody{Reader: strings.NewReader(sse().addRaw("{invalid").done().build())}
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: tracked}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	if !tracked.closed.Load() {
		t.Fatal("body not closed on fatal parse error")
	}
}

func TestCompletionStreamResponseBodyClosesOnContextCancellation(t *testing.T) {
	// Build a stream that hangs after the first chunk.
	tracked := &trackedBody{Reader: strings.NewReader(
		sse().add(completionChunk("hi")).build() + "data: never\n",
	)}
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: tracked}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := completionProvider(t, f)
	ctx, cancel := context.WithCancel(context.Background())
	result, err := p.Completion("gpt-3").DoStream(ctx, StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Consume start, then cancel.
	<-result.Stream
	cancel()
	collectStream(result.Stream)
	if !tracked.closed.Load() {
		t.Fatal("body not closed after context cancellation")
	}
}

func TestCompletionStreamGoroutineExitsAndChannelClosesOnContextCancellation(t *testing.T) {
	tracked := &trackedBody{Reader: strings.NewReader(
		sse().add(completionChunk("hi")).build() + "data: never\n",
	)}
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: tracked}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := completionProvider(t, f)
	ctx, cancel := context.WithCancel(context.Background())
	result, err := p.Completion("gpt-3").DoStream(ctx, StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	<-result.Stream // start
	cancel()
	// Channel must close (goroutine must exit).
	deadline := time.After(3 * time.Second)
	for {
		select {
		case _, ok := <-result.Stream:
			if !ok {
				return // channel closed – pass
			}
		case <-deadline:
			t.Fatal("channel did not close after context cancellation")
		}
	}
}

func TestCompletionStreamResponseIncludesClonedHeadersAndMetadata(t *testing.T) {
	headers := http.Header{"X-Custom": []string{"original"}}
	resp := &http.Response{
		StatusCode: 200,
		Header:     headers,
		Body:       io.NopCloser(strings.NewReader(sse().add(completionChunk("hi")).done().build())),
	}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := completionProvider(t, f)
	result, err := p.Completion("gpt-3").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	collectStream(result.Stream)
	headers.Set("X-Custom", "changed")
	if got := result.Response.Headers.Get("X-Custom"); got != "original" {
		t.Fatalf("headers not cloned: %q", got)
	}
	if result.Response.ID != "cmp-1" || result.Response.ModelID != "gpt-3" {
		t.Fatalf("response metadata = %#v", result.Response)
	}
}
