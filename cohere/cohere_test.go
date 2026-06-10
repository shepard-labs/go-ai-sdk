package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"
)

type mockFetcher struct {
	responses []*http.Response
	errs      []error
	requests  []*http.Request
	bodies    [][]byte
}

func (m *mockFetcher) Do(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	m.requests = append(m.requests, req)
	m.bodies = append(m.bodies, body)
	i := len(m.requests) - 1
	var err error
	if i < len(m.errs) {
		err = m.errs[i]
	}
	if err != nil {
		return nil, err
	}
	if i < len(m.responses) {
		return m.responses[i], nil
	}
	return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("{}"))}, nil
}

func jsonResp(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{"x-request-id": {"rid"}}, Body: io.NopCloser(strings.NewReader(body))}
}

func decodeBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestProviderAuthHeadersAndMissingKey(t *testing.T) {
	fetch := &mockFetcher{responses: []*http.Response{jsonResp(200, `{"generation_id":"g","message":{"role":"assistant","content":[]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}`)}}
	p := CreateCohere(ProviderSettings{BaseURL: "https://example.test/v2/", APIKey: "key", Headers: http.Header{"x-custom": {"provider"}}, Fetch: fetch, GenerateID: func() string { return "req-id" }, Retry: &RetryOptions{MaxRetries: 0}})
	_, err := p.LanguageModel("command").DoGenerate(context.Background(), GenerateOptions{Headers: http.Header{"x-custom": {"call"}}})
	if err != nil {
		t.Fatal(err)
	}
	req := fetch.requests[0]
	if req.URL.String() != "https://example.test/v2/chat" {
		t.Fatalf("url = %s", req.URL.String())
	}
	if req.Header.Get("Authorization") != "Bearer key" || req.Header.Get("User-Agent") != "ai-sdk-go/cohere/"+Version || req.Header.Get("Content-Type") != "application/json" || req.Header.Get("x-request-id") != "req-id" || req.Header.Get("x-custom") != "call" {
		t.Fatalf("unexpected headers: %#v", req.Header)
	}
	if p.Name() != "cohere" || p.LanguageModel("m").Provider() != "cohere.chat" || p.EmbeddingModel("e").Provider() != "cohere.textEmbedding" || p.RerankingModel("r").Provider() != "cohere.reranking" {
		t.Fatal("unexpected provider names")
	}

	missingFetch := &mockFetcher{}
	missing := CreateCohere(ProviderSettings{Fetch: missingFetch})
	if !errors.Is(missing.Err(), ErrMissingAPIKey) {
		t.Fatalf("Err() = %v", missing.Err())
	}
	if _, err := missing.LanguageModel("m").DoGenerate(context.Background(), GenerateOptions{}); !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("missing key error = %v", err)
	}
	if len(missingFetch.requests) != 0 {
		t.Fatal("missing key performed network I/O")
	}
}

func TestPromptConversionAndTools(t *testing.T) {
	u, _ := url.Parse("https://example.com/image.png")
	prompt, err := convertToCohereChatPrompt([]Message{
		SystemMessage{Content: "sys"},
		UserMessage{Content: []UserContent{TextContent{Text: "hello"}, FileContent{Data: u, MediaType: "image/png", ProviderOptions: ProviderMetadata{"cohere": map[string]any{"detail": "high"}}}, FileContent{Data: []byte("doc"), MediaType: "text/plain", Filename: "doc.txt"}}},
		AssistantMessage{Content: []AssistantContent{ReasoningContent{Text: "ignored"}, ToolCallContent{ToolCallID: "tc", ToolName: "fn", Input: json.RawMessage(`{"a":1}`)}}},
		ToolMessage{Content: []ToolContent{ToolResultContent{ToolCallID: "tc", Output: ToolResultOutput{Type: "json", Value: map[string]any{"ok": true}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Messages[1]["content"].([]map[string]any)[1]["image_url"].(map[string]any)["detail"] != "high" {
		t.Fatal("image detail not preserved")
	}
	if prompt.Documents[0].Data["title"] != "doc.txt" || prompt.Documents[0].Data["text"] != "doc" {
		t.Fatalf("unexpected docs: %#v", prompt.Documents)
	}
	if _, ok := prompt.Messages[2]["content"]; ok {
		t.Fatal("assistant tool call emitted content")
	}

	tools, choice, warnings, err := prepareTools([]Tool{{Type: "provider", ID: "web"}, {Type: "function", Name: "fn", Description: "", InputSchema: map[string]any{"type": "object"}}, {Type: "function", Name: "other"}}, &ToolChoice{Type: "tool", ToolName: "fn"})
	if err != nil {
		t.Fatal(err)
	}
	if choice == nil || *choice != "REQUIRED" || len(tools) != 1 || warnings[0].Feature != "provider-defined tool web" {
		t.Fatalf("unexpected tools=%#v choice=%v warnings=%#v", tools, choice, warnings)
	}
	fn := tools[0]["function"].(map[string]any)
	if _, ok := fn["description"]; ok {
		t.Fatal("empty description was not omitted")
	}

	_, err = convertToCohereChatPrompt([]Message{UserMessage{Content: []UserContent{FileContent{Data: []byte("x"), MediaType: "application/pdf"}}}})
	var unsupported UnsupportedFunctionalityError
	if !errors.As(err, &unsupported) || unsupported.Message != "Media type 'application/pdf' is not supported. Supported media types are: text/* and application/json." {
		t.Fatalf("unexpected unsupported error: %v", err)
	}
}

func TestGenerateRequestAndResponseParsing(t *testing.T) {
	fetch := &mockFetcher{responses: []*http.Response{jsonResp(200, `{"generation_id":"gen","message":{"role":"assistant","content":[{"type":"text","text":"hi"},{"type":"thinking","thinking":"why"}],"citations":[{"start":0,"end":2,"text":"hi","type":"TEXT_CONTENT","sources":[{"document":{"text":"doc","title":"Title"}}]}],"tool_calls":[{"id":"tc","type":"function","function":{"name":"fn","arguments":"null"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":3,"output_tokens":4}}}`)}}
	p := CreateCohere(ProviderSettings{APIKey: "key", Fetch: fetch, GenerateID: func() string { return "src" }, Retry: &RetryOptions{MaxRetries: 0}})
	max, topK, seed := 5, 7, 9
	temp, topP, freq, pres := 0.3, 0.8, 0.1, 0.2
	budget := 10
	res, err := p.LanguageModel("command").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, MaxOutputTokens: &max, Temperature: &temp, TopK: &topK, TopP: &topP, FrequencyPenalty: &freq, PresencePenalty: &pres, Seed: &seed, StopSequences: []string{"stop"}, ResponseFormat: &ResponseFormat{Type: "json", Schema: map[string]any{"type": "object"}}, StructuredOutput: &StructuredOutput{Schema: map[string]any{"type": "object"}}, ProviderOptions: ProviderOptions{"cohere": {"thinking": map[string]any{"tokenBudget": budget}}}})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeBody(t, fetch.bodies[0])
	for _, key := range []string{"model", "messages", "max_tokens", "temperature", "p", "k", "seed", "stop_sequences", "frequency_penalty", "presence_penalty", "response_format", "thinking"} {
		if _, ok := body[key]; !ok {
			t.Fatalf("missing %s in %#v", key, body)
		}
	}
	if res.Warnings[0].Message != "StructuredOutput takes precedence over ResponseFormat." || res.FinishReason.Unified != "tool-calls" || res.Usage.Raw == nil || res.Response.ID != "gen" {
		t.Fatalf("unexpected result: %#v", res)
	}
	if string(res.Content[3].(ToolCallContent).Input) != "{}" || res.Content[2].(SourceContent).Title != "Title" {
		t.Fatalf("unexpected content: %#v", res.Content)
	}
}

func TestStreamingStateRawAndMalformed(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"type":"message-start","id":"msg"}`,
		``,
		`data: {"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}`,
		``,
		`data: {bad`,
		``,
		`data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"hi"}}}}`,
		``,
		`data: {"type":"content-end","index":0}`,
		``,
		`data: {"type":"tool-call-start","delta":{"message":{"tool_calls":{"id":"tc","type":"function","function":{"name":"fn","arguments":"{ \"b\" : 1"}}}}}`,
		``,
		`data: {"type":"tool-call-delta","delta":{"message":{"tool_calls":{"function":{"arguments":" }"}}}}}`,
		``,
		`data: {"type":"tool-call-end"}`,
		``,
		`data: {"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}}`,
		``,
	}, "\n")
	fetch := &mockFetcher{responses: []*http.Response{{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(stream))}}}
	p := CreateCohere(ProviderSettings{APIKey: "key", Fetch: fetch, Retry: &RetryOptions{MaxRetries: 0}})
	res, err := p.LanguageModel("command").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{}, IncludeRawChunks: true})
	if err != nil {
		t.Fatal(err)
	}
	var parts []StreamPart
	for part := range res.Stream {
		parts = append(parts, part)
	}
	seenRaw, seenErr, seenTool := false, false, false
	finish := FinishReason{}
	for _, part := range parts {
		switch p := part.(type) {
		case StreamRaw:
			seenRaw = true
		case StreamError:
			seenErr = true
		case StreamToolCall:
			seenTool = string(p.Input) == `{"b":1}`
		case StreamFinish:
			finish = p.FinishReason
		}
	}
	if !seenRaw || !seenErr || !seenTool || finish.Unified != "stop" {
		t.Fatalf("unexpected stream parts: %#v finish=%#v", parts, finish)
	}
}

func TestEmbeddingRerankingErrorsRetryAndLimits(t *testing.T) {
	fetch := &mockFetcher{responses: []*http.Response{
		jsonResp(200, `{"embeddings":{"float":[[1,2]]},"meta":{"billed_units":{"input_tokens":7}}}`),
		jsonResp(200, `{"id":"rr","results":[{"index":1,"relevance_score":0.9}],"meta":{}}`),
	}}
	p := CreateCohere(ProviderSettings{APIKey: "key", Fetch: fetch, Retry: &RetryOptions{MaxRetries: 0}})
	dim := 512
	emb, err := p.EmbeddingModel("embed").DoEmbed(context.Background(), EmbedOptions{Values: []string{"a"}, ProviderOptions: ProviderOptions{"cohere": {"inputType": "search_document", "truncate": "END", "outputDimension": dim}}})
	if err != nil || !reflect.DeepEqual(emb.Embeddings, [][]float64{{1, 2}}) || emb.Usage.Tokens != 7 {
		t.Fatalf("embed=%#v err=%v", emb, err)
	}
	embBody := decodeBody(t, fetch.bodies[0])
	if embBody["input_type"] != "search_document" || embBody["truncate"] != "END" || embBody["output_dimension"].(float64) != 512 {
		t.Fatalf("bad embed body %#v", embBody)
	}
	maxDoc := 100
	rerank, err := p.RerankingModel("rerank").DoRerank(context.Background(), RerankOptions{Query: "q", Documents: ObjectDocuments(map[string]any{"a": 1}), ProviderOptions: ProviderOptions{"cohere": {"maxTokensPerDoc": maxDoc, "priority": 1}}})
	if err != nil || rerank.Response.ID != "rr" || rerank.Ranking[0].Index != 1 || rerank.Warnings[0].Feature != "object documents" {
		t.Fatalf("rerank=%#v err=%v", rerank, err)
	}

	tooMany := make([]string, 97)
	before := len(fetch.requests)
	_, err = p.EmbeddingModel("embed").DoEmbed(context.Background(), EmbedOptions{Values: tooMany})
	var tooManyErr TooManyEmbeddingValuesForCallError
	if !errors.As(err, &tooManyErr) || len(fetch.requests) != before {
		t.Fatalf("too many err=%v requests=%d before=%d", err, len(fetch.requests), before)
	}

	errFetch := &mockFetcher{responses: []*http.Response{jsonResp(429, `{"message":"slow down"}`), jsonResp(200, `{"generation_id":"g","message":{"role":"assistant","content":[]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}`)}}
	retryProvider := CreateCohere(ProviderSettings{APIKey: "key", Fetch: errFetch, Retry: &RetryOptions{MaxRetries: 1, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}})
	_, err = retryProvider.LanguageModel("command").DoGenerate(context.Background(), GenerateOptions{})
	if err != nil || len(errFetch.requests) != 2 {
		t.Fatalf("retry err=%v requests=%d", err, len(errFetch.requests))
	}

	apiFetch := &mockFetcher{responses: []*http.Response{jsonResp(400, `{"message":"bad request"}`)}}
	apiProvider := CreateCohere(ProviderSettings{APIKey: "key", Fetch: apiFetch, Retry: &RetryOptions{MaxRetries: 0}})
	_, err = apiProvider.LanguageModel("command").DoGenerate(context.Background(), GenerateOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) || apiErr.Message != "bad request" || apiErr.Retryable {
		t.Fatalf("api err=%#v", err)
	}

	truncFetch := &mockFetcher{responses: []*http.Response{{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("abcdef"))}}}
	truncProvider := CreateCohere(ProviderSettings{APIKey: "key", Fetch: truncFetch, MaxResponseBodyBytes: 3, Retry: &RetryOptions{MaxRetries: 0}})
	_, err = truncProvider.LanguageModel("command").DoGenerate(context.Background(), GenerateOptions{})
	if !errors.As(err, &apiErr) || !apiErr.Truncated {
		t.Fatalf("trunc err=%#v", err)
	}
}
