package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeFetcher struct {
	t       *testing.T
	mu      sync.Mutex
	reqs    []*http.Request
	bodies  []map[string]any
	handler func(*http.Request, map[string]any) (*http.Response, error)
}

func (f *fakeFetcher) Do(req *http.Request) (*http.Response, error) {
	f.mu.Lock()
	f.reqs = append(f.reqs, req.Clone(req.Context()))
	var body map[string]any
	if req.Body != nil {
		raw, _ := io.ReadAll(req.Body)
		_ = json.Unmarshal(raw, &body)
	}
	f.bodies = append(f.bodies, body)
	f.mu.Unlock()
	return f.handler(req, body)
}

func jsonResp(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(body))}
}

func sseResp(body string) *http.Response {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(body))}
}

func TestProviderDefaultsHeadersAndRouting(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "env-key")
	f := &fakeFetcher{t: t, handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		if got := req.Header.Get("Authorization"); got != "Bearer env-key" {
			t.Fatalf("auth = %q", got)
		}
		if got := req.Header.Get("X-OpenRouter-Title"); got != "app" {
			t.Fatalf("title = %q", got)
		}
		if got := req.Header.Get("HTTP-Referer"); got != "https://app.test" {
			t.Fatalf("referer = %q", got)
		}
		if got := req.Header.Get("X-Provider-API-Keys"); !strings.Contains(got, "anthropic") {
			t.Fatalf("byok = %q", got)
		}
		if got := req.Header.Get("User-Agent"); got != "custom-agent" {
			t.Fatalf("user-agent override = %q", got)
		}
		return jsonResp(200, `{"id":"id","model":"m","provider":"p","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`), nil
	}}
	p := CreateOpenRouter(ProviderSettings{Fetch: f, Headers: http.Header{"User-Agent": {"custom-agent"}}, AppName: "app", AppURL: "https://app.test", APIKeys: map[string]string{"anthropic": "k"}})
	if p.Name() != "openrouter" {
		t.Fatalf("name = %s", p.Name())
	}
	if p.Model("openai/gpt-3.5-turbo-instruct").Provider() != "openrouter.completion" {
		t.Fatal("instruct model did not route to completion")
	}
	if p.Model("openai/gpt-4o").Provider() != "openrouter.chat" {
		t.Fatal("chat model did not route to chat")
	}
	_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChatRequestBodyPrecedenceAndTypedOptions(t *testing.T) {
	f := &fakeFetcher{t: t, handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		if body["x"].(string) != "call" {
			t.Fatalf("extra precedence = %#v", body["x"])
		}
		if body["temperature"].(float64) != 0.9 {
			t.Fatalf("temperature = %#v", body["temperature"])
		}
		if _, ok := body["stream_options"]; ok {
			t.Fatal("compatible mode included stream_options")
		}
		if _, ok := body["cache_control"]; !ok {
			t.Fatal("cacheControl was not normalized")
		}
		plugins := body["plugins"].([]any)
		if plugins[0].(map[string]any)["id"] != "web" {
			t.Fatalf("plugin id missing: %#v", plugins[0])
		}
		return sseResp("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"), nil
	}}
	tempModel, tempCall := 0.1, 0.9
	p := CreateOpenRouter(ProviderSettings{Fetch: f, ExtraBody: map[string]any{"x": "provider"}})
	model := p.Chat("m", ChatOptions{Temperature: &tempModel, ExtraBody: map[string]any{"x": "model"}, Plugins: []Plugin{WebPlugin{}}})
	res, err := model.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}, Temperature: &tempCall, ProviderOptions: ProviderOptions{"openrouter": {"x": "call", "cacheControl": CacheControl{Type: "ephemeral"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	for range res.Stream {
	}
}

func TestStrictStreamOptions(t *testing.T) {
	f := &fakeFetcher{handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		so, ok := body["stream_options"].(map[string]any)
		if !ok || so["include_usage"] != true {
			t.Fatalf("missing strict stream options: %#v", body["stream_options"])
		}
		return sseResp("data: [DONE]\n\n"), nil
	}}
	res, err := CreateOpenRouter(ProviderSettings{Fetch: f, Compatibility: CompatibilityStrict}).Chat("m").DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	for range res.Stream {
	}
}

func TestChatMessagesReasoningAndFiles(t *testing.T) {
	msgs, err := convertChatMessages([]Message{
		SystemMessage{Content: "sys"},
		UserMessage{ProviderOptions: ProviderOptions{"openrouter": {"cacheControl": CacheControl{Type: "ephemeral"}}}, Content: []UserContent{TextContent{Text: "look"}, FileContent{Data: []byte("abc"), MediaType: "image/png"}}},
		AssistantMessage{Content: []AssistantContent{ReasoningContent{Text: "think", ProviderMetadata: ProviderMetadata{"openrouter": map[string]any{"reasoning_details": []ReasoningDetail{{Type: "reasoning.text", Text: "bad", Format: "anthropic-claude-v1"}, {Type: "reasoning.text", Text: "ok", Signature: "sig", ID: "r1", Format: "anthropic-claude-v1"}}}}}, TextContent{Text: "answer"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if msgs[1].Content.([]apiPart)[1].Type != "image_url" {
		t.Fatalf("image part not converted: %#v", msgs[1].Content)
	}
	if len(msgs[2].ReasoningDetails) != 1 || msgs[2].ReasoningDetails[0].Text != "ok" {
		t.Fatalf("reasoning filter failed: %#v", msgs[2].ReasoningDetails)
	}
}

func TestChatResponseParsing(t *testing.T) {
	f := &fakeFetcher{handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		return jsonResp(200, `{"id":"id","model":"m","provider":"p","choices":[{"message":{"content":"text","reasoning_details":[{"type":"reasoning.encrypted","data":"secret"}],"tool_calls":[{"id":"","type":"function","function":{"name":"fn","arguments":"{\"a\":1}"}},{"id":"","type":"function","function":{"name":"fn2","arguments":"{}"}}],"images":[{"image_url":{"url":"data:image/png;base64,abc"}}],"annotations":[{"type":"url_citation","url_citation":{"url":"https://x","title":"X","content":"c","start_index":1,"end_index":2}},{"type":"file_citation","file_citation":{"file_id":"f"}}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},"completion_tokens_details":{"reasoning_tokens":1},"cost":0.01,"cost_details":{"upstream_inference_cost":0.02}}}`), nil
	}}
	p := CreateOpenRouter(ProviderSettings{Fetch: f, GenerateID: func() string { return "gen" }})
	res, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != FinishReasonToolCalls {
		t.Fatalf("finish = %s", res.FinishReason)
	}
	if res.Usage.InputTokensDetails.CachedTokens != 3 || res.Usage.OutputTokensDetails.ReasoningTokens != 1 {
		t.Fatalf("usage = %#v", res.Usage)
	}
	if len(res.Content) != 5 {
		t.Fatalf("content len = %d %#v", len(res.Content), res.Content)
	}
}

func TestHTTP200ErrorPayload(t *testing.T) {
	f := &fakeFetcher{handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		return jsonResp(200, `{"error":{"message":"bad","code":123,"metadata":{"provider_name":"x"}},"user_id":"u"}`), nil
	}}
	_, err := CreateOpenRouter(ProviderSettings{Fetch: f}).Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}})
	api, ok := err.(*APICallError)
	if !ok || api.StatusCode != 200 || api.ProviderName != "x" || api.UserID != "u" {
		t.Fatalf("err = %#v", err)
	}
}

func TestCompletionEmbeddingImageAndVideo(t *testing.T) {
	step := 0
	f := &fakeFetcher{handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		step++
		switch step {
		case 1:
			if req.URL.Path != "/api/v1/completions" || body["prompt"] != "hello" {
				t.Fatalf("completion request %#v %s", body, req.URL.Path)
			}
			return jsonResp(200, `{"model":"m","provider":"p","choices":[{"text":"done","finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`), nil
		case 2:
			if body["input"].([]any)[1] != "b" {
				t.Fatalf("embedding input = %#v", body["input"])
			}
			return jsonResp(200, `{"model":"e","provider":"p","data":[{"embedding":[2],"index":1},{"embedding":[1],"index":0}],"usage":{"prompt_tokens":2}}`), nil
		case 3:
			if req.URL.Path != "/api/v1/chat/completions" || body["modalities"] == nil {
				t.Fatalf("image request %#v", body)
			}
			return jsonResp(200, `{"id":"img","model":"im","choices":[{"message":{"images":[{"image_url":{"url":"data:image/png;base64,zzz"}}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`), nil
		case 4:
			if req.URL.Path != "/api/v1/videos" || body["size"] != "720p" {
				t.Fatalf("video submit %#v %s", body, req.URL.Path)
			}
			return jsonResp(200, `{"id":"vid","generation_id":"g","status":"queued"}`), nil
		case 5:
			if req.URL.Path != "/api/v1/videos/vid" {
				t.Fatalf("poll path %s", req.URL.Path)
			}
			return jsonResp(200, `{"id":"vid","generation_id":"g","status":"completed","unsigned_urls":["https://v"],"cost":0.5}`), nil
		}
		return jsonResp(500, `{}`), nil
	}}
	p := CreateOpenRouter(ProviderSettings{Fetch: f})
	c, err := p.Completion("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}})
	if err != nil || c.FinishReason != FinishReasonLength {
		t.Fatalf("completion %#v %v", c, err)
	}
	e, err := p.Embedding("e").DoEmbed(context.Background(), EmbedOptions{Values: []string{"a", "b"}})
	if err != nil || e.Embeddings[0][0] != 1 || !p.Embedding("e").SupportsParallelCalls() {
		t.Fatalf("embedding %#v %v", e, err)
	}
	i, err := p.Image("im").DoGenerate(context.Background(), ImageGenerateOptions{Prompt: "draw", N: 2, Size: "1024x1024", AspectRatio: "1:1", InputFiles: []FileContent{{Data: []byte("x"), MediaType: "image/png"}}})
	if err != nil || i.Images[0] != "zzz" || len(i.Warnings) != 2 {
		t.Fatalf("image %#v %v", i, err)
	}
	v, err := p.VideoModel("vm", VideoOptions{PollInterval: time.Millisecond, MaxPollTime: time.Second}).DoGenerate(context.Background(), VideoGenerateOptions{Prompt: "go", Resolution: "720p", N: 2})
	if err != nil || v.Videos[0].URL != "https://v" || len(v.Warnings) != 1 {
		t.Fatalf("video %#v %v", v, err)
	}
}

func TestVideoTimeoutAndCancellation(t *testing.T) {
	f := &fakeFetcher{handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		if req.Method == http.MethodPost {
			return jsonResp(200, `{"id":"vid","status":"queued"}`), nil
		}
		return jsonResp(200, `{"id":"vid","status":"running"}`), nil
	}}
	_, err := CreateOpenRouter(ProviderSettings{Fetch: f}).VideoModel("v", VideoOptions{PollInterval: time.Millisecond, MaxPollTime: 2 * time.Millisecond}).DoGenerate(context.Background(), VideoGenerateOptions{Prompt: "x"})
	api, ok := err.(*APICallError)
	if !ok || api.StatusCode != http.StatusRequestTimeout || !api.Retryable {
		t.Fatalf("timeout err = %#v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = CreateOpenRouter(ProviderSettings{Fetch: f}).VideoModel("v", VideoOptions{PollInterval: time.Millisecond}).DoGenerate(ctx, VideoGenerateOptions{Prompt: "x"})
	if err == nil {
		t.Fatal("expected cancellation")
	}
}

func TestHeaderExplicitAuthorizationOverride(t *testing.T) {
	os.Setenv("OPENROUTER_API_KEY", "env")
	defer os.Unsetenv("OPENROUTER_API_KEY")
	f := &fakeFetcher{handler: func(req *http.Request, body map[string]any) (*http.Response, error) {
		if req.Header.Get("Authorization") != "Bearer caller" {
			t.Fatalf("auth = %q", req.Header.Get("Authorization"))
		}
		return jsonResp(200, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`), nil
	}}
	_, err := CreateOpenRouter(ProviderSettings{Fetch: f, Headers: http.Header{"Authorization": {"Bearer caller"}}}).Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}})
	if err != nil {
		t.Fatal(err)
	}
}

var _ = bytes.Compare
