package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/openai"
)

// ----------------------------------------------------------------------------
// test helpers
// ----------------------------------------------------------------------------

type roundTripFetcher struct {
	fn func(*http.Request) (*http.Response, error)
}

func (f roundTripFetcher) Do(req *http.Request) (*http.Response, error) { return f.fn(req) }

func jsonResponse(status int, body string) *http.Response {
	resp := textResponse(status, body)
	resp.Header.Set("Content-Type", "application/json")
	return resp
}

func textResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// routingResponseBody returns a successful anthropic response that
// contains the routing JSON in a single text block — what a real
// claude-haiku-4-5 call with outputFormat would produce.
func routingResponseBody(payload string) string {
	resp := map[string]any{
		"id":          "msg_routing",
		"type":        "message",
		"role":        "assistant",
		"model":       "claude-haiku-4-5-20251001",
		"content":     []map[string]any{{"type": "text", "text": payload}},
		"stop_reason": "end_turn",
		"usage":       map[string]any{"input_tokens": 50, "output_tokens": 20},
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

// anthropicTextResponse is what a downstream Anthropic call returns
// when it picked a non-Haiku model.
func anthropicTextResponse(model, text string) string {
	resp := map[string]any{
		"id":          "msg_a",
		"type":        "message",
		"role":        "assistant",
		"model":       model,
		"content":     []map[string]any{{"type": "text", "text": text}},
		"stop_reason": "end_turn",
		"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

// openaiChatResponse is what a downstream OpenAI chat call returns.
func openaiChatResponse(model, content string) string {
	resp := map[string]any{
		"id":      "chatcmpl-a",
		"object":  "chat.completion",
		"model":   model,
		"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

// ----------------------------------------------------------------------------
// tests
// ----------------------------------------------------------------------------

func TestCreateRouterValidation(t *testing.T) {
	t.Run("empty catalog", func(t *testing.T) {
		r := CreateRouter(RouterSettings{
			Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{APIKey: "k"}),
		})
		if r.Err() != ErrEmptyCatalog {
			t.Fatalf("err = %v, want ErrEmptyCatalog", r.Err())
		}
	})

	t.Run("no providers", func(t *testing.T) {
		r := CreateRouter(RouterSettings{
			Catalog: ProviderCatalog{"anthropic": {"x"}},
		})
		if r.Err() != ErrNoProvidersConfigured {
			t.Fatalf("err = %v, want ErrNoProvidersConfigured", r.Err())
		}
	})

	t.Run("happy path", func(t *testing.T) {
		r := CreateRouter(RouterSettings{
			Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{APIKey: "k"}),
			Catalog:   ProviderCatalog{"anthropic": {"claude-sonnet-4-5"}},
		})
		if r.Err() != nil {
			t.Fatalf("err = %v, want nil", r.Err())
		}
	})
}

func TestRoutingCallShape(t *testing.T) {
	var routingBody []byte
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(req.Body)
		if strings.Contains(string(body), `format":{"type":"json_schema"`) {
			routingBody = body
		}
		return jsonResponse(200, routingResponseBody(`{"provider":"anthropic","model":"claude-sonnet-4-5","reason":"writing task"}`)), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-haiku-4-5-20251001", "claude-sonnet-4-5"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}

	_, sel, err := r.Generate(context.Background(), RouterOptions{
		Prompt: "Write a sonnet about the sea.",
	})
	if err != nil {
		t.Fatal(err)
	}
	if sel.Provider != "anthropic" || sel.Model != "claude-sonnet-4-5" {
		t.Fatalf("sel = %+v", sel)
	}

	// The captured body should be the routing call.
	if len(routingBody) == 0 {
		t.Fatalf("routingBody was empty — fetcher did not see the structured-output request")
	}
	var got map[string]any
	if err := json.Unmarshal(routingBody, &got); err != nil {
		t.Fatal(err)
	}
	if got["model"] != "claude-haiku-4-5-20251001" {
		t.Fatalf("model = %v", got["model"])
	}
	oc, ok := got["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("output_config missing or wrong type: %T", got["output_config"])
	}
	format, ok := oc["format"].(map[string]any)
	if !ok || format["type"] != "json_schema" {
		t.Fatalf("output_config.format = %#v", oc["format"])
	}
	// System message should contain the catalog entries.
	sys, ok := got["system"].([]any)
	if !ok || len(sys) == 0 {
		t.Fatalf("system = %#v", got["system"])
	}
	systemText := sys[0].(map[string]any)["text"].(string)
	if !strings.Contains(systemText, "claude-sonnet-4-5") || !strings.Contains(systemText, "claude-haiku-4-5-20251001") {
		t.Fatalf("system prompt missing catalog entries: %q", systemText)
	}
}

func TestDispatchToAnthropic(t *testing.T) {
	routing := routingResponseBody(`{"provider":"anthropic","model":"claude-sonnet-4-5","reason":"writing"}`)
	downstream := anthropicTextResponse("claude-sonnet-4-5", "answer")
	var bodies []string
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(req.Body)
		bodies = append(bodies, string(body))
		if strings.Contains(string(body), `format":{"type":"json_schema"`) {
			return jsonResponse(200, routing), nil
		}
		return jsonResponse(200, downstream), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-haiku-4-5-20251001", "claude-sonnet-4-5"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}

	res, sel, err := r.Generate(context.Background(), RouterOptions{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if sel.Model != "claude-sonnet-4-5" {
		t.Fatalf("sel = %+v", sel)
	}
	if res.Kind != ResultKindAnthropic || res.Anthropic == nil {
		t.Fatalf("res = %+v", res)
	}
	if len(res.Anthropic.Value.Content) == 0 {
		t.Fatalf("res.Anthropic.Value.Content is empty")
	}
	// The downstream body should have model == claude-sonnet-4-5.
	if len(bodies) < 2 {
		t.Fatalf("expected 2 bodies, got %d", len(bodies))
	}
	var downstreamBody map[string]any
	if err := json.Unmarshal([]byte(bodies[1]), &downstreamBody); err != nil {
		t.Fatal(err)
	}
	if downstreamBody["model"] != "claude-sonnet-4-5" {
		t.Fatalf("downstream model = %v, want claude-sonnet-4-5", downstreamBody["model"])
	}
}

func TestForceProviderBypass(t *testing.T) {
	var hits int
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		hits++
		body, _ := io.ReadAll(req.Body)
		// Should never see the routing call (no json_schema format).
		if strings.Contains(string(body), `format":{"type":"json_schema"`) {
			t.Fatalf("routing call happened despite ForceProvider: %s", body)
		}
		return jsonResponse(200, anthropicTextResponse("claude-sonnet-4-5", "forced answer")), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-sonnet-4-5"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}

	res, sel, err := r.Generate(context.Background(), RouterOptions{
		Prompt:        "hello",
		ForceProvider: "anthropic",
		ForceModel:    "claude-sonnet-4-5",
	})
	if err != nil {
		t.Fatal(err)
	}
	if hits != 1 {
		t.Fatalf("hits = %d, want 1 (routing bypassed)", hits)
	}
	if sel.Reason != "forced" {
		t.Fatalf("sel.Reason = %q", sel.Reason)
	}
	if res.Kind != ResultKindAnthropic {
		t.Fatalf("res.Kind = %v", res.Kind)
	}
}

func TestUnknownProviderFromHaiku(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		// Routing picks a provider not in the catalog.
		return jsonResponse(200, routingResponseBody(
			`{"provider":"google","model":"gemini-x","reason":"nope"}`)), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-haiku-4-5-20251001"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}

	_, sel, err := r.Generate(context.Background(), RouterOptions{Prompt: "hi"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if sel.Provider != "google" {
		t.Fatalf("sel.Provider = %q, want google", sel.Provider)
	}
	var nsp *NoSuchProviderError
	if !errors.As(err, &nsp) || nsp.Provider != "google" {
		t.Fatalf("err = %v (%T), want *NoSuchProviderError{Provider:google}", err, err)
	}
}

func TestRoutingDecisionMalformed(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		return jsonResponse(200, routingResponseBody("not json")), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-haiku-4-5-20251001"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}
	_, _, err := r.Generate(context.Background(), RouterOptions{Prompt: "x"})
	if err == nil || !strings.Contains(err.Error(), "malformed routing decision") {
		t.Fatalf("err = %v", err)
	}
}

func TestRoutingDecisionUnknownModel(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		return jsonResponse(200, routingResponseBody(
			`{"provider":"anthropic","model":"claude-evil-9000","reason":"x"}`)), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-sonnet-4-5"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}
	_, _, err := r.Generate(context.Background(), RouterOptions{Prompt: "x"})
	if err == nil || !strings.Contains(err.Error(), "model not in catalog") {
		t.Fatalf("err = %v", err)
	}
}

func TestStreamForwardsAnthropicResult(t *testing.T) {
	sse := "event: message_start\ndata: {\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\"}}\n\n" +
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n"
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(req.Body)
		if strings.Contains(string(body), `format":{"type":"json_schema"`) {
			return jsonResponse(200, routingResponseBody(
				`{"provider":"anthropic","model":"claude-sonnet-4-5","reason":"stream"}`)), nil
		}
		return textResponse(200, sse), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-haiku-4-5-20251001", "claude-sonnet-4-5"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}
	env, sel, err := r.Stream(context.Background(), RouterOptions{Prompt: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if sel.Model != "claude-sonnet-4-5" {
		t.Fatalf("sel = %+v", sel)
	}
	if env.Kind != StreamKindAnthropic || env.Anthropic == nil {
		t.Fatalf("env = %+v", env)
	}
	if env.Anthropic.Stream == nil {
		t.Fatal("env.Anthropic.Stream is nil")
	}
}

func TestDispatchToOpenAI(t *testing.T) {
	routing := routingResponseBody(`{"provider":"openai","model":"gpt-5","reason":"cheap"}`)
	downstream := openaiChatResponse("gpt-5", "hello from openai")
	var bodies []string
	fetcher := roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(req.Body)
		bodies = append(bodies, string(body))
		// The routing call is to anthropic /messages. The
		// downstream call goes to openai /chat/completions.
		if strings.Contains(string(body), `format":{"type":"json_schema"`) {
			return jsonResponse(200, routing), nil
		}
		return jsonResponse(200, downstream), nil
	}}
	r := CreateRouter(RouterSettings{
		Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{
			APIKey: "k", BaseURL: "https://anthropic.test", Fetch: fetcher,
		}),
		OpenAI: openai.CreateOpenAI(openai.ProviderSettings{
			APIKey: "k", BaseURL: "https://openai.test", Fetch: fetcher,
		}),
		Catalog: ProviderCatalog{
			"anthropic": {"claude-haiku-4-5-20251001"},
			"openai":    {"gpt-5"},
		},
	})
	if r.Err() != nil {
		t.Fatal(r.Err())
	}

	res, sel, err := r.Generate(context.Background(), RouterOptions{Prompt: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if sel.Provider != "openai" || sel.Model != "gpt-5" {
		t.Fatalf("sel = %+v", sel)
	}
	if res.Kind != ResultKindOpenAI || res.OpenAI == nil {
		t.Fatalf("res = %+v", res)
	}
	// The downstream call should have model == gpt-5.
	if len(bodies) < 2 {
		t.Fatalf("expected 2 bodies, got %d", len(bodies))
	}
	var downstreamBody map[string]any
	if err := json.Unmarshal([]byte(bodies[1]), &downstreamBody); err != nil {
		t.Fatal(err)
	}
	if downstreamBody["model"] != "gpt-5" {
		t.Fatalf("downstream model = %v, want gpt-5", downstreamBody["model"])
	}
}
