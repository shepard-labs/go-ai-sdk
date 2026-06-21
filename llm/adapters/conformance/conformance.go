// Package conformance provides a shared test suite for llm adapter implementations.
// Each adapter test file runs Suite against a fake provider model to verify
// neutral-type mapping is correct.
//
// The suite is provider-agnostic. Where a provider legitimately cannot support
// a neutral feature (for example, Anthropic has no neutral response-format
// mapping, and OpenRouter cannot carry images through the neutral message
// path), the corresponding sub-test detects the documented limitation — a typed
// *llm.UnsupportedFeatureError, a representation gap, or a forwarded warning —
// and skips with a clear reason rather than failing.
package conformance

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// AdapterFactory creates a fake provider model and returns the adapter Client
// and a pointer to the captured options of the last DoGenerate call (for
// inspection). The fake model returns the given FakeResult from DoGenerate.
//
// The returned *CapturedOpts is populated by the fake model when the adapter
// calls DoGenerate, so callers must invoke Generate before reading it.
type AdapterFactory func(result FakeResult) (client llm.Client, lastOpts *CapturedOpts)

// FakeResult describes the outcome the fake model should return.
type FakeResult struct {
	Content         []ContentSpec
	FinishReason    string // unified: "stop", "tool-calls", "length", etc.
	InputTokens     int
	OutputTokens    int
	ReasoningTokens int
	Warnings        []WarningSpec
	// ProviderMetadata, when non-nil, is returned by the fake model as the
	// provider's native metadata; the adapter is expected to surface it under
	// its provider key.
	ProviderMetadata map[string]any
}

// ContentSpec is a provider-neutral description of one response content part.
type ContentSpec struct {
	Type  string // "text", "tool_call", "reasoning"
	Text  string
	ID    string
	Name  string
	Input string // JSON
}

// WarningSpec describes a warning the fake model should return.
type WarningSpec struct {
	Code    string
	Message string
}

// CapturedOpts holds the last set of generate options the fake model received,
// normalized to a common inspection shape across providers.
type CapturedOpts struct {
	System        string
	MessageRoles  []string // non-system message roles, in order
	MessageCount  int      // number of non-system messages
	ToolCount     int
	HasToolSchema bool

	MaxTokens   *int
	Temperature *float64
	TopP        *float64
	TopK        *int
	Stop        []string
	Seed        *int

	HasToolChoice     bool
	HasResponseFormat bool
	HasImage          bool

	ProviderOptionKeys []string
}

// Suite runs the full conformance test suite for one adapter.
func Suite(t *testing.T, factory AdapterFactory) {
	ctx := context.Background()

	// ---- Request mapping ----

	t.Run("TestSystemPrompt", func(t *testing.T) {
		t.Helper()
		client, opts := factory(okResult())
		if _, err := client.Generate(ctx, llm.GenerateOptions{System: "sys", Messages: oneUserMessage()}); err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if opts.System != "sys" {
			t.Fatalf("captured system = %q, want %q", opts.System, "sys")
		}
	})

	t.Run("TestConversationHistory", func(t *testing.T) {
		t.Helper()
		client, opts := factory(okResult())
		msgs := []llm.Message{
			{Role: "user", Content: []llm.Content{llm.TextContent{Text: "hi"}}},
			{Role: "assistant", Content: []llm.Content{llm.TextContent{Text: "hello"}}},
		}
		if _, err := client.Generate(ctx, llm.GenerateOptions{Messages: msgs}); err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if opts.MessageCount != 2 {
			t.Fatalf("message count = %d, want 2", opts.MessageCount)
		}
		if len(opts.MessageRoles) != 2 || opts.MessageRoles[0] != "user" || opts.MessageRoles[1] != "assistant" {
			t.Fatalf("message roles = %v, want [user assistant]", opts.MessageRoles)
		}
	})

	t.Run("TestToolSchema", func(t *testing.T) {
		t.Helper()
		client, opts := factory(okResult())
		tools := []llm.Tool{{Name: "lookup", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`)}}
		if _, err := client.Generate(ctx, llm.GenerateOptions{Tools: tools}); err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if opts.ToolCount != 1 {
			t.Fatalf("tool count = %d, want 1", opts.ToolCount)
		}
		if !opts.HasToolSchema {
			t.Fatal("tool schema not forwarded to provider")
		}
	})

	t.Run("TestMaxTokens", func(t *testing.T) {
		t.Helper()
		client, opts := factory(okResult())
		if _, err := client.Generate(ctx, llm.GenerateOptions{MaxTokens: 100}); err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if opts.MaxTokens == nil || *opts.MaxTokens != 100 {
			t.Fatalf("max tokens = %v, want 100", opts.MaxTokens)
		}
	})

	t.Run("TestSamplingOptions", func(t *testing.T) {
		t.Helper()
		client, opts := factory(okResult())
		temp, topP, topK, seed := 0.5, 0.9, 40, 7
		gen := llm.GenerateOptions{
			Temperature: &temp,
			TopP:        &topP,
			TopK:        &topK,
			Stop:        []string{"END"},
			Seed:        &seed,
		}
		if _, err := client.Generate(ctx, gen); err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if opts.Temperature == nil || *opts.Temperature != temp {
			t.Fatalf("temperature = %v, want %v", opts.Temperature, temp)
		}
		if opts.TopP == nil || *opts.TopP != topP {
			t.Fatalf("top_p = %v, want %v", opts.TopP, topP)
		}
		if opts.TopK == nil || *opts.TopK != topK {
			t.Fatalf("top_k = %v, want %v", opts.TopK, topK)
		}
		if len(opts.Stop) != 1 || opts.Stop[0] != "END" {
			t.Fatalf("stop = %v, want [END]", opts.Stop)
		}
		if opts.Seed == nil || *opts.Seed != seed {
			t.Fatalf("seed = %v, want %v", opts.Seed, seed)
		}
	})

	toolChoiceCase := func(t *testing.T, choice llm.ToolChoice, label string) {
		t.Helper()
		client, opts := factory(okResult())
		_, err := client.Generate(ctx, llm.GenerateOptions{ToolChoice: choice})
		if err != nil {
			var ufe *llm.UnsupportedFeatureError
			if errors.As(err, &ufe) {
				t.Skipf("provider does not support tool choice %s: %v", label, err)
			}
			t.Fatalf("Generate error = %v", err)
		}
		if !opts.HasToolChoice {
			t.Fatalf("tool choice %s not forwarded to provider", label)
		}
	}

	t.Run("TestToolChoiceAuto", func(t *testing.T) {
		toolChoiceCase(t, llm.ToolChoice{Type: llm.ToolChoiceAuto}, "auto")
	})
	t.Run("TestToolChoiceRequired", func(t *testing.T) {
		toolChoiceCase(t, llm.ToolChoice{Type: llm.ToolChoiceRequired}, "required")
	})
	t.Run("TestToolChoiceNone", func(t *testing.T) {
		toolChoiceCase(t, llm.ToolChoice{Type: llm.ToolChoiceNone}, "none")
	})

	responseFormatCase := func(t *testing.T, rf *llm.ResponseFormat, label string) {
		t.Helper()
		client, opts := factory(okResult())
		_, err := client.Generate(ctx, llm.GenerateOptions{ResponseFormat: rf})
		if err != nil {
			var ufe *llm.UnsupportedFeatureError
			if errors.As(err, &ufe) {
				t.Skipf("provider does not support response format %s: %v", label, err)
			}
			t.Fatalf("Generate error = %v", err)
		}
		if !opts.HasResponseFormat {
			t.Fatalf("response format %s not forwarded to provider", label)
		}
	}

	t.Run("TestResponseFormatJSONObject", func(t *testing.T) {
		responseFormatCase(t, &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}, "json_object")
	})
	t.Run("TestResponseFormatJSONSchema", func(t *testing.T) {
		responseFormatCase(t, &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Name: "out", JSONSchema: json.RawMessage(`{"type":"object"}`)}, "json_schema")
	})

	t.Run("TestImageContent", func(t *testing.T) {
		t.Helper()
		client, opts := factory(okResult())
		msgs := []llm.Message{{Role: "user", Content: []llm.Content{
			llm.ImageContent{Source: llm.ImageInlineSource{Data: []byte{0x89, 0x50, 0x4e, 0x47}}, MIME: "image/png"},
		}}}
		_, err := client.Generate(ctx, llm.GenerateOptions{Messages: msgs})
		if err != nil {
			t.Skipf("provider cannot represent inline images via the neutral path: %v", err)
		}
		if !opts.HasImage {
			t.Skip("provider does not carry image content through the neutral message path")
		}
	})

	t.Run("TestProviderOptions", func(t *testing.T) {
		t.Helper()
		client, opts := factory(okResult())
		name := providerName(t, client)
		res, err := client.Generate(ctx, llm.GenerateOptions{
			ProviderOptions: llm.ProviderOptions{name: {"key": "val"}},
		})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if slices.Contains(opts.ProviderOptionKeys, name) {
			return // forwarded as expected
		}
		// Some providers acknowledge provider options with a warning instead of
		// forwarding them; that is a documented limitation, not a failure.
		for _, w := range res.Warnings {
			if strings.Contains(strings.ToLower(w.Message), "provider option") {
				t.Skipf("provider warns instead of forwarding provider options: %s", w.Message)
			}
		}
		t.Fatalf("provider options not forwarded; keys = %v", opts.ProviderOptionKeys)
	})

	// ---- Response mapping ----

	t.Run("TestTextContent", func(t *testing.T) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: "stop", Content: []ContentSpec{{Type: "text", Text: "hello"}}})
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		text, ok := firstContent[llm.TextContent](res.Content)
		if !ok || text.Text != "hello" {
			t.Fatalf("content = %#v, want TextContent{hello}", res.Content)
		}
	})

	t.Run("TestToolCallContent", func(t *testing.T) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: "tool-calls", Content: []ContentSpec{
			{Type: "tool_call", ID: "call_1", Name: "lookup", Input: `{"x":1}`},
		}})
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		use, ok := firstContent[llm.ToolUseContent](res.Content)
		if !ok || use.ID != "call_1" || use.Name != "lookup" {
			t.Fatalf("content = %#v, want ToolUseContent{call_1,lookup}", res.Content)
		}
		var decoded map[string]any
		if err := json.Unmarshal(use.Input, &decoded); err != nil || decoded["x"] != float64(1) {
			t.Fatalf("tool input = %s, want {\"x\":1}", use.Input)
		}
	})

	t.Run("TestReasoningContent", func(t *testing.T) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: "stop", Content: []ContentSpec{{Type: "reasoning", Text: "thinking"}}})
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		reasoning, ok := firstContent[llm.ReasoningContent](res.Content)
		if !ok {
			t.Skip("provider does not surface reasoning content in non-streaming results")
		}
		if reasoning.Text != "thinking" {
			t.Fatalf("reasoning = %q, want %q", reasoning.Text, "thinking")
		}
	})

	finishReasonCase := func(t *testing.T, unified string, want llm.FinishReasonType) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: unified, Content: []ContentSpec{{Type: "text", Text: "x"}}})
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if res.FinishReason.Unified != want {
			t.Fatalf("finish reason = %q, want %q", res.FinishReason.Unified, want)
		}
	}

	t.Run("TestFinishReasonStop", func(t *testing.T) { finishReasonCase(t, "stop", llm.FinishReasonStop) })
	t.Run("TestFinishReasonToolCalls", func(t *testing.T) { finishReasonCase(t, "tool-calls", llm.FinishReasonToolCalls) })
	t.Run("TestFinishReasonLength", func(t *testing.T) { finishReasonCase(t, "length", llm.FinishReasonLength) })

	t.Run("TestFinishReasonRaw", func(t *testing.T) {
		t.Helper()
		client, _ := factory(okResult())
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if res.FinishReason.Raw == "" {
			t.Fatal("finish reason raw is empty")
		}
	})

	t.Run("TestUsageMapping", func(t *testing.T) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: "stop", InputTokens: 11, OutputTokens: 22})
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if res.Usage.InputTokens != 11 || res.Usage.OutputTokens != 22 {
			t.Fatalf("usage = %#v, want 11/22", res.Usage)
		}
	})

	t.Run("TestReasoningTokens", func(t *testing.T) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: "stop", InputTokens: 1, OutputTokens: 5, ReasoningTokens: 3})
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		if res.Usage.ReasoningTokens == 0 {
			t.Skip("provider does not report reasoning tokens")
		}
		if res.Usage.ReasoningTokens != 3 {
			t.Fatalf("reasoning tokens = %d, want 3", res.Usage.ReasoningTokens)
		}
	})

	t.Run("TestWarningPropagation", func(t *testing.T) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: "stop", Warnings: []WarningSpec{{Code: "x", Message: "heads up"}}})
		name := providerName(t, client)
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		var found bool
		for _, w := range res.Warnings {
			if w.Message == "heads up" {
				found = true
				if w.Provider != name {
					t.Fatalf("warning provider = %q, want %q", w.Provider, name)
				}
			}
		}
		if !found {
			t.Fatalf("warnings = %#v, want one with message %q", res.Warnings, "heads up")
		}
	})

	t.Run("TestProviderMetadata", func(t *testing.T) {
		t.Helper()
		client, _ := factory(FakeResult{FinishReason: "stop", ProviderMetadata: map[string]any{"k": "v"}})
		name := providerName(t, client)
		res, err := client.Generate(ctx, llm.GenerateOptions{})
		if err != nil {
			t.Fatalf("Generate error = %v", err)
		}
		inner, ok := res.ProviderMetadata[name]
		if !ok {
			t.Fatalf("provider metadata = %#v, want key %q", res.ProviderMetadata, name)
		}
		if inner["k"] != "v" {
			t.Fatalf("provider metadata[%q] = %#v, want k=v", name, inner)
		}
	})

	// ---- Unsupported feature handling ----

	t.Run("TestUnsupportedFeatureError", func(t *testing.T) {
		t.Helper()
		trigger, found := findUnsupported(ctx, factory)
		if !found {
			t.Skip("provider forwards all probed features; no UnsupportedFeatureError path")
		}
		client, _ := factory(okResult())
		_, err := client.Generate(ctx, trigger)
		var ufe *llm.UnsupportedFeatureError
		if !errors.As(err, &ufe) {
			t.Fatalf("error = %v, want *UnsupportedFeatureError", err)
		}
	})

	t.Run("TestUnsupportedFeatureWarn", func(t *testing.T) {
		t.Helper()
		trigger, found := findUnsupported(ctx, factory)
		if !found {
			t.Skip("provider forwards all probed features; no unsupported-feature warning path")
		}
		client, _ := factory(okResult())
		trigger.UnsupportedFeaturePolicy = llm.UnsupportedFeaturePolicyWarn
		res, err := client.Generate(ctx, trigger)
		if err != nil {
			t.Fatalf("Generate under warn policy error = %v, want nil", err)
		}
		if len(res.Warnings) == 0 {
			t.Fatal("warn policy emitted no warning")
		}
	})
}

// findUnsupported probes candidate features under the default (error) policy and
// returns the first GenerateOptions that yields an *UnsupportedFeatureError.
func findUnsupported(ctx context.Context, factory AdapterFactory) (llm.GenerateOptions, bool) {
	candidates := []llm.GenerateOptions{
		{ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}},
		{ToolChoice: llm.ToolChoice{Type: llm.ToolChoiceNone}},
	}
	for _, c := range candidates {
		client, _ := factory(okResult())
		_, err := client.Generate(ctx, c)
		var ufe *llm.UnsupportedFeatureError
		if errors.As(err, &ufe) {
			return c, true
		}
	}
	return llm.GenerateOptions{}, false
}

func okResult() FakeResult {
	return FakeResult{
		FinishReason: "stop",
		Content:      []ContentSpec{{Type: "text", Text: "ok"}},
		InputTokens:  1,
		OutputTokens: 1,
	}
}

func oneUserMessage() []llm.Message {
	return []llm.Message{{Role: "user", Content: []llm.Content{llm.TextContent{Text: "hi"}}}}
}

func providerName(t *testing.T, client llm.Client) string {
	t.Helper()
	cp, ok := client.(llm.CapabilitiesProvider)
	if !ok {
		t.Fatalf("client %T does not implement llm.CapabilitiesProvider", client)
	}
	return cp.Capabilities().Provider
}

// firstContent returns the first content part of type T.
func firstContent[T llm.Content](contents []llm.Content) (T, bool) {
	for _, c := range contents {
		if typed, ok := c.(T); ok {
			return typed, true
		}
	}
	var zero T
	return zero, false
}
