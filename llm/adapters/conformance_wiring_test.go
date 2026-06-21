package adapters

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/google"
	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/adapters/conformance"
	"github.com/shepard-labs/go-ai-sdk/openai"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

// capturingClient forwards to an adapter Client and runs capture after each
// Generate to populate the shared CapturedOpts. It re-exposes Capabilities so
// the conformance suite can learn the provider key.
type capturingClient struct {
	inner   Client
	capture func()
}

func (c *capturingClient) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	res, err := c.inner.Generate(ctx, opts)
	c.capture()
	return res, err
}

func (c *capturingClient) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	return c.inner.Stream(ctx, opts)
}

func (c *capturingClient) Capabilities() Capabilities {
	return c.inner.(llm.CapabilitiesProvider).Capabilities()
}

func intPtrOrNil(v int) *int {
	if v == 0 {
		return nil
	}
	return &v
}

// ---- Anthropic ----

func TestAnthropicConformance(t *testing.T) {
	conformance.Suite(t, func(result conformance.FakeResult) (llm.Client, *conformance.CapturedOpts) {
		captured := &conformance.CapturedOpts{}
		model := &fakeAnthropicModel{result: anthropicResult(result)}
		client := &capturingClient{inner: NewAnthropicAdapter(model), capture: func() {
			opts := model.lastOptions
			captured.MaxTokens = intPtrOrNil(opts.MaxTokens)
			captured.Temperature = opts.Temperature
			captured.TopP = opts.TopP
			captured.TopK = opts.TopK
			captured.Stop = opts.StopSequences
			captured.Seed = opts.Seed
			captured.ToolCount = len(opts.Tools)
			captured.HasToolSchema = len(opts.Tools) > 0 && opts.Tools[0].InputSchema != nil
			captured.HasToolChoice = opts.ToolChoice != nil
			captured.HasResponseFormat = opts.ResponseFormat != nil
			for _, m := range opts.Messages {
				switch msg := m.(type) {
				case anthropic.SystemMessage:
					captured.System = msg.Content
				case anthropic.UserMessage:
					captured.MessageRoles = append(captured.MessageRoles, "user")
					captured.MessageCount++
					for _, c := range msg.Content {
						if _, ok := c.(anthropic.ImageContent); ok {
							captured.HasImage = true
						}
					}
				case anthropic.AssistantMessage:
					captured.MessageRoles = append(captured.MessageRoles, "assistant")
					captured.MessageCount++
				}
			}
		}}
		return client, captured
	})
}

func anthropicResult(r conformance.FakeResult) *anthropic.GenerateResult {
	out := &anthropic.GenerateResult{
		FinishReason: anthropic.FinishReason(r.FinishReason),
		Usage: anthropic.Usage{
			InputTokens:  anthropic.TokenUsage{Total: r.InputTokens},
			OutputTokens: anthropic.TokenUsage{Total: r.OutputTokens},
		},
	}
	for _, c := range r.Content {
		switch c.Type {
		case "text":
			out.Content = append(out.Content, anthropic.TextContent{Text: c.Text})
		case "tool_call":
			out.Content = append(out.Content, anthropic.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: json.RawMessage(c.Input)})
		case "reasoning":
			out.Content = append(out.Content, anthropic.ReasoningContent{Text: c.Text})
		}
	}
	for _, w := range r.Warnings {
		out.Warnings = append(out.Warnings, anthropic.Warning{Type: w.Code, Message: w.Message})
	}
	if r.ProviderMetadata != nil {
		out.ProviderMetadata = anthropic.ProviderMetadata(r.ProviderMetadata)
	}
	return out
}

// ---- OpenAI ----

func TestOpenAIConformance(t *testing.T) {
	conformance.Suite(t, func(result conformance.FakeResult) (llm.Client, *conformance.CapturedOpts) {
		captured := &conformance.CapturedOpts{}
		model := &fakeOpenAIModel{result: openAIResult(result)}
		client := &capturingClient{inner: NewOpenAIAdapter(model), capture: func() {
			captureOpenAICompatible(captured, model.lastOptions)
		}}
		return client, captured
	})
}

func openAIResult(r conformance.FakeResult) *openai.GenerateResult {
	out := &openai.GenerateResult{
		FinishReason: openai.FinishReason{Unified: r.FinishReason, Raw: r.FinishReason},
		Usage: openai.Usage{
			InputTokens:  openai.TokenCounts{Total: intPtrOrNil(r.InputTokens)},
			OutputTokens: openai.OutputTokenCounts{Total: intPtrOrNil(r.OutputTokens), Reasoning: intPtrOrNil(r.ReasoningTokens)},
		},
	}
	for _, c := range r.Content {
		switch c.Type {
		case "text":
			out.Content = append(out.Content, openai.TextContent{Text: c.Text})
		case "tool_call":
			out.Content = append(out.Content, openai.ToolCallContent{ToolCallContentEmbed: openai.ToolCallContentEmbed{ToolCallID: c.ID, ToolName: c.Name, Input: json.RawMessage(c.Input)}})
		case "reasoning":
			out.Content = append(out.Content, openai.ReasoningContent{Text: c.Text})
		}
	}
	for _, w := range r.Warnings {
		out.Warnings = append(out.Warnings, openai.Warning{Type: w.Code, Message: w.Message})
	}
	if r.ProviderMetadata != nil {
		out.ProviderMetadata = openai.ProviderMetadata(r.ProviderMetadata)
	}
	return out
}

// ---- OpenAI-compatible ----

func TestOpenAICompatibleConformance(t *testing.T) {
	conformance.Suite(t, func(result conformance.FakeResult) (llm.Client, *conformance.CapturedOpts) {
		captured := &conformance.CapturedOpts{}
		model := &fakeOpenAICompatibleModel{result: openAICompatibleResult(result)}
		client := &capturingClient{inner: NewOpenAICompatibleAdapter(model), capture: func() {
			captureOpenAICompatible(captured, model.lastOptions)
		}}
		return client, captured
	})
}

func openAICompatibleResult(r conformance.FakeResult) *openaicompatible.GenerateResult {
	out := &openaicompatible.GenerateResult{
		FinishReason: openaicompatible.FinishReason{Unified: r.FinishReason, Raw: r.FinishReason},
		Usage: openaicompatible.Usage{
			InputTokens:  openaicompatible.TokenCounts{Total: intPtrOrNil(r.InputTokens)},
			OutputTokens: openaicompatible.OutputTokenCounts{Total: intPtrOrNil(r.OutputTokens), Reasoning: intPtrOrNil(r.ReasoningTokens)},
		},
	}
	for _, c := range r.Content {
		switch c.Type {
		case "text":
			out.Content = append(out.Content, openaicompatible.TextContent{Text: c.Text})
		case "tool_call":
			out.Content = append(out.Content, openaicompatible.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: json.RawMessage(c.Input)})
		case "reasoning":
			out.Content = append(out.Content, openaicompatible.ReasoningContent{Text: c.Text})
		}
	}
	for _, w := range r.Warnings {
		out.Warnings = append(out.Warnings, openaicompatible.Warning{Type: w.Code, Message: w.Message})
	}
	if r.ProviderMetadata != nil {
		out.ProviderMetadata = openaicompatible.ProviderMetadata(r.ProviderMetadata)
	}
	return out
}

// captureOpenAICompatible populates captured from the shared openaicompatible
// GenerateOptions used by both the OpenAI and OpenAI-compatible adapters.
func captureOpenAICompatible(captured *conformance.CapturedOpts, opts openaicompatible.GenerateOptions) {
	captured.MaxTokens = opts.MaxOutputTokens
	captured.Temperature = opts.Temperature
	captured.TopP = opts.TopP
	captured.TopK = opts.TopK
	captured.Stop = opts.StopSequences
	captured.Seed = opts.Seed
	captured.ToolCount = len(opts.Tools)
	captured.HasToolSchema = len(opts.Tools) > 0 && opts.Tools[0].InputSchema != nil
	captured.HasToolChoice = opts.ToolChoice != nil
	captured.HasResponseFormat = opts.ResponseFormat != nil
	for k := range opts.ProviderOptions {
		captured.ProviderOptionKeys = append(captured.ProviderOptionKeys, k)
	}
	for _, m := range opts.Messages {
		switch msg := m.(type) {
		case openaicompatible.SystemMessage:
			captured.System = msg.Content
		case openaicompatible.UserMessage:
			captured.MessageRoles = append(captured.MessageRoles, "user")
			captured.MessageCount++
			for _, c := range msg.Content {
				if _, ok := c.(openaicompatible.FileContent); ok {
					captured.HasImage = true
				}
			}
		case openaicompatible.AssistantMessage:
			captured.MessageRoles = append(captured.MessageRoles, "assistant")
			captured.MessageCount++
		}
	}
}

// ---- Google ----

func TestGoogleConformance(t *testing.T) {
	conformance.Suite(t, func(result conformance.FakeResult) (llm.Client, *conformance.CapturedOpts) {
		captured := &conformance.CapturedOpts{}
		model := &fakeGoogleModel{result: googleResult(result)}
		client := &capturingClient{inner: NewGoogleAdapter(model), capture: func() {
			opts := model.lastOptions
			captured.MaxTokens = opts.MaxOutputTokens
			captured.Temperature = opts.Temperature
			captured.TopP = opts.TopP
			captured.TopK = opts.TopK
			captured.Stop = opts.StopSequences
			captured.Seed = opts.Seed
			captured.ToolCount = len(opts.Tools)
			captured.HasToolSchema = len(opts.Tools) > 0 && opts.Tools[0].InputSchema != nil
			captured.HasToolChoice = opts.ToolChoice != nil
			captured.HasResponseFormat = opts.ResponseFormat != nil
			for k := range opts.ProviderOptions {
				captured.ProviderOptionKeys = append(captured.ProviderOptionKeys, k)
			}
			for _, m := range opts.Messages {
				switch msg := m.(type) {
				case google.SystemMessage:
					captured.System = msg.Content
				case google.UserMessage:
					captured.MessageRoles = append(captured.MessageRoles, "user")
					captured.MessageCount++
					for _, c := range msg.Content {
						if _, ok := c.(google.ImageContent); ok {
							captured.HasImage = true
						}
					}
				case google.AssistantMessage:
					captured.MessageRoles = append(captured.MessageRoles, "assistant")
					captured.MessageCount++
				}
			}
		}}
		return client, captured
	})
}

func googleResult(r conformance.FakeResult) *google.GenerateResult {
	out := &google.GenerateResult{
		FinishReason: google.FinishReason{Unified: r.FinishReason, Raw: r.FinishReason},
		Usage: google.Usage{
			InputTokens:  google.TokenCounts{Total: intPtrOrNil(r.InputTokens)},
			OutputTokens: google.OutputTokenCounts{Total: intPtrOrNil(r.OutputTokens), Reasoning: intPtrOrNil(r.ReasoningTokens)},
		},
	}
	for _, c := range r.Content {
		switch c.Type {
		case "text":
			out.Content = append(out.Content, google.TextContent{Text: c.Text})
		case "tool_call":
			out.Content = append(out.Content, google.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: json.RawMessage(c.Input)})
		case "reasoning":
			out.Content = append(out.Content, google.ReasoningContent{Text: c.Text})
		}
	}
	for _, w := range r.Warnings {
		out.Warnings = append(out.Warnings, google.Warning{Type: w.Code, Message: w.Message})
	}
	if r.ProviderMetadata != nil {
		out.ProviderMetadata = google.ProviderMetadata(r.ProviderMetadata)
	}
	return out
}

// ---- OpenRouter ----

func TestOpenRouterConformance(t *testing.T) {
	conformance.Suite(t, func(result conformance.FakeResult) (llm.Client, *conformance.CapturedOpts) {
		captured := &conformance.CapturedOpts{}
		model := &fakeOpenRouterModel{result: openRouterResult(result)}
		client := &capturingClient{inner: NewOpenRouterAdapter(model), capture: func() {
			opts := model.lastOptions
			captured.MaxTokens = opts.MaxTokens
			captured.Temperature = opts.Temperature
			captured.TopP = opts.TopP
			captured.TopK = opts.TopK
			captured.Stop = opts.Stop
			captured.Seed = opts.Seed
			captured.ToolCount = len(opts.Tools)
			captured.HasToolSchema = len(opts.Tools) > 0 && opts.Tools[0].InputSchema != nil
			captured.HasToolChoice = opts.ToolChoice.Type != ""
			captured.HasResponseFormat = opts.ResponseFormat != nil
			for k := range opts.ProviderOptions {
				captured.ProviderOptionKeys = append(captured.ProviderOptionKeys, k)
			}
			for _, m := range opts.Messages {
				switch msg := m.(type) {
				case openrouter.SystemMessage:
					captured.System = msg.Content
				case openrouter.UserMessage:
					captured.MessageRoles = append(captured.MessageRoles, "user")
					captured.MessageCount++
				case openrouter.AssistantMessage:
					captured.MessageRoles = append(captured.MessageRoles, "assistant")
					captured.MessageCount++
				}
			}
		}}
		return client, captured
	})
}

func openRouterResult(r conformance.FakeResult) *openrouter.GenerateResult {
	out := &openrouter.GenerateResult{
		FinishReason: openrouter.FinishReason(r.FinishReason),
		Usage: openrouter.Usage{
			InputTokens:         r.InputTokens,
			OutputTokens:        r.OutputTokens,
			OutputTokensDetails: openrouter.OutputTokensDetails{ReasoningTokens: r.ReasoningTokens},
		},
	}
	for _, c := range r.Content {
		switch c.Type {
		case "text":
			out.Content = append(out.Content, openrouter.TextContent{Text: c.Text})
		case "tool_call":
			out.Content = append(out.Content, openrouter.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: json.RawMessage(c.Input)})
		case "reasoning":
			out.Content = append(out.Content, openrouter.ReasoningContent{Text: c.Text})
		}
	}
	for _, w := range r.Warnings {
		out.Warnings = append(out.Warnings, openrouter.Warning{Type: w.Code, Message: w.Message})
	}
	if r.ProviderMetadata != nil {
		out.ProviderMetadata = openrouter.ProviderMetadata(r.ProviderMetadata)
	}
	return out
}
