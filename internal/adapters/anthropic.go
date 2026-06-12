package adapters

import (
	"context"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
)

// BuildAnthropicMessages converts a (prompt, system) pair into the
// []anthropic.Message shape. The system message (if non-empty) is
// prepended; the user message is always appended.
func BuildAnthropicMessages(prompt, system string) []anthropic.Message {
	msgs := make([]anthropic.Message, 0, 2)
	if system != "" {
		msgs = append(msgs, anthropic.SystemMessage{Content: system})
	}
	msgs = append(msgs, anthropic.UserMessage{Content: []anthropic.UserContent{
		anthropic.TextContent{Text: prompt},
	}})
	return msgs
}

// BuildAnthropicGenerateOpts translates the provider-agnostic call
// input into anthropic.GenerateOptions. The key translation is
// maxTokens *int → anthropic.GenerateOptions.MaxTokens int.
func BuildAnthropicGenerateOpts(prompt, system string, temperature *float64, maxTokens *int) anthropic.GenerateOptions {
	gen := anthropic.GenerateOptions{
		Messages: BuildAnthropicMessages(prompt, system),
	}
	if temperature != nil {
		gen.Temperature = temperature
	}
	gen.MaxTokens = MaxTokensAsInt(maxTokens)
	return gen
}

// BuildAnthropicStreamOpts is the streaming counterpart. The Anthropic
// package aliases StreamOptions = GenerateOptions, so the shapes match.
func BuildAnthropicStreamOpts(prompt, system string, temperature *float64, maxTokens *int) anthropic.StreamOptions {
	return anthropic.StreamOptions(BuildAnthropicGenerateOpts(prompt, system, temperature, maxTokens))
}

// GenerateAnthropic runs the chosen Anthropic model and returns its
// typed result.
func GenerateAnthropic(ctx context.Context, m anthropic.LanguageModel, opts anthropic.GenerateOptions) (anthropic.GenerateResult, error) {
	res, err := m.DoGenerate(ctx, opts)
	if err != nil {
		return anthropic.GenerateResult{}, err
	}
	if res == nil {
		return anthropic.GenerateResult{}, nil
	}
	return *res, nil
}

// StreamAnthropic runs the chosen Anthropic model in streaming mode.
func StreamAnthropic(ctx context.Context, m anthropic.LanguageModel, opts anthropic.StreamOptions) (*anthropic.StreamResult, error) {
	return m.DoStream(ctx, opts)
}
