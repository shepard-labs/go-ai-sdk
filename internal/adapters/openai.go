package adapters

import (
	"context"

	"github.com/shepard-labs/go-ai-sdk/openai"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// BuildOpenAIMessages converts a (prompt, system) pair into the
// []openaicompatible.Message shape. The system message (if non-empty)
// is prepended; the user message is always appended. openai.* types
// are aliases for openaicompatible.* for these messages, so the
// constructed values satisfy both.
func BuildOpenAIMessages(prompt, system string) []openaicompatible.Message {
	msgs := make([]openaicompatible.Message, 0, 2)
	if system != "" {
		msgs = append(msgs, openaicompatible.SystemMessage{Content: system})
	}
	msgs = append(msgs, openaicompatible.UserMessage{Content: []openaicompatible.UserContent{
		openaicompatible.TextContent{Text: prompt},
	}})
	return msgs
}

// BuildOpenAIGenerateOpts translates the provider-agnostic call input
// into openaicompatible.GenerateOptions. The key translation is
// maxTokens *int → MaxOutputTokens *int.
func BuildOpenAIGenerateOpts(prompt, system string, temperature *float64, maxTokens *int) openaicompatible.GenerateOptions {
	gen := openaicompatible.GenerateOptions{
		Messages: BuildOpenAIMessages(prompt, system),
	}
	if temperature != nil {
		gen.Temperature = temperature
	}
	gen.MaxOutputTokens = MaxTokensCopy(maxTokens)
	return gen
}

// BuildOpenAIStreamOpts is the streaming counterpart. openaicompatible
// embeds GenerateOptions in StreamOptions.
func BuildOpenAIStreamOpts(prompt, system string, temperature *float64, maxTokens *int) openaicompatible.StreamOptions {
	return openaicompatible.StreamOptions{GenerateOptions: BuildOpenAIGenerateOpts(prompt, system, temperature, maxTokens)}
}

// GenerateOpenAI runs the chosen openai chat model and returns its
// typed result.
//
// IMPORTANT: callers must pass the model returned by
// openai.Provider.Chat(modelID), not openai.Provider.Model(modelID).
// The .Model() / .LanguageModel() / .Call() / .Responses() methods on
// the openai provider all return a ResponsesModel whose DoGenerate
// signature is *different* (it takes ResponsesGenerateOptions). The
// chat-style model returned by .Chat() is the one whose
// GenerateOptions matches the openaicompatible shape we translate to.
func GenerateOpenAI(ctx context.Context, m openai.LanguageModel, opts openaicompatible.GenerateOptions) (openai.GenerateResult, error) {
	res, err := m.DoGenerate(ctx, opts)
	if err != nil {
		return openai.GenerateResult{}, err
	}
	if res == nil {
		return openai.GenerateResult{}, nil
	}
	return *res, nil
}

// StreamOpenAI runs the chosen openai chat model in streaming mode.
func StreamOpenAI(ctx context.Context, m openai.LanguageModel, opts openaicompatible.StreamOptions) (*openai.StreamResult, error) {
	return m.DoStream(ctx, opts)
}
