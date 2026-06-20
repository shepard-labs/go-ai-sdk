package adapters

import (
	"context"
	"net/http"

	openaisdk "github.com/shepard-labs/go-ai-sdk/openai"
)

// OpenAIAdapter adapts an OpenAI chat language model to the Client interface.
//
// The openai package re-exports its request/response types from
// openaicompatible, so message, tool, finish-reason, and usage translation is
// shared with the OpenAI-compatible adapter.
type OpenAIAdapter struct {
	model openaisdk.LanguageModel
}

// OpenAISettings configures an OpenAI Client.
type OpenAISettings struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// NewOpenAIAdapter wraps an existing OpenAI chat model as a Client.
func NewOpenAIAdapter(model openaisdk.LanguageModel) Client {
	return &OpenAIAdapter{model: model}
}

// NewOpenAIClient creates an OpenAI-backed Client from an API key and model ID.
func NewOpenAIClient(apiKey string, modelID string) (Client, error) {
	return NewOpenAIClientWithSettings(OpenAISettings{APIKey: apiKey}, modelID)
}

// NewOpenAIClientWithSettings creates an OpenAI-backed Client honoring BaseURL
// and a custom HTTP client.
func NewOpenAIClientWithSettings(settings OpenAISettings, modelID string) (Client, error) {
	providerSettings := openaisdk.ProviderSettings{APIKey: settings.APIKey, BaseURL: settings.BaseURL}
	if settings.HTTPClient != nil {
		providerSettings.Fetch = settings.HTTPClient
	}
	provider := openaisdk.CreateOpenAI(providerSettings)
	if err := provider.Err(); err != nil {
		return nil, err
	}
	return NewOpenAIAdapter(provider.Chat(modelID)), nil
}

// Generate sends a completion request through the OpenAI SDK.
func (a *OpenAIAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, err := toOpenAICompatibleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	return fromOpenAIResult(result), nil
}

// Stream sends a streaming completion request through the OpenAI SDK and maps
// provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *OpenAIAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, err := toOpenAICompatibleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoStream(ctx, openaisdk.StreamOptions{GenerateOptions: sdkOpts})
	if err != nil {
		return nil, err
	}
	out := make(chan StreamPart)
	go func() {
		defer close(out)
		parts := result.Parts
		if parts == nil {
			parts = result.Stream
		}
		for {
			select {
			case <-ctx.Done():
				out <- StreamError{Err: ctx.Err()}
				return
			case part, ok := <-parts:
				if !ok {
					return
				}
				if mapped, emit := mapOpenAIStreamPart(part); emit {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapOpenAIStreamPart converts an openaisdk.StreamPart to a neutral StreamPart.
// Returns (zero, false) for parts with no neutral equivalent (StreamStart,
// StreamResponseMetadata, StreamToolApprovalRequest, StreamCompactionEnd,
// SourceContent, StreamCustomPart). spec §1.1
func mapOpenAIStreamPart(part openaisdk.StreamPart) (StreamPart, bool) {
	switch p := part.(type) {
	case openaisdk.StreamTextStart:
		return StreamTextStart{}, true
	case openaisdk.StreamTextDelta:
		return StreamTextDelta{Text: p.Text}, true
	case openaisdk.StreamTextEnd:
		return StreamTextEnd{}, true
	case openaisdk.StreamReasoningStart:
		return StreamReasoningStart{}, true
	case openaisdk.StreamReasoningDelta:
		return StreamReasoningDelta{Text: p.Text}, true
	case openaisdk.StreamReasoningEnd:
		return StreamReasoningEnd{}, true
	case openaisdk.StreamToolInputStart:
		return StreamToolCallStart{ID: p.ID, Name: p.ToolName}, true
	case openaisdk.StreamToolInputDelta:
		return StreamToolInputDelta{ID: p.ID, JSON: p.Delta}, true
	case openaisdk.StreamToolInputEnd:
		return StreamToolInputEnd{ID: p.ID}, true
	case openaisdk.StreamToolCall:
		return StreamToolInputEnd{ID: p.ToolCallID, Input: cloneRawMessage(p.Input)}, true
	case openaisdk.StreamFinish:
		return StreamFinish{FinishReason: fromUnifiedFinishReason(p.FinishReason.Unified), Usage: Usage{InputTokens: derefInt(p.Usage.InputTokens.Total), OutputTokens: derefInt(p.Usage.OutputTokens.Total)}}, true
	case openaisdk.StreamError:
		return StreamError{Err: p.Err}, true
	case openaisdk.StreamRaw:
		return StreamRaw{Bytes: p.Raw}, true
	default:
		return nil, false
	}
}

func fromOpenAIResult(result *openaisdk.GenerateResult) *GenerateResult {
	if result == nil {
		return nil
	}
	return &GenerateResult{
		Content:      fromOpenAIContent(result.Content),
		FinishReason: fromUnifiedFinishReason(result.FinishReason.Unified),
		Usage:        Usage{InputTokens: derefInt(result.Usage.InputTokens.Total), OutputTokens: derefInt(result.Usage.OutputTokens.Total)},
	}
}

// fromOpenAIContent converts openai response content. The openai package emits
// its own ToolCallContent (embedding ToolCallContentEmbed) and ReasoningContent;
// text content is the shared openaicompatible.TextContent alias.
func fromOpenAIContent(contents []openaisdk.Content) []Content {
	converted := make([]Content, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case openaisdk.TextContent:
			converted = append(converted, TextContent{Text: c.Text})
		case openaisdk.ToolCallContent:
			converted = append(converted, ToolUseContent{ID: c.ToolCallID, Name: c.ToolName, Input: cloneRawMessage(c.Input)})
		case openaisdk.ReasoningContent:
			converted = append(converted, ReasoningContent{Text: c.Text})
		}
	}
	return converted
}
