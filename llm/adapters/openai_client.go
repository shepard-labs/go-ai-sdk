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

// Capabilities reports the feature set supported by the OpenAI adapter.
func (a *OpenAIAdapter) Capabilities() Capabilities {
	return Capabilities{
		Provider:           "openai",
		Streaming:          true,
		ToolCalling:        true,
		ToolChoiceAuto:     true,
		ToolChoiceNone:     true,
		ToolChoiceRequired: true,
		ToolChoiceTool:     true,
		StructuredOutput:   true,
		JSONMode:           true,
		Images:             true,
		Reasoning:          true,
		ParallelToolCalls:  true,
		PromptCaching:      false,
	}
}

// Generate sends a completion request through the OpenAI SDK.
func (a *OpenAIAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, warnings, err := toOpenAICompatibleOptions(opts, true)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	out := fromOpenAIResult(result)
	if out != nil && len(warnings) > 0 {
		out.Warnings = append(warnings, out.Warnings...)
	}
	return out, nil
}

// Stream sends a streaming completion request through the OpenAI SDK and maps
// provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *OpenAIAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, _, err := toOpenAICompatibleOptions(opts, true)
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
				for _, mapped := range mapOpenAIStreamPart(part) {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapOpenAIStreamPart converts an openaisdk.StreamPart to zero or more neutral
// StreamParts. Returns nil for parts with no neutral equivalent
// (StreamToolApprovalRequest, StreamCompactionEnd, SourceContent,
// StreamCustomPart). spec §1.1
func mapOpenAIStreamPart(part openaisdk.StreamPart) []StreamPart {
	switch p := part.(type) {
	case openaisdk.StreamStart:
		return openAICompatibleStartWarnings(p.Warnings)
	case openaisdk.StreamResponseMetadata:
		return []StreamPart{openAICompatibleMetadata(p)}
	case openaisdk.StreamTextStart:
		return []StreamPart{StreamTextStart{}}
	case openaisdk.StreamTextDelta:
		return []StreamPart{StreamTextDelta{Text: p.Text}}
	case openaisdk.StreamTextEnd:
		return []StreamPart{StreamTextEnd{}}
	case openaisdk.StreamReasoningStart:
		return []StreamPart{StreamReasoningStart{}}
	case openaisdk.StreamReasoningDelta:
		return []StreamPart{StreamReasoningDelta{Text: p.Text}}
	case openaisdk.StreamReasoningEnd:
		return []StreamPart{StreamReasoningEnd{}}
	case openaisdk.StreamToolInputStart:
		return []StreamPart{StreamToolCallStart{ID: p.ID, Name: p.ToolName}}
	case openaisdk.StreamToolInputDelta:
		return []StreamPart{StreamToolInputDelta{ID: p.ID, JSON: p.Delta}}
	case openaisdk.StreamToolInputEnd:
		return []StreamPart{StreamToolInputEnd{ID: p.ID}}
	case openaisdk.StreamToolCall:
		return []StreamPart{StreamToolInputEnd{ID: p.ToolCallID, Input: cloneRawMessage(p.Input)}}
	case openaisdk.StreamFinish:
		return []StreamPart{StreamFinish{
			FinishReason: fromUnifiedFinishReason(p.FinishReason.Unified),
			Usage: Usage{
				InputTokens:     derefInt(p.Usage.InputTokens.Total),
				OutputTokens:    derefInt(p.Usage.OutputTokens.Total),
				ReasoningTokens: derefInt(p.Usage.OutputTokens.Reasoning),
			},
			ProviderMetadata: openAICompatibleProviderMetadata("openai", p.ProviderMetadata),
		}}
	case openaisdk.StreamError:
		return []StreamPart{StreamError{Err: p.Err}}
	case openaisdk.StreamRaw:
		return []StreamPart{StreamRaw{Bytes: p.Raw}}
	default:
		return nil
	}
}

func fromOpenAIResult(result *openaisdk.GenerateResult) *GenerateResult {
	if result == nil {
		return nil
	}
	out := &GenerateResult{
		Content:      fromOpenAIContent(result.Content),
		FinishReason: fromUnifiedFinishReason(result.FinishReason.Unified),
		Usage: Usage{
			InputTokens:     derefInt(result.Usage.InputTokens.Total),
			OutputTokens:    derefInt(result.Usage.OutputTokens.Total),
			ReasoningTokens: derefInt(result.Usage.OutputTokens.Reasoning),
		},
	}
	out.Response = ResponseMetadata{
		ID:      result.Response.ID,
		ModelID: result.Response.ModelID,
		Headers: flattenHeader(result.Response.Headers),
	}
	if result.Response.Timestamp != nil {
		out.Response.Timestamp = *result.Response.Timestamp
	}
	for _, w := range result.Warnings {
		out.Warnings = append(out.Warnings, Warning{Code: w.Type, Message: w.Message, Provider: "openai"})
	}
	if result.ProviderMetadata != nil {
		pm := make(ProviderMetadata)
		pm["openai"] = result.ProviderMetadata
		out.ProviderMetadata = pm
	}
	return out
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
