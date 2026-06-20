package adapters

import (
	"context"
	"fmt"
	"net/http"

	. "github.com/shepard-labs/go-ai-sdk/llm"
	openaicompatiblesdk "github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// OpenAICompatibleAdapter adapts an OpenAI-compatible language model to the
// Client interface. It is used for self-hosted and OpenAI-API-compatible
// endpoints such as Ollama and LM Studio via a custom BaseURL.
type OpenAICompatibleAdapter struct {
	model openaicompatiblesdk.LanguageModel
}

// OpenAICompatibleSettings configures an OpenAI-compatible Client. Name and
// BaseURL are required by the underlying provider.
type OpenAICompatibleSettings struct {
	Name       string
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

// NewOpenAICompatibleAdapter wraps an existing OpenAI-compatible model as a Client.
func NewOpenAICompatibleAdapter(model openaicompatiblesdk.LanguageModel) Client {
	return &OpenAICompatibleAdapter{model: model}
}

// NewOpenAICompatibleClient creates an OpenAI-compatible Client from settings and a model ID.
func NewOpenAICompatibleClient(settings OpenAICompatibleSettings, modelID string) (Client, error) {
	providerSettings := openaicompatiblesdk.ProviderSettings{
		Name:    settings.Name,
		BaseURL: settings.BaseURL,
		APIKey:  settings.APIKey,
	}
	if settings.HTTPClient != nil {
		providerSettings.Fetch = settings.HTTPClient
	}
	provider := openaicompatiblesdk.CreateOpenAICompatible(providerSettings)
	if err := provider.Err(); err != nil {
		return nil, err
	}
	return NewOpenAICompatibleAdapter(provider.LanguageModel(modelID)), nil
}

// Generate sends a completion request through the OpenAI-compatible SDK.
func (a *OpenAICompatibleAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, err := toOpenAICompatibleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	return fromOpenAICompatibleResult(result), nil
}

// Stream sends a streaming completion request through the OpenAI-compatible
// SDK and maps provider-native StreamPart values into the neutral StreamPart
// union. spec §1.1
func (a *OpenAICompatibleAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, err := toOpenAICompatibleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoStream(ctx, openaicompatiblesdk.StreamOptions{GenerateOptions: sdkOpts})
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
				if mapped, emit := mapOpenAICompatibleStreamPart(part); emit {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapOpenAICompatibleStreamPart converts an openaicompatiblesdk.StreamPart to a
// neutral StreamPart. Returns (zero, false) for parts with no neutral
// equivalent (StreamResponseMetadata, StreamCustomPart). spec §1.1
func mapOpenAICompatibleStreamPart(part openaicompatiblesdk.StreamPart) (StreamPart, bool) {
	switch p := part.(type) {
	case openaicompatiblesdk.StreamTextStart:
		return StreamTextStart{}, true
	case openaicompatiblesdk.StreamTextDelta:
		return StreamTextDelta{Text: p.Text}, true
	case openaicompatiblesdk.StreamTextEnd:
		return StreamTextEnd{}, true
	case openaicompatiblesdk.StreamReasoningStart:
		return StreamReasoningStart{}, true
	case openaicompatiblesdk.StreamReasoningDelta:
		return StreamReasoningDelta{Text: p.Text}, true
	case openaicompatiblesdk.StreamReasoningEnd:
		return StreamReasoningEnd{}, true
	case openaicompatiblesdk.StreamToolInputStart:
		return StreamToolCallStart{ID: p.ID, Name: p.ToolName}, true
	case openaicompatiblesdk.StreamToolInputDelta:
		return StreamToolInputDelta{ID: p.ID, JSON: p.Delta}, true
	case openaicompatiblesdk.StreamToolInputEnd:
		return StreamToolInputEnd{ID: p.ID}, true
	case openaicompatiblesdk.StreamToolCall:
		return StreamToolInputEnd{ID: p.ToolCallID, Input: cloneRawMessage(p.Input)}, true
	case openaicompatiblesdk.StreamFinish:
		return StreamFinish{FinishReason: fromUnifiedFinishReason(p.FinishReason.Unified), Usage: Usage{InputTokens: derefInt(p.Usage.InputTokens.Total), OutputTokens: derefInt(p.Usage.OutputTokens.Total)}}, true
	case openaicompatiblesdk.StreamError:
		return StreamError{Err: p.Err}, true
	case openaicompatiblesdk.StreamRaw:
		return StreamRaw{Bytes: p.Raw}, true
	default:
		return nil, false
	}
}

func toOpenAICompatibleOptions(opts GenerateOptions) (openaicompatiblesdk.GenerateOptions, error) {
	messages := make([]openaicompatiblesdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, openaicompatiblesdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		converted, err := toOpenAICompatibleMessages(message)
		if err != nil {
			return openaicompatiblesdk.GenerateOptions{}, err
		}
		messages = append(messages, converted...)
	}
	tools, err := toOpenAICompatibleTools(opts.Tools)
	if err != nil {
		return openaicompatiblesdk.GenerateOptions{}, err
	}
	sdkOpts := openaicompatiblesdk.GenerateOptions{Messages: messages, Tools: tools}
	if opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		sdkOpts.MaxOutputTokens = &maxTokens
	}
	return sdkOpts, nil
}

// toOpenAICompatibleMessages converts one neutral message. A "user" message
// carrying tool results is split into a UserMessage (text) and a ToolMessage
// (tool results), as the OpenAI wire format requires.
func toOpenAICompatibleMessages(message Message) ([]openaicompatiblesdk.Message, error) {
	switch message.Role {
	case "user":
		var userContent []openaicompatiblesdk.UserContent
		var toolContent []openaicompatiblesdk.ToolContent
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				userContent = append(userContent, openaicompatiblesdk.TextContent{Text: c.Text})
			case ToolResultContent:
				toolContent = append(toolContent, openaicompatiblesdk.ToolResultContent{
					ToolCallID: c.ToolUseID,
					Output:     openaicompatiblesdk.ToolResultOutput{Type: toolResultOutputType(c.IsError), Value: c.Text},
				})
			case ImageContent:
				if fc, ok := openAICompatibleImageContent(c); ok {
					userContent = append(userContent, fc)
				}
			default:
				return nil, fmt.Errorf("llm: unsupported user content %T", content)
			}
		}
		var out []openaicompatiblesdk.Message
		if len(userContent) > 0 {
			out = append(out, openaicompatiblesdk.UserMessage{Content: userContent})
		}
		if len(toolContent) > 0 {
			out = append(out, openaicompatiblesdk.ToolMessage{Content: toolContent})
		}
		return out, nil
	case "assistant":
		contents := make([]openaicompatiblesdk.AssistantContent, 0, len(message.Content))
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				contents = append(contents, openaicompatiblesdk.TextContent{Text: c.Text})
			case ToolUseContent:
				contents = append(contents, openaicompatiblesdk.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: cloneRawMessage(c.Input)})
			case ReasoningContent:
				contents = append(contents, openaicompatiblesdk.ReasoningContent{Text: c.Text})
			default:
				return nil, fmt.Errorf("llm: unsupported assistant content %T", content)
			}
		}
		return []openaicompatiblesdk.Message{openaicompatiblesdk.AssistantMessage{Content: contents}}, nil
	default:
		return nil, fmt.Errorf("llm: unknown message role %q", message.Role)
	}
}

// openAICompatibleImageContent converts a neutral ImageContent to the
// openaicompatible FileContent (inline bytes) form. URL-source images are
// dropped with a best-effort text fallback per spec §1.2 (the openaicompatible
// wire format does not carry a bare image URL part in the spike). Returns
// (zero, false) when the image cannot be represented.
func openAICompatibleImageContent(c ImageContent) (openaicompatiblesdk.UserContent, bool) {
	if src, ok := c.Source.(ImageInlineSource); ok {
		return openaicompatiblesdk.FileContent{Data: src.Data, MediaType: c.MIME}, true
	}
	return nil, false
}

func toOpenAICompatibleTools(tools []Tool) ([]openaicompatiblesdk.Tool, error) {
	sdkTools := make([]openaicompatiblesdk.Tool, len(tools))
	for i, tool := range tools {
		schema, err := decodeSchema(tool.InputSchema)
		if err != nil {
			return nil, err
		}
		sdkTools[i] = openaicompatiblesdk.Tool{Type: "function", Name: tool.Name, Description: tool.Description, InputSchema: schema}
	}
	return sdkTools, nil
}

func fromOpenAICompatibleResult(result *openaicompatiblesdk.GenerateResult) *GenerateResult {
	if result == nil {
		return nil
	}
	return &GenerateResult{
		Content:      fromOpenAICompatibleContent(result.Content),
		FinishReason: fromUnifiedFinishReason(result.FinishReason.Unified),
		Usage:        Usage{InputTokens: derefInt(result.Usage.InputTokens.Total), OutputTokens: derefInt(result.Usage.OutputTokens.Total)},
	}
}

func fromOpenAICompatibleContent(contents []openaicompatiblesdk.Content) []Content {
	converted := make([]Content, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case openaicompatiblesdk.TextContent:
			converted = append(converted, TextContent{Text: c.Text})
		case openaicompatiblesdk.ToolCallContent:
			converted = append(converted, ToolUseContent{ID: c.ToolCallID, Name: c.ToolName, Input: cloneRawMessage(c.Input)})
		case openaicompatiblesdk.ReasoningContent:
			converted = append(converted, ReasoningContent{Text: c.Text})
		}
	}
	return converted
}
