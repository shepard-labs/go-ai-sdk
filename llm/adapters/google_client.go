package adapters

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"

	googlesdk "github.com/shepard-labs/go-ai-sdk/google"
)

// GoogleAdapter adapts a Google Gemini language model to the Client interface.
type GoogleAdapter struct {
	model googlesdk.LanguageModel
}

// GoogleSettings configures a Google Client.
type GoogleSettings struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// NewGoogleAdapter wraps an existing Google language model as a Client.
func NewGoogleAdapter(model googlesdk.LanguageModel) Client {
	return &GoogleAdapter{model: model}
}

// NewGoogleClient creates a Google-backed Client from an API key and model ID.
func NewGoogleClient(apiKey string, modelID string) (Client, error) {
	return NewGoogleClientWithSettings(GoogleSettings{APIKey: apiKey}, modelID)
}

// NewGoogleClientWithSettings creates a Google-backed Client honoring BaseURL
// and a custom HTTP client.
func NewGoogleClientWithSettings(settings GoogleSettings, modelID string) (Client, error) {
	providerSettings := googlesdk.ProviderSettings{APIKey: settings.APIKey, BaseURL: settings.BaseURL}
	if settings.HTTPClient != nil {
		providerSettings.Fetch = settings.HTTPClient
	}
	provider := googlesdk.CreateGoogle(providerSettings)
	if err := provider.Err(); err != nil {
		return nil, err
	}
	return NewGoogleAdapter(provider.LanguageModel(modelID)), nil
}

// Generate sends a completion request through the Google SDK.
func (a *GoogleAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, err := toGoogleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	return fromGoogleResult(result), nil
}

// Stream sends a streaming completion request through the Google SDK and maps
// provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *GoogleAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, err := toGoogleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoStream(ctx, googlesdk.StreamOptions{GenerateOptions: sdkOpts})
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
				if mapped, emit := mapGoogleStreamPart(part); emit {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapGoogleStreamPart converts a googlesdk.StreamPart to a neutral StreamPart.
// Returns (zero, false) for parts with no neutral equivalent (StreamStart,
// StreamResponseMetadata, StreamSource, StreamFile, StreamReasoningFile,
// StreamToolResult). spec §1.1
func mapGoogleStreamPart(part googlesdk.StreamPart) (StreamPart, bool) {
	switch p := part.(type) {
	case googlesdk.StreamTextStart:
		return StreamTextStart{}, true
	case googlesdk.StreamTextDelta:
		return StreamTextDelta{Text: p.Text}, true
	case googlesdk.StreamTextEnd:
		return StreamTextEnd{}, true
	case googlesdk.StreamReasoningStart:
		return StreamReasoningStart{}, true
	case googlesdk.StreamReasoningDelta:
		return StreamReasoningDelta{Text: p.Text}, true
	case googlesdk.StreamReasoningEnd:
		return StreamReasoningEnd{}, true
	case googlesdk.StreamToolInputStart:
		return StreamToolCallStart{ID: p.ID, Name: p.ToolName}, true
	case googlesdk.StreamToolInputDelta:
		return StreamToolInputDelta{ID: p.ID, JSON: p.Delta}, true
	case googlesdk.StreamToolInputEnd:
		return StreamToolInputEnd{ID: p.ID}, true
	case googlesdk.StreamToolCall:
		return StreamToolInputEnd{ID: p.ToolCall.ToolCallID, Input: cloneRawMessage(p.ToolCall.Input)}, true
	case googlesdk.StreamFinish:
		return StreamFinish{FinishReason: fromUnifiedFinishReason(p.FinishReason.Unified), Usage: Usage{InputTokens: derefInt(p.Usage.InputTokens.Total), OutputTokens: derefInt(p.Usage.OutputTokens.Total)}}, true
	case googlesdk.StreamError:
		return StreamError{Err: p.Err}, true
	case googlesdk.StreamRaw:
		return StreamRaw{Bytes: p.Raw}, true
	default:
		return nil, false
	}
}

func toGoogleOptions(opts GenerateOptions) (googlesdk.GenerateOptions, error) {
	messages := make([]googlesdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, googlesdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		converted, err := toGoogleMessages(message)
		if err != nil {
			return googlesdk.GenerateOptions{}, err
		}
		messages = append(messages, converted...)
	}
	tools, err := toGoogleTools(opts.Tools)
	if err != nil {
		return googlesdk.GenerateOptions{}, err
	}
	sdkOpts := googlesdk.GenerateOptions{Messages: messages, Tools: tools}
	if opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		sdkOpts.MaxOutputTokens = &maxTokens
	}
	return sdkOpts, nil
}

func toGoogleMessages(message Message) ([]googlesdk.Message, error) {
	switch message.Role {
	case "user":
		var userContent []googlesdk.UserContent
		var toolContent []googlesdk.ToolContent
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				userContent = append(userContent, googlesdk.TextContent{Text: c.Text})
			case ToolResultContent:
				toolContent = append(toolContent, googlesdk.ToolResultContent{
					ToolCallID: c.ToolUseID,
					Output:     googlesdk.ToolResultOutput{Type: toolResultOutputType(c.IsError), Value: c.Text},
				})
			case ImageContent:
				if ic, ok := googleImageContent(c); ok {
					userContent = append(userContent, ic)
				}
			default:
				return nil, fmt.Errorf("llm: unsupported user content %T", content)
			}
		}
		var out []googlesdk.Message
		if len(userContent) > 0 {
			out = append(out, googlesdk.UserMessage{Content: userContent})
		}
		if len(toolContent) > 0 {
			out = append(out, googlesdk.ToolMessage{Content: toolContent})
		}
		return out, nil
	case "assistant":
		contents := make([]googlesdk.AssistantContent, 0, len(message.Content))
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				contents = append(contents, googlesdk.TextContent{Text: c.Text})
			case ToolUseContent:
				contents = append(contents, googlesdk.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: cloneRawMessage(c.Input)})
			case ReasoningContent:
				contents = append(contents, googlesdk.ReasoningContent{Text: c.Text})
			default:
				return nil, fmt.Errorf("llm: unsupported assistant content %T", content)
			}
		}
		return []googlesdk.Message{googlesdk.AssistantMessage{Content: contents}}, nil
	default:
		return nil, fmt.Errorf("llm: unknown message role %q", message.Role)
	}
}

// googleImageContent converts a neutral ImageContent to a googlesdk.ImageContent.
// Inline bytes use the "data" source type (base64); URL sources use "url".
// spec §1.2
func googleImageContent(c ImageContent) (googlesdk.UserContent, bool) {
	switch src := c.Source.(type) {
	case ImageURLSource:
		return googlesdk.ImageContent{Source: googlesdk.ImageSource{Type: "url", MediaType: c.MIME, URL: src.URL}}, true
	case ImageInlineSource:
		return googlesdk.ImageContent{Source: googlesdk.ImageSource{Type: "data", MediaType: c.MIME, Data: base64.StdEncoding.EncodeToString(src.Data)}}, true
	}
	return nil, false
}

func toGoogleTools(tools []Tool) ([]googlesdk.Tool, error) {
	sdkTools := make([]googlesdk.Tool, len(tools))
	for i, tool := range tools {
		schema, err := decodeSchema(tool.InputSchema)
		if err != nil {
			return nil, err
		}
		sdkTools[i] = googlesdk.Tool{Type: "function", Name: tool.Name, Description: tool.Description, InputSchema: schema}
	}
	return sdkTools, nil
}

func fromGoogleResult(result *googlesdk.GenerateResult) *GenerateResult {
	if result == nil {
		return nil
	}
	return &GenerateResult{
		Content:      fromGoogleContent(result.Content),
		FinishReason: fromUnifiedFinishReason(result.FinishReason.Unified),
		Usage:        Usage{InputTokens: derefInt(result.Usage.InputTokens.Total), OutputTokens: derefInt(result.Usage.OutputTokens.Total)},
	}
}

func fromGoogleContent(contents []googlesdk.Content) []Content {
	converted := make([]Content, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case googlesdk.TextContent:
			converted = append(converted, TextContent{Text: c.Text})
		case googlesdk.ToolCallContent:
			converted = append(converted, ToolUseContent{ID: c.ToolCallID, Name: c.ToolName, Input: cloneRawMessage(c.Input)})
		case googlesdk.ReasoningContent:
			converted = append(converted, ReasoningContent{Text: c.Text})
		}
	}
	return converted
}
