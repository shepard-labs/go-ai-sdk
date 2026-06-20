package adapters

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	openroutersdk "github.com/shepard-labs/go-ai-sdk/openrouter"
)

// OpenRouterAdapter adapts an OpenRouter language model to the Client interface.
type OpenRouterAdapter struct {
	model openroutersdk.LanguageModel
}

// OpenRouterSettings configures an OpenRouter Client.
type OpenRouterSettings struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// NewOpenRouterAdapter wraps an existing OpenRouter language model as a Client.
func NewOpenRouterAdapter(model openroutersdk.LanguageModel) Client {
	return &OpenRouterAdapter{model: model}
}

// NewOpenRouterClient creates an OpenRouter-backed Client from an API key and model ID.
func NewOpenRouterClient(apiKey string, modelID string) (Client, error) {
	return NewOpenRouterClientWithSettings(OpenRouterSettings{APIKey: apiKey}, modelID)
}

// NewOpenRouterClientWithSettings creates an OpenRouter-backed Client honoring
// BaseURL and a custom HTTP client.
func NewOpenRouterClientWithSettings(settings OpenRouterSettings, modelID string) (Client, error) {
	providerSettings := openroutersdk.ProviderSettings{APIKey: settings.APIKey, BaseURL: settings.BaseURL}
	if settings.HTTPClient != nil {
		providerSettings.Fetch = settings.HTTPClient
	}
	provider := openroutersdk.CreateOpenRouter(providerSettings)
	if err := provider.Err(); err != nil {
		return nil, err
	}
	return NewOpenRouterAdapter(provider.LanguageModel(modelID)), nil
}

// Generate sends a completion request through the OpenRouter SDK.
func (a *OpenRouterAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, err := toOpenRouterOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	return fromOpenRouterResult(result), nil
}

// Stream sends a streaming completion request through the OpenRouter SDK and
// maps provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *OpenRouterAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, err := toOpenRouterOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoStream(ctx, openroutersdk.StreamOptions{GenerateOptions: sdkOpts})
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
				if mapped, emit := mapOpenRouterStreamPart(part); emit {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapOpenRouterStreamPart converts an openroutersdk.StreamPart to a neutral
// StreamPart. Returns (zero, false) for parts with no neutral equivalent
// (StreamResponseMetadata, StreamFile, StreamSource). spec §1.1
func mapOpenRouterStreamPart(part openroutersdk.StreamPart) (StreamPart, bool) {
	switch p := part.(type) {
	case openroutersdk.StreamTextStart:
		return StreamTextStart{}, true
	case openroutersdk.StreamTextDelta:
		return StreamTextDelta{Text: p.Delta}, true
	case openroutersdk.StreamTextEnd:
		return StreamTextEnd{}, true
	case openroutersdk.StreamReasoningStart:
		return StreamReasoningStart{}, true
	case openroutersdk.StreamReasoningDelta:
		return StreamReasoningDelta{Text: p.Delta}, true
	case openroutersdk.StreamReasoningEnd:
		return StreamReasoningEnd{}, true
	case openroutersdk.StreamToolInputStart:
		return StreamToolCallStart{ID: p.ID, Name: p.ToolName}, true
	case openroutersdk.StreamToolInputDelta:
		return StreamToolInputDelta{ID: p.ID, JSON: p.Delta}, true
	case openroutersdk.StreamToolInputEnd:
		return StreamToolInputEnd{ID: p.ID}, true
	case openroutersdk.StreamToolCall:
		return StreamToolInputEnd{ID: p.ToolCallID, Input: openRouterInputToRaw(p.Input)}, true
	case openroutersdk.StreamFinish:
		return StreamFinish{FinishReason: fromOpenRouterFinishReason(p.FinishReason), Usage: Usage{InputTokens: p.Usage.InputTokens, OutputTokens: p.Usage.OutputTokens}}, true
	case openroutersdk.StreamError:
		return StreamError{Err: p.Err}, true
	case openroutersdk.StreamRaw:
		return StreamRaw{}, true
	default:
		return nil, false
	}
}

func toOpenRouterOptions(opts GenerateOptions) (openroutersdk.GenerateOptions, error) {
	messages := make([]openroutersdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, openroutersdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		converted, err := toOpenRouterMessages(message)
		if err != nil {
			return openroutersdk.GenerateOptions{}, err
		}
		messages = append(messages, converted...)
	}
	tools, err := toOpenRouterTools(opts.Tools)
	if err != nil {
		return openroutersdk.GenerateOptions{}, err
	}
	sdkOpts := openroutersdk.GenerateOptions{Messages: messages, Tools: tools}
	if opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		sdkOpts.MaxTokens = &maxTokens
	}
	return sdkOpts, nil
}

func toOpenRouterMessages(message Message) ([]openroutersdk.Message, error) {
	switch message.Role {
	case "user":
		var userContent []openroutersdk.UserContent
		var toolContent []openroutersdk.ToolContent
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				userContent = append(userContent, openroutersdk.TextContent{Text: c.Text})
			case ToolResultContent:
				toolContent = append(toolContent, openroutersdk.ToolResultContent{
					ToolCallID: c.ToolUseID,
					Output:     c.Text,
					IsError:    c.IsError,
				})
			default:
				return nil, fmt.Errorf("llm: unsupported user content %T", content)
			}
		}
		var out []openroutersdk.Message
		if len(userContent) > 0 {
			out = append(out, openroutersdk.UserMessage{Content: userContent})
		}
		if len(toolContent) > 0 {
			out = append(out, openroutersdk.ToolMessage{Content: toolContent})
		}
		return out, nil
	case "assistant":
		contents := make([]openroutersdk.AssistantContent, 0, len(message.Content))
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				contents = append(contents, openroutersdk.TextContent{Text: c.Text})
			case ToolUseContent:
				contents = append(contents, openroutersdk.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: openRouterInput(c.Input)})
			default:
				return nil, fmt.Errorf("llm: unsupported assistant content %T", content)
			}
		}
		return []openroutersdk.Message{openroutersdk.AssistantMessage{Content: contents}}, nil
	default:
		return nil, fmt.Errorf("llm: unknown message role %q", message.Role)
	}
}

func toOpenRouterTools(tools []Tool) ([]openroutersdk.Tool, error) {
	sdkTools := make([]openroutersdk.Tool, len(tools))
	for i, tool := range tools {
		schema, err := decodeSchema(tool.InputSchema)
		if err != nil {
			return nil, err
		}
		sdkTools[i] = openroutersdk.Tool{Type: "function", Name: tool.Name, Description: tool.Description, InputSchema: schema}
	}
	return sdkTools, nil
}

func fromOpenRouterResult(result *openroutersdk.GenerateResult) *GenerateResult {
	if result == nil {
		return nil
	}
	return &GenerateResult{
		Content:      fromOpenRouterContent(result.Content),
		FinishReason: fromOpenRouterFinishReason(result.FinishReason),
		Usage:        Usage{InputTokens: result.Usage.InputTokens, OutputTokens: result.Usage.OutputTokens},
	}
}

func fromOpenRouterContent(contents []openroutersdk.Content) []Content {
	converted := make([]Content, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case openroutersdk.TextContent:
			converted = append(converted, TextContent{Text: c.Text})
		case openroutersdk.ToolCallContent:
			converted = append(converted, ToolUseContent{ID: c.ToolCallID, Name: c.ToolName, Input: openRouterInputToRaw(c.Input)})
		}
	}
	return converted
}

func fromOpenRouterFinishReason(reason openroutersdk.FinishReason) FinishReason {
	switch reason {
	case openroutersdk.FinishReasonStop:
		return FinishReasonStop
	case openroutersdk.FinishReasonToolCalls:
		return FinishReasonToolCalls
	case openroutersdk.FinishReasonLength:
		return FinishReasonLength
	default:
		return FinishReasonError
	}
}

// openRouterInput converts a neutral json.RawMessage tool input to the value
// OpenRouter expects (its ToolCallContent.Input is any). A decoded value is
// passed through so it serializes as a JSON object rather than a string.
func openRouterInput(input json.RawMessage) any {
	if len(input) == 0 {
		return nil
	}
	var decoded any
	if err := json.Unmarshal(input, &decoded); err != nil {
		return string(input)
	}
	return decoded
}

// openRouterInputToRaw normalizes an OpenRouter tool-call input (any) back to a
// neutral json.RawMessage.
func openRouterInputToRaw(input any) json.RawMessage {
	if input == nil {
		return nil
	}
	if raw, ok := input.(json.RawMessage); ok {
		return cloneRawMessage(raw)
	}
	if s, ok := input.(string); ok {
		return json.RawMessage(s)
	}
	data, err := json.Marshal(input)
	if err != nil {
		return nil
	}
	return data
}
