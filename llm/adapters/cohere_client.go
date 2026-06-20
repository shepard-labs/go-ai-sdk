package adapters

import (
	"context"
	"fmt"
	"net/http"

	coheresdk "github.com/shepard-labs/go-ai-sdk/cohere"
)

// CohereAdapter adapts a Cohere language model to the Client interface.
type CohereAdapter struct {
	streamNotSupported
	model coheresdk.LanguageModel
}

// CohereSettings configures a Cohere Client.
type CohereSettings struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// NewCohereAdapter wraps an existing Cohere language model as a Client.
func NewCohereAdapter(model coheresdk.LanguageModel) Client {
	return &CohereAdapter{model: model}
}

// NewCohereClient creates a Cohere-backed Client from an API key and model ID.
func NewCohereClient(apiKey string, modelID string) (Client, error) {
	return NewCohereClientWithSettings(CohereSettings{APIKey: apiKey}, modelID)
}

// NewCohereClientWithSettings creates a Cohere-backed Client honoring BaseURL
// and a custom HTTP client.
func NewCohereClientWithSettings(settings CohereSettings, modelID string) (Client, error) {
	providerSettings := coheresdk.ProviderSettings{APIKey: settings.APIKey, BaseURL: settings.BaseURL}
	if settings.HTTPClient != nil {
		providerSettings.Fetch = settings.HTTPClient
	}
	provider := coheresdk.CreateCohere(providerSettings)
	if err := provider.Err(); err != nil {
		return nil, err
	}
	return NewCohereAdapter(provider.LanguageModel(modelID)), nil
}

// Generate sends a completion request through the Cohere SDK.
func (a *CohereAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, err := toCohereOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	return fromCohereResult(result), nil
}

func toCohereOptions(opts GenerateOptions) (coheresdk.GenerateOptions, error) {
	messages := make([]coheresdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, coheresdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		converted, err := toCohereMessages(message)
		if err != nil {
			return coheresdk.GenerateOptions{}, err
		}
		messages = append(messages, converted...)
	}
	tools, err := toCohereTools(opts.Tools)
	if err != nil {
		return coheresdk.GenerateOptions{}, err
	}
	sdkOpts := coheresdk.GenerateOptions{Messages: messages, Tools: tools}
	if opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		sdkOpts.MaxOutputTokens = &maxTokens
	}
	return sdkOpts, nil
}

func toCohereMessages(message Message) ([]coheresdk.Message, error) {
	switch message.Role {
	case "user":
		var userContent []coheresdk.UserContent
		var toolContent []coheresdk.ToolContent
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				userContent = append(userContent, coheresdk.TextContent{Text: c.Text})
			case ToolResultContent:
				toolContent = append(toolContent, coheresdk.ToolResultContent{
					ToolCallID: c.ToolUseID,
					Output:     coheresdk.ToolResultOutput{Type: toolResultOutputType(c.IsError), Value: c.Text},
				})
			default:
				return nil, fmt.Errorf("llm: unsupported user content %T", content)
			}
		}
		var out []coheresdk.Message
		if len(userContent) > 0 {
			out = append(out, coheresdk.UserMessage{Content: userContent})
		}
		if len(toolContent) > 0 {
			out = append(out, coheresdk.ToolMessage{Content: toolContent})
		}
		return out, nil
	case "assistant":
		contents := make([]coheresdk.AssistantContent, 0, len(message.Content))
		for _, content := range message.Content {
			switch c := content.(type) {
			case TextContent:
				contents = append(contents, coheresdk.TextContent{Text: c.Text})
			case ToolUseContent:
				contents = append(contents, coheresdk.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: cloneRawMessage(c.Input)})
			default:
				return nil, fmt.Errorf("llm: unsupported assistant content %T", content)
			}
		}
		return []coheresdk.Message{coheresdk.AssistantMessage{Content: contents}}, nil
	default:
		return nil, fmt.Errorf("llm: unknown message role %q", message.Role)
	}
}

func toCohereTools(tools []Tool) ([]coheresdk.Tool, error) {
	sdkTools := make([]coheresdk.Tool, len(tools))
	for i, tool := range tools {
		schema, err := decodeSchema(tool.InputSchema)
		if err != nil {
			return nil, err
		}
		sdkTools[i] = coheresdk.Tool{Type: "function", Name: tool.Name, Description: tool.Description, InputSchema: schema}
	}
	return sdkTools, nil
}

func fromCohereResult(result *coheresdk.GenerateResult) *GenerateResult {
	if result == nil {
		return nil
	}
	return &GenerateResult{
		Content:      fromCohereContent(result.Content),
		FinishReason: fromUnifiedFinishReason(result.FinishReason.Unified),
		Usage:        Usage{InputTokens: derefInt(result.Usage.InputTokens.Total), OutputTokens: derefInt(result.Usage.OutputTokens.Total)},
	}
}

func fromCohereContent(contents []coheresdk.Content) []Content {
	converted := make([]Content, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case coheresdk.TextContent:
			converted = append(converted, TextContent{Text: c.Text})
		case coheresdk.ToolCallContent:
			converted = append(converted, ToolUseContent{ID: c.ToolCallID, Name: c.ToolName, Input: cloneRawMessage(c.Input)})
		}
	}
	return converted
}
