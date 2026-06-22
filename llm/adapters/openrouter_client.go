package adapters

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/shepard-labs/go-ai-sdk/llm"
	openroutersdk "github.com/shepard-labs/go-ai-sdk/openrouter"
)

// OpenRouterAdapter adapts an OpenRouter language model to the Client interface.
type OpenRouterAdapter struct {
	model          openroutersdk.LanguageModel
	defaultModelID string
	newModel       func(modelID string) openroutersdk.LanguageModel
}

// OpenRouterSettings configures an OpenRouter Client.
type OpenRouterSettings struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// NewOpenRouterAdapter wraps an existing OpenRouter language model as a Client.
func NewOpenRouterAdapter(model openroutersdk.LanguageModel) Client {
	return &OpenRouterAdapter{model: model, defaultModelID: model.ModelID()}
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
	return &OpenRouterAdapter{
		model:          provider.LanguageModel(modelID),
		defaultModelID: modelID,
		newModel: func(id string) openroutersdk.LanguageModel {
			return provider.LanguageModel(id)
		},
	}, nil
}

// Capabilities reports the feature set supported by the OpenRouter adapter.
func (a *OpenRouterAdapter) Capabilities() Capabilities {
	return a.CapabilitiesForModel(a.defaultModelID)
}

func (a *OpenRouterAdapter) CapabilitiesForModel(modelID string) Capabilities {
	if modelID == "" {
		modelID = a.defaultModelID
	}
	return capabilitiesWithModelID(Capabilities{
		Provider:           "openrouter",
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
	}, modelID)
}

func (a *OpenRouterAdapter) Identity() Identity {
	return Identity{Provider: "openrouter", ModelID: a.defaultModelID}
}

func (a *OpenRouterAdapter) RequestIdentity(opts GenerateOptions) (Identity, error) {
	modelID, err := a.effectiveModelID(opts)
	if err != nil {
		return Identity{}, err
	}
	return Identity{Provider: "openrouter", ModelID: modelID}, nil
}

func (a *OpenRouterAdapter) effectiveModelID(opts GenerateOptions) (string, error) {
	if err := llm.ValidateModelID(opts.ModelID); err != nil {
		return "", err
	}
	if opts.ModelID == "" {
		return a.defaultModelID, nil
	}
	return opts.ModelID, nil
}

func (a *OpenRouterAdapter) modelForOptions(opts GenerateOptions) (openroutersdk.LanguageModel, string, error) {
	modelID, err := a.effectiveModelID(opts)
	if err != nil {
		return nil, "", err
	}
	if modelID == a.defaultModelID {
		return a.model, modelID, nil
	}
	if a.newModel == nil {
		return nil, "", modelOverrideError("openrouter")
	}
	return a.newModel(modelID), modelID, nil
}

// Generate sends a completion request through the OpenRouter SDK.
func (a *OpenRouterAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	model, effectiveModelID, err := a.modelForOptions(opts)
	if err != nil {
		return nil, err
	}
	sdkOpts, warnings, err := toOpenRouterOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	out := fromOpenRouterResult(result)
	if out != nil && out.Response.ModelID == "" {
		out.Response.ModelID = effectiveModelID
	}
	if out != nil && len(warnings) > 0 {
		out.Warnings = append(warnings, out.Warnings...)
	}
	return out, nil
}

// Stream sends a streaming completion request through the OpenRouter SDK and
// maps provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *OpenRouterAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	model, effectiveModelID, err := a.modelForOptions(opts)
	if err != nil {
		return nil, err
	}
	sdkOpts, _, err := toOpenRouterOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := model.DoStream(ctx, openroutersdk.StreamOptions{GenerateOptions: sdkOpts})
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
				for _, mapped := range mapOpenRouterStreamPart(part) {
					out <- fillStreamMetadataModelID(mapped, effectiveModelID)
				}
			}
		}
	}()
	return out, nil
}

// mapOpenRouterStreamPart converts an openroutersdk.StreamPart to zero or more
// neutral StreamParts. Returns nil for parts with no neutral equivalent
// (StreamFile, StreamSource). spec §1.1
func mapOpenRouterStreamPart(part openroutersdk.StreamPart) []StreamPart {
	switch p := part.(type) {
	case openroutersdk.StreamResponseMetadata:
		return []StreamPart{StreamMetadata{Response: ResponseMetadata{ID: p.ID, ModelID: p.ModelID}}}
	case openroutersdk.StreamTextStart:
		return []StreamPart{StreamTextStart{}}
	case openroutersdk.StreamTextDelta:
		return []StreamPart{StreamTextDelta{Text: p.Delta}}
	case openroutersdk.StreamTextEnd:
		return []StreamPart{StreamTextEnd{}}
	case openroutersdk.StreamReasoningStart:
		return []StreamPart{StreamReasoningStart{}}
	case openroutersdk.StreamReasoningDelta:
		return []StreamPart{StreamReasoningDelta{Text: p.Delta}}
	case openroutersdk.StreamReasoningEnd:
		return []StreamPart{StreamReasoningEnd{}}
	case openroutersdk.StreamToolInputStart:
		return []StreamPart{StreamToolCallStart{ID: p.ID, Name: p.ToolName}}
	case openroutersdk.StreamToolInputDelta:
		return []StreamPart{StreamToolInputDelta{ID: p.ID, JSON: p.Delta}}
	case openroutersdk.StreamToolInputEnd:
		return []StreamPart{StreamToolInputEnd{ID: p.ID}}
	case openroutersdk.StreamToolCall:
		return []StreamPart{StreamToolInputEnd{ID: p.ToolCallID, Input: openRouterInputToRaw(p.Input)}}
	case openroutersdk.StreamFinish:
		return []StreamPart{StreamFinish{
			FinishReason: fromOpenRouterFinishReason(p.FinishReason),
			Usage: Usage{
				InputTokens:       p.Usage.InputTokens,
				OutputTokens:      p.Usage.OutputTokens,
				TotalTokens:       p.Usage.TotalTokens,
				CachedInputTokens: p.Usage.InputTokensDetails.CachedTokens,
				ReasoningTokens:   p.Usage.OutputTokensDetails.ReasoningTokens,
			},
			ProviderMetadata: openRouterProviderMetadata(p.ProviderMetadata),
		}}
	case openroutersdk.StreamError:
		return []StreamPart{StreamError{Err: p.Err}}
	case openroutersdk.StreamRaw:
		return []StreamPart{StreamRaw{}}
	default:
		return nil
	}
}

// openRouterProviderMetadata wraps openrouter provider metadata under the
// "openrouter" key, or returns nil when empty.
func openRouterProviderMetadata(pm openroutersdk.ProviderMetadata) ProviderMetadata {
	if pm == nil {
		return nil
	}
	return ProviderMetadata{"openrouter": pm}
}

func toOpenRouterOptions(opts GenerateOptions) (openroutersdk.GenerateOptions, []Warning, error) {
	messages := make([]openroutersdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, openroutersdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		converted, err := toOpenRouterMessages(message)
		if err != nil {
			return openroutersdk.GenerateOptions{}, nil, err
		}
		messages = append(messages, converted...)
	}
	tools, err := toOpenRouterTools(opts.Tools)
	if err != nil {
		return openroutersdk.GenerateOptions{}, nil, err
	}
	sdkOpts := openroutersdk.GenerateOptions{Messages: messages, Tools: tools}
	if opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		sdkOpts.MaxTokens = &maxTokens
	}
	sdkOpts.Temperature = opts.Temperature
	sdkOpts.TopP = opts.TopP
	sdkOpts.TopK = opts.TopK
	sdkOpts.Stop = opts.Stop
	sdkOpts.Seed = opts.Seed

	if opts.ToolChoice.Type != "" {
		sdkOpts.ToolChoice = openroutersdk.ToolChoice{Type: string(opts.ToolChoice.Type), ToolName: opts.ToolChoice.ToolName}
	}
	if opts.ResponseFormat != nil {
		rf, err := openRouterResponseFormat(opts.ResponseFormat)
		if err != nil {
			return openroutersdk.GenerateOptions{}, nil, err
		}
		sdkOpts.ResponseFormat = rf
	}
	if opts.ProviderOptions != nil {
		if po, ok := opts.ProviderOptions["openrouter"]; ok {
			sdkOpts.ProviderOptions = openroutersdk.ProviderOptions{"openrouter": po}
		}
	}
	sdkOpts.Headers = toHTTPHeader(opts.Headers)
	return sdkOpts, nil, nil
}

// openRouterResponseFormat maps a neutral ResponseFormat to the OpenRouter
// provider form. OpenRouter uses "json_object"/"json_schema" types directly.
func openRouterResponseFormat(rf *ResponseFormat) (*openroutersdk.ResponseFormat, error) {
	switch rf.Type {
	case llm.ResponseFormatText:
		return &openroutersdk.ResponseFormat{Type: "text"}, nil
	case llm.ResponseFormatJSONObject:
		return &openroutersdk.ResponseFormat{Type: "json_object", Name: rf.Name, Strict: rf.Strict}, nil
	case llm.ResponseFormatJSONSchema:
		schema, err := decodeSchema(rf.JSONSchema)
		if err != nil {
			return nil, err
		}
		return &openroutersdk.ResponseFormat{Type: "json_schema", Schema: schema, Name: rf.Name, Strict: rf.Strict}, nil
	default:
		return &openroutersdk.ResponseFormat{Type: string(rf.Type), Name: rf.Name, Strict: rf.Strict}, nil
	}
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
	out := &GenerateResult{
		Content:      fromOpenRouterContent(result.Content),
		FinishReason: fromOpenRouterFinishReason(result.FinishReason),
		Usage: Usage{
			InputTokens:       result.Usage.InputTokens,
			OutputTokens:      result.Usage.OutputTokens,
			TotalTokens:       result.Usage.TotalTokens,
			CachedInputTokens: result.Usage.InputTokensDetails.CachedTokens,
			ReasoningTokens:   result.Usage.OutputTokensDetails.ReasoningTokens,
		},
	}
	out.Response = ResponseMetadata{
		ID:        result.Response.ID,
		ModelID:   result.Response.ModelID,
		Headers:   flattenHeader(result.Response.Headers),
		Timestamp: result.Response.Timestamp,
	}
	for _, w := range result.Warnings {
		out.Warnings = append(out.Warnings, Warning{Code: w.Type, Message: w.Message, Provider: "openrouter"})
	}
	if result.ProviderMetadata != nil {
		pm := make(ProviderMetadata)
		pm["openrouter"] = result.ProviderMetadata
		out.ProviderMetadata = pm
	}
	return out
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
	raw := string(reason)
	switch reason {
	case openroutersdk.FinishReasonStop:
		return FinishReason{Unified: FinishReasonStop, Raw: raw}
	case openroutersdk.FinishReasonToolCalls:
		return FinishReason{Unified: FinishReasonToolCalls, Raw: raw}
	case openroutersdk.FinishReasonLength:
		return FinishReason{Unified: FinishReasonLength, Raw: raw}
	case openroutersdk.FinishReasonContentFilter:
		return FinishReason{Unified: FinishReasonContentFilter, Raw: raw}
	case openroutersdk.FinishReasonOther:
		return FinishReason{Unified: FinishReasonOther, Raw: raw}
	default:
		return FinishReason{Unified: FinishReasonError, Raw: raw}
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
