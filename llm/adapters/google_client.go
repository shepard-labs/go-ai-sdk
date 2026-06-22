package adapters

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"

	googlesdk "github.com/shepard-labs/go-ai-sdk/google"
	"github.com/shepard-labs/go-ai-sdk/llm"
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

// Capabilities reports the feature set supported by the Google adapter.
func (a *GoogleAdapter) Capabilities() Capabilities {
	return Capabilities{
		Provider:           "google",
		Streaming:          true,
		ToolCalling:        true,
		ToolChoiceAuto:     true,
		ToolChoiceNone:     false,
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

// Generate sends a completion request through the Google SDK.
func (a *GoogleAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, warnings, err := toGoogleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	out := fromGoogleResult(result)
	if out != nil && len(warnings) > 0 {
		out.Warnings = append(warnings, out.Warnings...)
	}
	return out, nil
}

// Stream sends a streaming completion request through the Google SDK and maps
// provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *GoogleAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, _, err := toGoogleOptions(opts)
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
				for _, mapped := range mapGoogleStreamPart(part) {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapGoogleStreamPart converts a googlesdk.StreamPart to zero or more neutral
// StreamParts. Returns nil for parts with no neutral equivalent (StreamSource,
// StreamFile, StreamReasoningFile, StreamToolResult). spec §1.1
func mapGoogleStreamPart(part googlesdk.StreamPart) []StreamPart {
	switch p := part.(type) {
	case googlesdk.StreamStart:
		return googleStartWarnings(p.Warnings)
	case googlesdk.StreamResponseMetadata:
		meta := StreamMetadata{Response: ResponseMetadata{ID: p.ID, ModelID: p.ModelID}}
		if p.Timestamp != nil {
			meta.Response.Timestamp = *p.Timestamp
		}
		return []StreamPart{meta}
	case googlesdk.StreamTextStart:
		return []StreamPart{StreamTextStart{}}
	case googlesdk.StreamTextDelta:
		return []StreamPart{StreamTextDelta{Text: p.Text}}
	case googlesdk.StreamTextEnd:
		return []StreamPart{StreamTextEnd{}}
	case googlesdk.StreamReasoningStart:
		return []StreamPart{StreamReasoningStart{}}
	case googlesdk.StreamReasoningDelta:
		return []StreamPart{StreamReasoningDelta{Text: p.Text}}
	case googlesdk.StreamReasoningEnd:
		return []StreamPart{StreamReasoningEnd{}}
	case googlesdk.StreamToolInputStart:
		return []StreamPart{StreamToolCallStart{ID: p.ID, Name: p.ToolName}}
	case googlesdk.StreamToolInputDelta:
		return []StreamPart{StreamToolInputDelta{ID: p.ID, JSON: p.Delta}}
	case googlesdk.StreamToolInputEnd:
		return []StreamPart{StreamToolInputEnd{ID: p.ID}}
	case googlesdk.StreamToolCall:
		return []StreamPart{StreamToolInputEnd{ID: p.ToolCall.ToolCallID, Input: cloneRawMessage(p.ToolCall.Input)}}
	case googlesdk.StreamFinish:
		return []StreamPart{StreamFinish{
			FinishReason: fromUnifiedFinishReason(p.FinishReason.Unified),
			Usage: Usage{
				InputTokens:     derefInt(p.Usage.InputTokens.Total),
				OutputTokens:    derefInt(p.Usage.OutputTokens.Total),
				ReasoningTokens: derefInt(p.Usage.OutputTokens.Reasoning),
			},
			ProviderMetadata: googleProviderMetadata(p.ProviderMetadata),
		}}
	case googlesdk.StreamError:
		return []StreamPart{StreamError{Err: p.Err}}
	case googlesdk.StreamRaw:
		return []StreamPart{StreamRaw{Bytes: p.Raw}}
	default:
		return nil
	}
}

// googleStartWarnings converts provider StreamStart warnings to neutral
// StreamWarning parts.
func googleStartWarnings(warnings []googlesdk.Warning) []StreamPart {
	if len(warnings) == 0 {
		return nil
	}
	out := make([]StreamPart, 0, len(warnings))
	for _, w := range warnings {
		out = append(out, StreamWarning{Warning: Warning{Code: w.Type, Message: w.Message, Provider: "google"}})
	}
	return out
}

// googleProviderMetadata wraps google provider metadata under the "google" key,
// or returns nil when empty.
func googleProviderMetadata(pm googlesdk.ProviderMetadata) ProviderMetadata {
	if pm == nil {
		return nil
	}
	return ProviderMetadata{"google": pm}
}

func toGoogleOptions(opts GenerateOptions) (googlesdk.GenerateOptions, []Warning, error) {
	var warnings []Warning
	if err := validateReasoning(opts.UnsupportedFeaturePolicy, "google", opts.Reasoning, &warnings); err != nil {
		return googlesdk.GenerateOptions{}, nil, err
	}
	messages := make([]googlesdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, googlesdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		converted, err := toGoogleMessages(message)
		if err != nil {
			return googlesdk.GenerateOptions{}, nil, err
		}
		messages = append(messages, converted...)
	}
	tools, err := toGoogleTools(opts.Tools)
	if err != nil {
		return googlesdk.GenerateOptions{}, nil, err
	}
	sdkOpts := googlesdk.GenerateOptions{Messages: messages, Tools: tools}
	if opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		sdkOpts.MaxOutputTokens = &maxTokens
	}
	sdkOpts.Temperature = opts.Temperature
	sdkOpts.TopP = opts.TopP
	sdkOpts.TopK = opts.TopK
	sdkOpts.StopSequences = opts.Stop
	sdkOpts.Seed = opts.Seed

	if opts.ToolChoice.Type != "" {
		sdkOpts.ToolChoice = &googlesdk.ToolChoice{Type: string(opts.ToolChoice.Type), ToolName: opts.ToolChoice.ToolName}
	}
	if opts.ResponseFormat != nil {
		rf, err := googleResponseFormat(opts.ResponseFormat)
		if err != nil {
			return googlesdk.GenerateOptions{}, nil, err
		}
		sdkOpts.ResponseFormat = rf
	}
	if opts.ProviderOptions != nil {
		if po, ok := opts.ProviderOptions["google"]; ok {
			sdkOpts.ProviderOptions = googlesdk.ProviderOptions{"google": cloneProviderOptionMap(po)}
		}
	}
	if opts.Reasoning != nil {
		if opts.Reasoning.BudgetTokens != nil {
			if err := reasoningFeature(opts.UnsupportedFeaturePolicy, "google", "reasoning_budget_unsupported", "reasoning_budget", "budgetTokens are not supported by the neutral Google adapter; use ProviderOptions[\"google\"].thinkingConfig for exact Gemini budgets", &warnings); err != nil {
				return googlesdk.GenerateOptions{}, nil, err
			}
		}
		if opts.Reasoning.Effort != "" {
			sdkOpts.Reasoning = string(opts.Reasoning.Effort)
		}
	}
	sdkOpts.Headers = toHTTPHeader(opts.Headers)
	return sdkOpts, warnings, nil
}

// googleResponseFormat maps a neutral ResponseFormat to the Google provider
// form. Google uses Type "json" for both json_object and json_schema and
// "text" otherwise; a schema is attached when present.
func googleResponseFormat(rf *ResponseFormat) (*googlesdk.ResponseFormat, error) {
	switch rf.Type {
	case llm.ResponseFormatText:
		return &googlesdk.ResponseFormat{Type: "text"}, nil
	case llm.ResponseFormatJSONObject:
		return &googlesdk.ResponseFormat{Type: "json", Name: rf.Name}, nil
	case llm.ResponseFormatJSONSchema:
		schema, err := decodeSchema(rf.JSONSchema)
		if err != nil {
			return nil, err
		}
		return &googlesdk.ResponseFormat{Type: "json", Schema: schema, Name: rf.Name}, nil
	default:
		return &googlesdk.ResponseFormat{Type: string(rf.Type), Name: rf.Name}, nil
	}
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
	out := &GenerateResult{
		Content:      fromGoogleContent(result.Content),
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
		out.Warnings = append(out.Warnings, Warning{Code: w.Type, Message: w.Message, Provider: "google"})
	}
	if result.ProviderMetadata != nil {
		pm := make(ProviderMetadata)
		pm["google"] = result.ProviderMetadata
		out.ProviderMetadata = pm
	}
	return out
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
