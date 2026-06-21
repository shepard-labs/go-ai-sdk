package adapters

import (
	"context"
	"fmt"
	"net/http"

	"github.com/shepard-labs/go-ai-sdk/llm"
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

// Capabilities reports the feature set supported by the OpenAI-compatible
// adapter. Actual support varies by the backing endpoint (Ollama, LM Studio,
// vLLM, etc.); these flags reflect what the adapter is able to forward, not a
// guarantee the endpoint honors them. Reasoning is reported false because the
// generic OpenAI-compatible wire format does not standardize reasoning output.
func (a *OpenAICompatibleAdapter) Capabilities() Capabilities {
	return Capabilities{
		Provider:           "openaicompatible",
		Streaming:          true,
		ToolCalling:        true,
		ToolChoiceAuto:     true,
		ToolChoiceNone:     true,
		ToolChoiceRequired: true,
		ToolChoiceTool:     true,
		StructuredOutput:   true,
		JSONMode:           true,
		Images:             true,
		Reasoning:          false,
		ParallelToolCalls:  true,
		PromptCaching:      false,
	}
}

// Generate sends a completion request through the OpenAI-compatible SDK.
func (a *OpenAICompatibleAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, warnings, err := toOpenAICompatibleOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	out := fromOpenAICompatibleResult(result)
	if out != nil && len(warnings) > 0 {
		out.Warnings = append(warnings, out.Warnings...)
	}
	return out, nil
}

// Stream sends a streaming completion request through the OpenAI-compatible
// SDK and maps provider-native StreamPart values into the neutral StreamPart
// union. spec §1.1
func (a *OpenAICompatibleAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, _, err := toOpenAICompatibleOptions(opts)
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
				for _, mapped := range mapOpenAICompatibleStreamPart(part) {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapOpenAICompatibleStreamPart converts an openaicompatiblesdk.StreamPart to
// zero or more neutral StreamParts. Returns nil for parts with no neutral
// equivalent (StreamCustomPart). spec §1.1
func mapOpenAICompatibleStreamPart(part openaicompatiblesdk.StreamPart) []StreamPart {
	switch p := part.(type) {
	case openaicompatiblesdk.StreamStart:
		return openAICompatibleStartWarnings(p.Warnings)
	case openaicompatiblesdk.StreamResponseMetadata:
		return []StreamPart{openAICompatibleMetadata(p)}
	case openaicompatiblesdk.StreamTextStart:
		return []StreamPart{StreamTextStart{}}
	case openaicompatiblesdk.StreamTextDelta:
		return []StreamPart{StreamTextDelta{Text: p.Text}}
	case openaicompatiblesdk.StreamTextEnd:
		return []StreamPart{StreamTextEnd{}}
	case openaicompatiblesdk.StreamReasoningStart:
		return []StreamPart{StreamReasoningStart{}}
	case openaicompatiblesdk.StreamReasoningDelta:
		return []StreamPart{StreamReasoningDelta{Text: p.Text}}
	case openaicompatiblesdk.StreamReasoningEnd:
		return []StreamPart{StreamReasoningEnd{}}
	case openaicompatiblesdk.StreamToolInputStart:
		return []StreamPart{StreamToolCallStart{ID: p.ID, Name: p.ToolName}}
	case openaicompatiblesdk.StreamToolInputDelta:
		return []StreamPart{StreamToolInputDelta{ID: p.ID, JSON: p.Delta}}
	case openaicompatiblesdk.StreamToolInputEnd:
		return []StreamPart{StreamToolInputEnd{ID: p.ID}}
	case openaicompatiblesdk.StreamToolCall:
		return []StreamPart{StreamToolInputEnd{ID: p.ToolCallID, Input: cloneRawMessage(p.Input)}}
	case openaicompatiblesdk.StreamFinish:
		return []StreamPart{StreamFinish{
			FinishReason: fromUnifiedFinishReason(p.FinishReason.Unified),
			Usage: Usage{
				InputTokens:     derefInt(p.Usage.InputTokens.Total),
				OutputTokens:    derefInt(p.Usage.OutputTokens.Total),
				ReasoningTokens: derefInt(p.Usage.OutputTokens.Reasoning),
			},
			ProviderMetadata: openAICompatibleProviderMetadata("openaicompatible", p.ProviderMetadata),
		}}
	case openaicompatiblesdk.StreamError:
		return []StreamPart{StreamError{Err: p.Err}}
	case openaicompatiblesdk.StreamRaw:
		return []StreamPart{StreamRaw{Bytes: p.Raw}}
	default:
		return nil
	}
}

// openAICompatibleStartWarnings converts provider StreamStart warnings to
// neutral StreamWarning parts.
func openAICompatibleStartWarnings(warnings []openaicompatiblesdk.Warning) []StreamPart {
	if len(warnings) == 0 {
		return nil
	}
	out := make([]StreamPart, 0, len(warnings))
	for _, w := range warnings {
		out = append(out, StreamWarning{Warning: Warning{Code: w.Type, Message: w.Message, Provider: "openaicompatible"}})
	}
	return out
}

// openAICompatibleMetadata converts provider response metadata to a neutral
// StreamMetadata part.
func openAICompatibleMetadata(p openaicompatiblesdk.StreamResponseMetadata) StreamMetadata {
	meta := StreamMetadata{Response: ResponseMetadata{ID: p.ID, ModelID: p.ModelID}}
	if p.Timestamp != nil {
		meta.Response.Timestamp = *p.Timestamp
	}
	return meta
}

// openAICompatibleProviderMetadata wraps provider metadata under the given
// provider key, or returns nil when empty.
func openAICompatibleProviderMetadata(provider string, pm openaicompatiblesdk.ProviderMetadata) ProviderMetadata {
	if pm == nil {
		return nil
	}
	return ProviderMetadata{provider: pm}
}

func toOpenAICompatibleOptions(opts GenerateOptions) (openaicompatiblesdk.GenerateOptions, []Warning, error) {
	messages := make([]openaicompatiblesdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, openaicompatiblesdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		converted, err := toOpenAICompatibleMessages(message)
		if err != nil {
			return openaicompatiblesdk.GenerateOptions{}, nil, err
		}
		messages = append(messages, converted...)
	}
	tools, err := toOpenAICompatibleTools(opts.Tools)
	if err != nil {
		return openaicompatiblesdk.GenerateOptions{}, nil, err
	}
	sdkOpts := openaicompatiblesdk.GenerateOptions{Messages: messages, Tools: tools}
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
		sdkOpts.ToolChoice = &openaicompatiblesdk.ToolChoice{Type: string(opts.ToolChoice.Type), ToolName: opts.ToolChoice.ToolName}
	}
	if opts.ResponseFormat != nil {
		rf, err := openAIResponseFormat(opts.ResponseFormat)
		if err != nil {
			return openaicompatiblesdk.GenerateOptions{}, nil, err
		}
		sdkOpts.ResponseFormat = rf
	}
	// Forward provider-specific options. The OpenAI adapter shares this
	// function and namespaces under "openai"; the OpenAI-compatible provider
	// uses "openaicompatible". Forward whichever the caller supplied.
	for _, key := range []string{"openai", "openaicompatible"} {
		if po, ok := opts.ProviderOptions[key]; ok {
			if sdkOpts.ProviderOptions == nil {
				sdkOpts.ProviderOptions = openaicompatiblesdk.ProviderOptions{}
			}
			sdkOpts.ProviderOptions[key] = po
		}
	}
	sdkOpts.Headers = toHTTPHeader(opts.Headers)
	return sdkOpts, nil, nil
}

// openAIResponseFormat maps a neutral ResponseFormat to the openaicompatible
// (and openai) provider form. The provider uses Type "json" for both
// json_object and json_schema; a schema is attached when present.
func openAIResponseFormat(rf *ResponseFormat) (*openaicompatiblesdk.ResponseFormat, error) {
	switch rf.Type {
	case llm.ResponseFormatText:
		return &openaicompatiblesdk.ResponseFormat{Type: "text"}, nil
	case llm.ResponseFormatJSONObject:
		return &openaicompatiblesdk.ResponseFormat{Type: "json", Name: rf.Name}, nil
	case llm.ResponseFormatJSONSchema:
		schema, err := decodeSchema(rf.JSONSchema)
		if err != nil {
			return nil, err
		}
		return &openaicompatiblesdk.ResponseFormat{Type: "json", Schema: schema, Name: rf.Name}, nil
	default:
		return &openaicompatiblesdk.ResponseFormat{Type: string(rf.Type), Name: rf.Name}, nil
	}
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
// openaicompatible FileContent (inline bytes) form. URL-source images are not
// representable: the openaicompatible wire format does not carry a bare image
// URL part, so a URL-source image is dropped per spec §1.2. Returns
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
	out := &GenerateResult{
		Content:      fromOpenAICompatibleContent(result.Content),
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
		out.Warnings = append(out.Warnings, Warning{Code: w.Type, Message: w.Message, Provider: "openaicompatible"})
	}
	if result.ProviderMetadata != nil {
		pm := make(ProviderMetadata)
		pm["openaicompatible"] = result.ProviderMetadata
		out.ProviderMetadata = pm
	}
	return out
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
