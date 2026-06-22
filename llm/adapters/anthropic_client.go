package adapters

import (
	"context"
	"encoding/base64"
	"fmt"
	"reflect"

	anthropicsdk "github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/llm"
)

// AnthropicAdapter adapts go-ai-sdk Anthropic models to the Client interface.
type AnthropicAdapter struct {
	model          anthropicsdk.LanguageModel
	defaultModelID string
	newModel       func(modelID string) anthropicsdk.LanguageModel
}

// AnthropicModelID names an Anthropic model accepted by NewAnthropicClient.
type AnthropicModelID = anthropicsdk.ModelID

const (
	// AnthropicModelClaudeHaiku45 is the Claude Haiku model identifier.
	AnthropicModelClaudeHaiku45 = anthropicsdk.ModelClaudeHaiku45
	// AnthropicModelClaudeSonnet46 is the Claude Sonnet model identifier.
	AnthropicModelClaudeSonnet46 = anthropicsdk.ModelClaudeSonnet46
	// AnthropicModelClaudeOpus48 is the Claude Opus model identifier.
	AnthropicModelClaudeOpus48 = anthropicsdk.ModelClaudeOpus48
)

// NewAnthropicAdapter wraps an existing Anthropic language model as a Client.
func NewAnthropicAdapter(model anthropicsdk.LanguageModel) Client {
	return &AnthropicAdapter{model: model, defaultModelID: model.ModelID()}
}

// NewAnthropicClient creates an Anthropic-backed Client from an API key and model ID.
func NewAnthropicClient(apiKey string, modelID AnthropicModelID) (Client, error) {
	provider := anthropicsdk.CreateAnthropic(anthropicsdk.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		return nil, err
	}
	defaultModelID := string(modelID)
	return &AnthropicAdapter{
		model:          provider.Model(defaultModelID),
		defaultModelID: defaultModelID,
		newModel: func(id string) anthropicsdk.LanguageModel {
			return provider.Model(id)
		},
	}, nil
}

// Capabilities reports the feature set supported by the Anthropic adapter.
func (a *AnthropicAdapter) Capabilities() llm.Capabilities {
	return a.CapabilitiesForModel(a.defaultModelID)
}

func (a *AnthropicAdapter) CapabilitiesForModel(modelID string) llm.Capabilities {
	if modelID == "" {
		modelID = a.defaultModelID
	}
	return capabilitiesWithModelID(llm.Capabilities{
		Provider:           "anthropic",
		Streaming:          true,
		ToolCalling:        true,
		ToolChoiceAuto:     true,
		ToolChoiceNone:     false,
		ToolChoiceRequired: true,
		ToolChoiceTool:     true,
		StructuredOutput:   false,
		JSONMode:           false,
		Images:             true,
		Reasoning:          true,
		ParallelToolCalls:  true,
		PromptCaching:      true,
	}, modelID)
}

func (a *AnthropicAdapter) Identity() llm.Identity {
	return llm.Identity{Provider: "anthropic", ModelID: a.defaultModelID}
}

func (a *AnthropicAdapter) RequestIdentity(opts GenerateOptions) (llm.Identity, error) {
	modelID, err := a.effectiveModelID(opts)
	if err != nil {
		return llm.Identity{}, err
	}
	return llm.Identity{Provider: "anthropic", ModelID: modelID}, nil
}

func (a *AnthropicAdapter) effectiveModelID(opts GenerateOptions) (string, error) {
	if err := llm.ValidateModelID(opts.ModelID); err != nil {
		return "", err
	}
	if opts.ModelID == "" {
		return a.defaultModelID, nil
	}
	return opts.ModelID, nil
}

func (a *AnthropicAdapter) modelForOptions(opts GenerateOptions) (anthropicsdk.LanguageModel, string, error) {
	modelID, err := a.effectiveModelID(opts)
	if err != nil {
		return nil, "", err
	}
	if modelID == a.defaultModelID {
		return a.model, modelID, nil
	}
	if a.newModel == nil {
		return nil, "", modelOverrideError("anthropic")
	}
	return a.newModel(modelID), modelID, nil
}

// Generate sends a completion request through the Anthropic SDK.
func (a *AnthropicAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	model, effectiveModelID, err := a.modelForOptions(opts)
	if err != nil {
		return nil, err
	}
	sdkOpts, warnings, err := toAnthropicOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	out := fromAnthropicResult(result)
	if out != nil && out.Response.ModelID == "" {
		out.Response.ModelID = effectiveModelID
	}
	if out != nil && len(warnings) > 0 {
		out.Warnings = append(warnings, out.Warnings...)
	}
	return out, nil
}

// Stream sends a streaming completion request through the Anthropic SDK and
// maps provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *AnthropicAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	model, effectiveModelID, err := a.modelForOptions(opts)
	if err != nil {
		return nil, err
	}
	sdkOpts, _, err := toAnthropicOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := model.DoStream(ctx, sdkOpts)
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
				for _, mapped := range mapAnthropicStreamPart(part) {
					out <- fillStreamMetadataModelID(mapped, effectiveModelID)
				}
			}
		}
	}()
	return out, nil
}

// mapAnthropicStreamPart converts an anthropicsdk.StreamPart to zero or more
// neutral StreamParts. Returns nil for parts with no neutral equivalent
// (StreamStart, StreamToolResult, StreamSource). spec §1.1
func mapAnthropicStreamPart(part anthropicsdk.StreamPart) []StreamPart {
	switch p := part.(type) {
	case anthropicsdk.StreamResponseMetadata:
		return []StreamPart{StreamMetadata{Response: ResponseMetadata{ID: p.ID, ModelID: p.ModelID}}}
	case anthropicsdk.StreamTextStart:
		return []StreamPart{StreamTextStart{}}
	case anthropicsdk.StreamTextDelta:
		return []StreamPart{StreamTextDelta{Text: p.Text}}
	case anthropicsdk.StreamTextEnd:
		return []StreamPart{StreamTextEnd{}}
	case anthropicsdk.StreamReasoningStart:
		return []StreamPart{StreamReasoningStart{}}
	case anthropicsdk.StreamReasoningDelta:
		return []StreamPart{StreamReasoningDelta{Text: anthropicReasoningDeltaText(p.Delta)}}
	case anthropicsdk.StreamReasoningEnd:
		return []StreamPart{StreamReasoningEnd{}}
	case anthropicsdk.StreamToolInputStart:
		return []StreamPart{StreamToolCallStart{ID: p.ID, Name: p.ToolName}}
	case anthropicsdk.StreamToolInputDelta:
		return []StreamPart{StreamToolInputDelta{ID: p.ID, JSON: anthropicToolInputDeltaText(p.Delta)}}
	case anthropicsdk.StreamToolInputEnd:
		return []StreamPart{StreamToolInputEnd{ID: p.ID}}
	case anthropicsdk.StreamToolCall:
		return []StreamPart{StreamToolInputEnd{ID: p.ToolCallID, Input: cloneRawMessage(p.Input)}}
	case anthropicsdk.StreamFinish:
		return []StreamPart{StreamFinish{
			FinishReason: fromAnthropicFinishReason(p.FinishReason),
			Usage: Usage{
				InputTokens:              p.Usage.InputTokens.Total,
				OutputTokens:             p.Usage.OutputTokens.Total,
				TotalTokens:              p.Usage.TotalTokens,
				CachedInputTokens:        p.Usage.InputTokens.CacheRead,
				CacheCreationInputTokens: p.Usage.InputTokens.CacheWrite,
			},
			ProviderMetadata: anthropicProviderMetadata(p.ProviderMetadata),
		}}
	case anthropicsdk.StreamError:
		return []StreamPart{StreamError{Err: p.Err}}
	case anthropicsdk.StreamRaw:
		return []StreamPart{StreamRaw{Bytes: nil}}
	default:
		return nil
	}
}

// anthropicProviderMetadata wraps anthropic provider metadata under the
// "anthropic" key, or returns nil when empty.
func anthropicProviderMetadata(pm anthropicsdk.ProviderMetadata) ProviderMetadata {
	if pm == nil {
		return nil
	}
	return ProviderMetadata{"anthropic": pm}
}

// anthropicReasoningDeltaText extracts the text fragment from a reasoning
// delta. Anthropic uses a Delta interface (ThinkingDelta).
func anthropicReasoningDeltaText(delta anthropicsdk.Delta) string {
	if d, ok := delta.(anthropicsdk.ThinkingDelta); ok {
		return d.Thinking
	}
	return ""
}

// anthropicToolInputDeltaText extracts the partial-JSON fragment from a tool
// input delta. Anthropic uses InputJSONDelta.
func anthropicToolInputDeltaText(delta anthropicsdk.Delta) string {
	if d, ok := delta.(anthropicsdk.InputJSONDelta); ok {
		return d.PartialJSON
	}
	return ""
}

func toAnthropicOptions(opts GenerateOptions) (anthropicsdk.GenerateOptions, []Warning, error) {
	var warnings []Warning
	if err := validateReasoning(opts.UnsupportedFeaturePolicy, "anthropic", opts.Reasoning, &warnings); err != nil {
		return anthropicsdk.GenerateOptions{}, nil, err
	}
	messages := make([]anthropicsdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, anthropicsdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		sdkMessage, err := toAnthropicMessage(message)
		if err != nil {
			return anthropicsdk.GenerateOptions{}, nil, err
		}
		messages = append(messages, sdkMessage)
	}
	tools, err := toAnthropicTools(opts.Tools)
	if err != nil {
		return anthropicsdk.GenerateOptions{}, nil, err
	}
	sdkOpts := anthropicsdk.GenerateOptions{
		Messages:  messages,
		Tools:     tools,
		MaxTokens: opts.MaxTokens,
	}
	sdkOpts.Temperature = opts.Temperature
	sdkOpts.TopP = opts.TopP
	sdkOpts.TopK = opts.TopK
	sdkOpts.StopSequences = opts.Stop
	sdkOpts.Seed = opts.Seed
	if opts.Reasoning != nil {
		thinking, err := anthropicThinkingFromReasoning(opts.Reasoning)
		if err != nil {
			return anthropicsdk.GenerateOptions{}, nil, err
		}
		if thinking != nil && !setAnthropicThinking(&sdkOpts, thinking) {
			if err := unsupportedFeature(opts.UnsupportedFeaturePolicy, "anthropic", "reasoning", "per-request thinking requires a newer Anthropic SDK dependency", &warnings); err != nil {
				return anthropicsdk.GenerateOptions{}, nil, err
			}
		}
	}

	if choice, err := toAnthropicToolChoice(opts.ToolChoice, opts.UnsupportedFeaturePolicy, &warnings); err != nil {
		return anthropicsdk.GenerateOptions{}, nil, err
	} else if choice != nil {
		sdkOpts.ToolChoice = choice
	}

	if opts.ResponseFormat != nil {
		if err := unsupportedFeature(opts.UnsupportedFeaturePolicy, "anthropic", "response_format", "structured output is not supported via the neutral ResponseFormat field", &warnings); err != nil {
			return anthropicsdk.GenerateOptions{}, nil, err
		}
	}

	if len(opts.Headers) > 0 {
		if err := unsupportedFeature(opts.UnsupportedFeaturePolicy, "anthropic", "custom_headers", "per-request headers are not supported", &warnings); err != nil {
			return anthropicsdk.GenerateOptions{}, nil, err
		}
	}

	if len(opts.ProviderOptions["anthropic"]) > 0 {
		warnings = append(warnings, Warning{
			Code:     "unsupported_feature",
			Message:  "provider_options: anthropic provider options are not forwarded",
			Provider: "anthropic",
		})
	}

	return sdkOpts, warnings, nil
}

func setAnthropicThinking(opts *anthropicsdk.GenerateOptions, thinking *anthropicsdk.ThinkingConfig) bool {
	if opts == nil || thinking == nil {
		return false
	}
	field := reflect.ValueOf(opts).Elem().FieldByName("Thinking")
	if !field.IsValid() || !field.CanSet() || field.Type() != reflect.TypeOf(thinking) {
		return false
	}
	field.Set(reflect.ValueOf(thinking))
	return true
}

func anthropicThinkingFromReasoning(reasoning *llm.ReasoningOptions) (*anthropicsdk.ThinkingConfig, error) {
	if reasoning == nil {
		return nil, nil
	}
	if reasoning.Effort == "" && reasoning.BudgetTokens == nil {
		return nil, nil
	}
	if reasoning.Effort == llm.ReasoningNone {
		return &anthropicsdk.ThinkingConfig{Type: anthropicsdk.ThinkingTypeDisabled}, nil
	}
	budget := anthropicReasoningBudget(reasoning.Effort)
	if reasoning.BudgetTokens != nil {
		budget = *reasoning.BudgetTokens
	}
	return &anthropicsdk.ThinkingConfig{Type: anthropicsdk.ThinkingTypeEnabled, BudgetTokens: budget}, nil
}

func anthropicReasoningBudget(effort llm.ReasoningEffort) int {
	switch effort {
	case llm.ReasoningMinimal:
		return 1024
	case llm.ReasoningLow:
		return 2048
	case llm.ReasoningMedium, "":
		return 4096
	case llm.ReasoningHigh:
		return 8192
	case llm.ReasoningXHigh:
		return 16384
	default:
		return 4096
	}
}

// toAnthropicToolChoice maps a neutral ToolChoice to the Anthropic SDK form.
// Returns (nil, nil) when no constraint is set. ToolChoiceNone is unsupported.
func toAnthropicToolChoice(choice ToolChoice, policy llm.UnsupportedFeaturePolicy, warnings *[]Warning) (*anthropicsdk.ToolChoice, error) {
	switch choice.Type {
	case "":
		return nil, nil
	case llm.ToolChoiceAuto:
		return &anthropicsdk.ToolChoice{Type: "auto"}, nil
	case llm.ToolChoiceRequired:
		return &anthropicsdk.ToolChoice{Type: "any"}, nil
	case llm.ToolChoiceTool:
		return &anthropicsdk.ToolChoice{Type: "tool", Name: choice.ToolName}, nil
	case llm.ToolChoiceNone:
		if err := unsupportedFeature(policy, "anthropic", "tool_choice_none", "tool choice \"none\" is not supported", warnings); err != nil {
			return nil, err
		}
		return nil, nil
	default:
		if err := unsupportedFeature(policy, "anthropic", "tool_choice", fmt.Sprintf("unsupported tool choice %q", choice.Type), warnings); err != nil {
			return nil, err
		}
		return nil, nil
	}
}

func toAnthropicMessage(message Message) (anthropicsdk.Message, error) {
	switch message.Role {
	case "user":
		contents := make([]anthropicsdk.UserContent, 0, len(message.Content))
		for _, content := range message.Content {
			sdkContent, err := toAnthropicUserContent(content)
			if err != nil {
				return nil, err
			}
			contents = append(contents, sdkContent)
		}
		return anthropicsdk.UserMessage{Content: contents}, nil
	case "assistant":
		contents := make([]anthropicsdk.AssistantContent, 0, len(message.Content))
		for _, content := range message.Content {
			sdkContent, err := toAnthropicAssistantContent(content)
			if err != nil {
				return nil, err
			}
			contents = append(contents, sdkContent)
		}
		return anthropicsdk.AssistantMessage{Content: contents}, nil
	default:
		return nil, fmt.Errorf("llm: unknown message role %q", message.Role)
	}
}

func toAnthropicUserContent(content Content) (anthropicsdk.UserContent, error) {
	switch c := content.(type) {
	case TextContent:
		return anthropicsdk.TextContent{Text: c.Text}, nil
	case ToolResultContent:
		return anthropicsdk.ToolResultContent{ToolCallID: c.ToolUseID, Result: []anthropicsdk.ToolResultPart{anthropicsdk.ToolResultText{Text: c.Text}}, IsError: c.IsError}, nil
	case ImageContent:
		return anthropicImageContent(c), nil
	default:
		return nil, fmt.Errorf("llm: unsupported user content %T", content)
	}
}

func toAnthropicAssistantContent(content Content) (anthropicsdk.AssistantContent, error) {
	switch c := content.(type) {
	case TextContent:
		return anthropicsdk.TextContent{Text: c.Text}, nil
	case ToolUseContent:
		return anthropicsdk.ToolCallContent{ToolCallID: c.ID, ToolName: c.Name, Input: cloneRawMessage(c.Input)}, nil
	case ReasoningContent:
		return anthropicsdk.ThinkingContent{Thinking: c.Text}, nil
	default:
		return nil, fmt.Errorf("llm: unsupported assistant content %T", content)
	}
}

// anthropicImageContent converts a neutral ImageContent to an Anthropic
// ImageContent. Inline bytes are base64-encoded; URL sources use the URL form.
// spec §1.2
func anthropicImageContent(c ImageContent) anthropicsdk.ImageContent {
	switch src := c.Source.(type) {
	case ImageURLSource:
		return anthropicsdk.ImageContent{Source: anthropicsdk.ImageSource{Type: "url", MediaType: c.MIME, URL: src.URL}}
	case ImageInlineSource:
		return anthropicsdk.ImageContent{Source: anthropicsdk.ImageSource{Type: "base64", MediaType: c.MIME, Data: base64.StdEncoding.EncodeToString(src.Data)}}
	}
	return anthropicsdk.ImageContent{}
}

func toAnthropicTools(tools []Tool) ([]anthropicsdk.Tool, error) {
	sdkTools := make([]anthropicsdk.Tool, len(tools))
	for i, tool := range tools {
		schema, err := decodeSchema(tool.InputSchema)
		if err != nil {
			return nil, err
		}
		sdkTools[i] = anthropicsdk.Tool{Name: tool.Name, Description: tool.Description, InputSchema: schema}
	}
	return sdkTools, nil
}

func fromAnthropicResult(result *anthropicsdk.GenerateResult) *GenerateResult {
	if result == nil {
		return nil
	}
	out := &GenerateResult{
		Content:      fromAnthropicContent(result.Content),
		FinishReason: fromAnthropicFinishReason(result.FinishReason),
		Usage: Usage{
			InputTokens:              result.Usage.InputTokens.Total,
			OutputTokens:             result.Usage.OutputTokens.Total,
			TotalTokens:              result.Usage.TotalTokens,
			CachedInputTokens:        result.Usage.InputTokens.CacheRead,
			CacheCreationInputTokens: result.Usage.InputTokens.CacheWrite,
		},
	}
	// Map message metadata → neutral response metadata.
	if id, ok := result.MessageMetadata["id"].(string); ok {
		out.Response.ID = id
	}
	if model, ok := result.MessageMetadata["model"].(string); ok {
		out.Response.ModelID = model
	}
	// Map provider warnings → neutral warnings.
	for _, w := range result.Warnings {
		out.Warnings = append(out.Warnings, Warning{Code: w.Type, Message: w.Message, Provider: "anthropic"})
	}
	// Map provider metadata under the "anthropic" key.
	if result.ProviderMetadata != nil {
		pm := make(ProviderMetadata)
		pm["anthropic"] = result.ProviderMetadata
		out.ProviderMetadata = pm
	}
	return out
}

func fromAnthropicContent(contents []anthropicsdk.Content) []Content {
	converted := make([]Content, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case anthropicsdk.TextContent:
			converted = append(converted, TextContent{Text: c.Text})
		case anthropicsdk.ToolCallContent:
			converted = append(converted, ToolUseContent{ID: c.ToolCallID, Name: c.ToolName, Input: cloneRawMessage(c.Input)})
		case anthropicsdk.ToolResultContent:
			converted = append(converted, ToolResultContent{ToolUseID: c.ToolCallID, Text: toolResultText(c.Result), IsError: c.IsError})
		case anthropicsdk.ReasoningContent:
			converted = append(converted, ReasoningContent{Text: c.Text})
		}
	}
	return converted
}

func toolResultText(parts []anthropicsdk.ToolResultPart) string {
	for _, part := range parts {
		if text, ok := part.(anthropicsdk.ToolResultText); ok {
			return text.Text
		}
	}
	return ""
}

func fromAnthropicFinishReason(reason anthropicsdk.FinishReason) FinishReason {
	raw := string(reason)
	switch reason {
	case anthropicsdk.FinishReasonStop:
		return FinishReason{Unified: FinishReasonStop, Raw: raw}
	case anthropicsdk.FinishReasonToolCalls:
		return FinishReason{Unified: FinishReasonToolCalls, Raw: raw}
	case anthropicsdk.FinishReasonLength:
		return FinishReason{Unified: FinishReasonLength, Raw: raw}
	case anthropicsdk.FinishReasonContentFilter:
		return FinishReason{Unified: FinishReasonContentFilter, Raw: raw}
	case anthropicsdk.FinishReasonError:
		return FinishReason{Unified: FinishReasonError, Raw: raw}
	default:
		return FinishReason{Unified: FinishReasonError, Raw: raw}
	}
}
