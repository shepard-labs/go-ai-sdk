package adapters

import (
	"context"
	"encoding/base64"
	"fmt"

	anthropicsdk "github.com/shepard-labs/go-ai-sdk/anthropic"
	. "github.com/shepard-labs/go-ai-sdk/llm"
)

// AnthropicAdapter adapts go-ai-sdk Anthropic models to the Client interface.
type AnthropicAdapter struct {
	model anthropicsdk.LanguageModel
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
	return &AnthropicAdapter{model: model}
}

// NewAnthropicClient creates an Anthropic-backed Client from an API key and model ID.
func NewAnthropicClient(apiKey string, modelID AnthropicModelID) (Client, error) {
	provider := anthropicsdk.CreateAnthropic(anthropicsdk.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		return nil, err
	}
	return NewAnthropicAdapter(provider.Model(string(modelID))), nil
}

// Generate sends a completion request through the Anthropic SDK.
func (a *AnthropicAdapter) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	sdkOpts, err := toAnthropicOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoGenerate(ctx, sdkOpts)
	if err != nil {
		return nil, err
	}
	return fromAnthropicResult(result), nil
}

// Stream sends a streaming completion request through the Anthropic SDK and
// maps provider-native StreamPart values into the neutral StreamPart union.
// spec §1.1
func (a *AnthropicAdapter) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	sdkOpts, err := toAnthropicOptions(opts)
	if err != nil {
		return nil, err
	}
	result, err := a.model.DoStream(ctx, sdkOpts)
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
				if mapped, emit := mapAnthropicStreamPart(part); emit {
					out <- mapped
				}
			}
		}
	}()
	return out, nil
}

// mapAnthropicStreamPart converts an anthropicsdk.StreamPart to a neutral
// StreamPart. Returns (zero, false) for parts with no neutral equivalent
// (StreamStart, StreamResponseMetadata, StreamToolResult, StreamSource,
// StreamRaw). spec §1.1
func mapAnthropicStreamPart(part anthropicsdk.StreamPart) (StreamPart, bool) {
	switch p := part.(type) {
	case anthropicsdk.StreamTextStart:
		return StreamTextStart{}, true
	case anthropicsdk.StreamTextDelta:
		return StreamTextDelta{Text: p.Text}, true
	case anthropicsdk.StreamTextEnd:
		return StreamTextEnd{}, true
	case anthropicsdk.StreamReasoningStart:
		return StreamReasoningStart{}, true
	case anthropicsdk.StreamReasoningDelta:
		return StreamReasoningDelta{Text: anthropicReasoningDeltaText(p.Delta)}, true
	case anthropicsdk.StreamReasoningEnd:
		return StreamReasoningEnd{}, true
	case anthropicsdk.StreamToolInputStart:
		return StreamToolCallStart{ID: p.ID, Name: p.ToolName}, true
	case anthropicsdk.StreamToolInputDelta:
		return StreamToolInputDelta{ID: p.ID, JSON: anthropicToolInputDeltaText(p.Delta)}, true
	case anthropicsdk.StreamToolInputEnd:
		return StreamToolInputEnd{ID: p.ID}, true
	case anthropicsdk.StreamToolCall:
		return StreamToolInputEnd{ID: p.ToolCallID, Input: cloneRawMessage(p.Input)}, true
	case anthropicsdk.StreamFinish:
		return StreamFinish{FinishReason: fromAnthropicFinishReason(p.FinishReason), Usage: Usage{InputTokens: p.Usage.InputTokens.Total, OutputTokens: p.Usage.OutputTokens.Total}}, true
	case anthropicsdk.StreamError:
		return StreamError{Err: p.Err}, true
	case anthropicsdk.StreamRaw:
		return StreamRaw{Bytes: nil}, true
	default:
		return nil, false
	}
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

func toAnthropicOptions(opts GenerateOptions) (anthropicsdk.GenerateOptions, error) {
	messages := make([]anthropicsdk.Message, 0, len(opts.Messages)+1)
	if opts.System != "" {
		messages = append(messages, anthropicsdk.SystemMessage{Content: opts.System})
	}
	for _, message := range opts.Messages {
		sdkMessage, err := toAnthropicMessage(message)
		if err != nil {
			return anthropicsdk.GenerateOptions{}, err
		}
		messages = append(messages, sdkMessage)
	}
	tools, err := toAnthropicTools(opts.Tools)
	if err != nil {
		return anthropicsdk.GenerateOptions{}, err
	}
	return anthropicsdk.GenerateOptions{Messages: messages, Tools: tools, MaxTokens: opts.MaxTokens}, nil
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
	return &GenerateResult{Content: fromAnthropicContent(result.Content), FinishReason: fromAnthropicFinishReason(result.FinishReason), Usage: Usage{InputTokens: result.Usage.InputTokens.Total, OutputTokens: result.Usage.OutputTokens.Total}}
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
	switch reason {
	case anthropicsdk.FinishReasonStop:
		return FinishReasonStop
	case anthropicsdk.FinishReasonToolCalls:
		return FinishReasonToolCalls
	case anthropicsdk.FinishReasonLength:
		return FinishReasonLength
	case anthropicsdk.FinishReasonError:
		return FinishReasonError
	default:
		return FinishReasonError
	}
}
