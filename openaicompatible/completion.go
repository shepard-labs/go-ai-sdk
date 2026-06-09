package openaicompatible

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type completionRequestConfig struct {
	Body     map[string]any
	Warnings []Warning
}

// buildCompletionRequest assembles the request body and warnings for a
// completion call. It does not apply TransformRequestBody (per spec).
func (m *openAICompatibleCompletionLanguageModel) buildCompletionRequest(opts GenerateOptions) (completionRequestConfig, error) {
	providerOptions := cloneProviderOptions(opts.ProviderOptions)
	merged := mergeCompletionProviderOptions(m.provider.name, providerOptions)

	var warnings []Warning

	// Warnings for unsupported settings.
	if opts.TopK != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "topK"})
	}
	if len(opts.Tools) > 0 {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "tools"})
	}
	if opts.ToolChoice != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "toolChoice"})
	}
	if opts.ResponseFormat != nil && opts.ResponseFormat.Type != "text" {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "responseFormat", Details: "JSON response format is not supported."})
	}

	// Flatten prompt to text.
	prompt, generatedStop, err := flattenCompletionPrompt(opts.Messages)
	if err != nil {
		return completionRequestConfig{}, err
	}

	body := map[string]any{"model": m.modelID}

	// Spread all provider option keys (including recognized ones) into body.
	for k, v := range merged {
		body[k] = v
	}

	// Map typed completion options from the merged map.
	if v, ok := merged["echo"]; ok {
		if b, ok := v.(*bool); ok && b != nil {
			body["echo"] = *b
		} else if b, ok := v.(bool); ok {
			body["echo"] = b
		}
	}
	if v, ok := merged["logitBias"]; ok {
		body["logit_bias"] = v
		delete(body, "logitBias")
	}
	if v, ok := merged["suffix"]; ok {
		if s, ok := v.(string); ok && s != "" {
			body["suffix"] = s
		}
	}
	if v, ok := merged["user"]; ok {
		if s, ok := v.(string); ok && s != "" {
			body["user"] = s
		}
	}

	// Standard generate options.
	if opts.MaxOutputTokens != nil {
		body["max_tokens"] = *opts.MaxOutputTokens
	}
	if opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if opts.TopP != nil {
		body["top_p"] = *opts.TopP
	}
	if opts.FrequencyPenalty != nil {
		body["frequency_penalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		body["presence_penalty"] = *opts.PresencePenalty
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}

	body["prompt"] = prompt

	// Stop sequence order: generated stop first, then caller stops.
	stops := []string{generatedStop}
	stops = append(stops, opts.StopSequences...)
	body["stop"] = stops

	return completionRequestConfig{Body: body, Warnings: warnings}, nil
}

// flattenCompletionPrompt converts a message list to a single string prompt
// following the exact algorithm from the spec. It returns the prompt string
// and the generated stop sequence.
func flattenCompletionPrompt(messages []Message) (string, string, error) {
	const generatedStop = "\nuser:"

	var sb strings.Builder
	remaining := messages

	// Step 2: if first message is system, prepend it.
	if len(remaining) > 0 {
		if sys, ok := remaining[0].(SystemMessage); ok {
			sb.WriteString(sys.Content)
			sb.WriteString("\n\n")
			remaining = remaining[1:]
		}
	}

	// Step 3: process remaining messages.
	for _, msg := range remaining {
		switch m := msg.(type) {
		case SystemMessage:
			return "", "", InvalidPromptError{
				Message: fmt.Sprintf("unexpected system message in completion prompt: %s", m.Content),
			}
		case UserMessage:
			var textParts strings.Builder
			for _, c := range m.Content {
				if tc, ok := c.(TextContent); ok {
					textParts.WriteString(tc.Text)
				}
				// Non-text user parts are silently ignored per spec.
			}
			sb.WriteString("user:\n")
			sb.WriteString(textParts.String())
			sb.WriteString("\n\n")
		case AssistantMessage:
			var textParts strings.Builder
			for _, c := range m.Content {
				switch part := c.(type) {
				case TextContent:
					textParts.WriteString(part.Text)
				case ToolCallContent:
					return "", "", UnsupportedFunctionalityError{Functionality: "tool-call messages"}
				}
			}
			sb.WriteString("assistant:\n")
			sb.WriteString(textParts.String())
			sb.WriteString("\n\n")
		case ToolMessage:
			return "", "", UnsupportedFunctionalityError{Functionality: "tool messages"}
		default:
			return "", "", InvalidPromptError{Message: fmt.Sprintf("unsupported message type %T", msg)}
		}
	}

	// Step 4: append assistant prefix.
	sb.WriteString("assistant:\n")

	return sb.String(), generatedStop, nil
}

// DoGenerate implements LanguageModel for the completion model.
func (m *openAICompatibleCompletionLanguageModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildCompletionRequest(opts)
	if err != nil {
		return nil, err
	}
	// Do NOT apply TransformRequestBody for completion requests (per spec).
	bodyBytes, err := json.Marshal(config.Body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointCompletions, bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	result, err := m.parseCompletionResponse(resp, bodyBytes)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(config.Warnings, result.Warnings...)
	return result, nil
}

// completionResponse is the non-streaming completion API response shape.
type completionResponse struct {
	ID      string               `json:"id"`
	Created *int64               `json:"created"`
	Model   string               `json:"model"`
	Choices []completionChoice   `json:"choices"`
	Usage   json.RawMessage      `json:"usage"`
}

type completionChoice struct {
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// completionUsageShape is the JSON shape for completion token usage.
type completionUsageShape struct {
	PromptTokens     *int `json:"prompt_tokens"`
	CompletionTokens *int `json:"completion_tokens"`
	TotalTokens      *int `json:"total_tokens"`
}

func (u *completionUsageShape) toPublic(raw json.RawMessage) *OpenAICompatibleCompletionUsage {
	if u == nil {
		return nil
	}
	return &OpenAICompatibleCompletionUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		Raw:              cloneRawMessage(raw),
	}
}

func (m *openAICompatibleCompletionLanguageModel) parseCompletionResponse(resp *apiResponse, requestBody []byte) (*GenerateResult, error) {
	var decoded completionResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, InvalidResponseDataError{Message: err.Error(), Data: string(resp.Body)}
	}

	// Parse usage if present.
	var usage *OpenAICompatibleCompletionUsage
	if len(decoded.Usage) > 0 && string(decoded.Usage) != "null" {
		var usageShape completionUsageShape
		if err := json.Unmarshal(decoded.Usage, &usageShape); err != nil {
			return nil, InvalidResponseDataError{Message: err.Error(), Data: string(decoded.Usage)}
		}
		usage = usageShape.toPublic(decoded.Usage)
	}

	// Use only first choice.
	var choice completionChoice
	if len(decoded.Choices) > 0 {
		choice = decoded.Choices[0]
	}

	// Return one text content part when first choice text is non-empty.
	var content []Content
	if choice.Text != "" {
		content = append(content, TextContent{Text: choice.Text})
	}

	return &GenerateResult{
		Content:      content,
		FinishReason: finishReasonFromOpenAI(choice.FinishReason),
		Usage:        completionUsage(usage),
		Request:      RequestMetadata{Body: append([]byte(nil), requestBody...)},
		Response:     responseMetadata(decoded.ID, decoded.Model, decoded.Created, resp.Headers, resp.Body),
	}, nil
}
