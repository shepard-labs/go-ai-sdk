package openai

import (
	"context"
	"encoding/json"
	"regexp"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// openaiCompletionLanguageModel implements LanguageModel for the legacy
// OpenAI text completions endpoint.
type openaiCompletionLanguageModel struct {
	provider *openaiProvider
	modelID  string
}

func newCompletionLanguageModel(p *openaiProvider, modelID string) LanguageModel {
	return &openaiCompletionLanguageModel{provider: p, modelID: modelID}
}

func (m *openaiCompletionLanguageModel) ModelID() string  { return m.modelID }
func (m *openaiCompletionLanguageModel) Provider() string { return "openai.completion" }
func (m *openaiCompletionLanguageModel) SupportURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{}
}

func (m *openaiCompletionLanguageModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	body, warnings, err := m.buildCompletionRequest(opts)
	if err != nil {
		return nil, err
	}
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointCompletions, encoded, opts.Headers)
	if err != nil {
		return nil, err
	}
	result, err := m.parseCompletionResponse(resp.Body, encoded)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(result.Warnings, warnings...)
	return result, nil
}

func (m *openaiCompletionLanguageModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	body, warnings, err := m.buildCompletionRequest(opts.GenerateOptions)
	if err != nil {
		return nil, err
	}
	body["stream"] = true
	body["stream_options"] = map[string]any{"include_usage": true}
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	reqBody := append([]byte(nil), encoded...)
	resp, err := m.provider.executeStream(ctx, endpointCompletions, encoded, opts.Headers)
	if err != nil {
		return nil, err
	}
	parts := make(chan StreamPart)
	sresp := &StreamResponse{Headers: resp.Headers.Clone()}
	result := &StreamResult{Stream: parts, Parts: parts, Request: RequestMetadata{Body: reqBody}, Response: sresp}
	go m.runCompletionStream(ctx, resp, parts, sresp, warnings)
	return result, nil
}

func (m *openaiCompletionLanguageModel) buildCompletionRequest(opts GenerateOptions) (map[string]any, []Warning, error) {
	var warnings []Warning
	body := map[string]any{"model": m.modelID}
	// Build a single prompt by concatenating messages in the
	// <system> / <user>: / <assistant>: format. Per spec:
	//   - System messages throw InvalidPromptError (no support in completions).
	//   - Tool calls and tool messages throw UnsupportedFunctionalityError.
	//   - Stop sequences automatically include "\n<user>:" unless overridden.
	var prompt string
	for _, msg := range opts.Messages {
		switch part := msg.(type) {
		case SystemMessage:
			return nil, nil, InvalidPromptError{Message: "unexpected system message in completion prompt: " + part.Content}
		case UserMessage:
			if len(part.Content) == 1 {
				if t, ok := part.Content[0].(TextContent); ok {
					prompt += "<user>:\n" + t.Text + "\n\n"
					continue
				}
			}
			// Multi-part or non-text user content isn't representable
			// in the legacy completions prompt format.
			return nil, nil, UnsupportedFunctionalityError{Functionality: "multi-part user messages in completion prompt"}
		case AssistantMessage:
			for _, c := range part.Content {
				switch cc := c.(type) {
				case TextContent:
					prompt += "<assistant>:\n" + cc.Text + "\n\n"
				case ToolCallContent:
					return nil, nil, UnsupportedFunctionalityError{Functionality: "tool calls in completion prompt"}
				}
			}
		case ToolMessage:
			return nil, nil, UnsupportedFunctionalityError{Functionality: "tool messages in completion prompt"}
		}
	}
	// Append the final <assistant>: prefix so the model continues from
	// there.
	prompt += "<assistant>:\n"
	body["prompt"] = prompt
	if opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if opts.TopP != nil {
		body["top_p"] = *opts.TopP
	}
	if opts.MaxOutputTokens != nil {
		body["max_tokens"] = *opts.MaxOutputTokens
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if opts.FrequencyPenalty != nil {
		body["frequency_penalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		body["presence_penalty"] = *opts.PresencePenalty
	}
	// Auto-generated stop sequence ("\n<user>:") is appended first,
	// then user-provided stop sequences (per spec).
	stops := []string{"\n<user>:"}
	if len(opts.StopSequences) > 0 {
		stops = append(stops, opts.StopSequences...)
	}
	body["stop"] = stops
	// Completion-specific provider options: echo, logit_bias, suffix, user.
	if v, ok := opts.ProviderOptions["openai"]; ok {
		if echo, ok := v["echo"].(bool); ok && echo {
			body["echo"] = true
		}
		if lb, ok := v["logitBias"].(map[string]any); ok {
			body["logit_bias"] = lb
		} else if lb, ok := v["logitBias"].(map[string]float64); ok {
			body["logit_bias"] = lb
		}
		if suffix, ok := v["suffix"].(string); ok && suffix != "" {
			body["suffix"] = suffix
		}
		if user, ok := v["user"].(string); ok && user != "" {
			body["user"] = user
		}
	}
	// Unsupported features per spec.
	if opts.TopK != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "topK", Message: "topK is not supported by the OpenAI completions API"})
	}
	if len(opts.Tools) > 0 {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "tools", Message: "tools are not supported by the OpenAI completions API"})
	}
	if opts.ToolChoice != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "toolChoice", Message: "toolChoice is not supported by the OpenAI completions API"})
	}
	if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json" {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "responseFormat", Message: "JSON response format is not supported."})
	}
	return body, warnings, nil
}

func (m *openaiCompletionLanguageModel) parseCompletionResponse(body, requestBody []byte) (*GenerateResult, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse: " + err.Error()}
	}
	result := &GenerateResult{
		Request:  RequestMetadata{Body: requestBody},
		Response: ResponseMetadata{Body: body, ModelID: m.modelID},
	}
	if id, ok := raw["id"].(string); ok {
		result.Response.ID = id
	}
	if fr, ok := raw["choices"].([]any); ok && len(fr) > 0 {
		if c, ok := fr[0].(map[string]any); ok {
			if reason, ok := c["finish_reason"].(string); ok {
				result.FinishReason = mapCompletionFinishReason(reason)
			}
			if text, ok := c["text"].(string); ok {
				result.Content = append(result.Content, openaicompatibleTextContent(text))
			}
			// Per spec: logprobs passed through as
			// providerMetadata["openai"].logprobs.
			if lp, ok := c["logprobs"]; ok && lp != nil {
				if result.ProviderMetadata == nil {
					result.ProviderMetadata = ProviderMetadata{}
				}
				om := map[string]any{}
				if existing, ok := result.ProviderMetadata["openai"].(map[string]any); ok {
					om = existing
				}
				om["logprobs"] = lp
				result.ProviderMetadata["openai"] = om
			}
		}
	}
	if usage, ok := raw["usage"].(map[string]any); ok {
		encoded, _ := json.Marshal(usage)
		var shape chatUsageShape
		_ = json.Unmarshal(encoded, &shape)
		result.Usage = buildChatUsage(encoded, shape)
	}
	return result, nil
}

// openaicompatibleTextContent builds an openaicompatible TextContent for
// appending to a result Content slice.
func openaicompatibleTextContent(text string) openaicompatible.TextContent {
	return openaicompatible.TextContent{Text: text}
}

func (m *openaiCompletionLanguageModel) runCompletionStream(ctx context.Context, resp *httpStreamResponse, parts chan<- StreamPart, sresp *StreamResponse, startWarnings []Warning) {
	defer close(parts)
	defer resp.Body.Close()
	parts <- StreamStart{Warnings: startWarnings}
	headers := resp.Headers.Clone()
	sresp.Headers = headers
	state := newCompletionStreamState()
	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}
		var chunk completionStreamChunk
		if err := json.Unmarshal(raw, &chunk); err != nil {
			parts <- StreamError{Err: InvalidResponseDataError{Message: err.Error()}}
			return false
		}
		if len(chunk.Choices) > 0 {
			if state.textStart == false {
				parts <- StreamTextStart{ID: "txt-0"}
				state.textStart = true
			}
			parts <- StreamTextDelta{ID: "txt-0", Text: chunk.Choices[0].Text}
		}
		if len(chunk.Usage) > 0 {
			_ = json.Unmarshal(chunk.Usage, &state.Usage)
		}
		return true
	})
	if state.textStart {
		parts <- StreamTextEnd{ID: "txt-0"}
	}
	parts <- StreamFinish{FinishReason: resultFinishReason("stop"), Usage: state.Usage}
}

type completionStreamChunk struct {
	Choices []struct {
		Text string `json:"text"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
}

type completionStreamState struct {
	textStart bool
	Usage     Usage
}

func newCompletionStreamState() *completionStreamState {
	return &completionStreamState{}
}

func mapCompletionFinishReason(reason string) FinishReason {
	switch reason {
	case "stop":
		return FinishReason{Unified: "stop", Raw: reason}
	case "length":
		return FinishReason{Unified: "length", Raw: reason}
	case "content_filter":
		return FinishReason{Unified: "content-filter", Raw: reason}
	}
	return FinishReason{Unified: "other", Raw: reason}
}

func resultFinishReason(reason string) FinishReason {
	return FinishReason{Unified: reason, Raw: reason}
}
