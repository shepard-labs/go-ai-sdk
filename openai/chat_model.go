package openai

import (
	"context"
	"regexp"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// openaiChatLanguageModel wraps the openaicompatible chat language model
// and applies OpenAI-specific request building, per-model capability
// branching, and provider-tool serialization.
type openaiChatLanguageModel struct {
	provider *openaiProvider
	modelID  string
}

func newChatLanguageModel(provider *openaiProvider, modelID string) LanguageModel {
	return &openaiChatLanguageModel{provider: provider, modelID: modelID}
}

func (m *openaiChatLanguageModel) ModelID() string  { return m.modelID }
func (m *openaiChatLanguageModel) Provider() string { return "openai.chat" }

func (m *openaiChatLanguageModel) SupportURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{}
}

// DoGenerate implements LanguageModel.
func (m *openaiChatLanguageModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	return m.doGenerateChat(ctx, opts)
}

// DoStream implements LanguageModel.
func (m *openaiChatLanguageModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	return m.doStreamChat(ctx, opts)
}

// doGenerateChat implements the non-streaming chat call.
func (m *openaiChatLanguageModel) doGenerateChat(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildChatRequest(opts)
	if err != nil {
		return nil, err
	}
	body, err := jsonMarshal(config.Body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointChatCompletions, body, opts.Headers)
	if err != nil {
		return nil, err
	}
	result, err := m.parseChatResponse(resp.Body, body, opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(config.Warnings, result.Warnings...)
	return &result, nil
}

// doStreamChat implements the streaming chat call.
func (m *openaiChatLanguageModel) doStreamChat(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildChatRequest(opts.GenerateOptions)
	if err != nil {
		return nil, err
	}
	body := config.Body
	body["stream"] = true
	body["stream_options"] = map[string]any{"include_usage": true}
	bodyBytes, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	reqBody := append([]byte(nil), bodyBytes...)
	resp, err := m.provider.executeStream(ctx, endpointChatCompletions, bodyBytes, opts.Headers)
	if err != nil {
		return nil, err
	}
	parts := make(chan StreamPart)
	sresp := &StreamResponse{Headers: resp.Headers.Clone()}
	result := &StreamResult{Stream: parts, Parts: parts, Request: RequestMetadata{Body: reqBody}, Response: sresp}
	go m.runChatStream(ctx, resp, parts, sresp, config.Warnings, opts)
	return result, nil
}

// buildChatRequest builds the OpenAI chat completion request body. The
// logic is similar to openaicompatible.buildChatRequest but applies
// OpenAI-specific per-model parameter stripping.
func (m *openaiChatLanguageModel) buildChatRequest(opts GenerateOptions) (chatRequestConfig, error) {
	caps := ModelCapabilitiesForID(m.modelID)
	providerOptions := cloneProviderOptions(opts.ProviderOptions)
	chatOptions, deprecatedWarnings := mergeChatOpenAIOptions(providerOptions)
	warnings := append([]Warning(nil), deprecatedWarnings...)
	body := map[string]any{"model": m.modelID}

	// Per-model parameter stripping.
	// Temperature is kept only when:
	//   - model is not reasoning at all, OR
	//   - model is gpt-5.1 family (SupportsNonReasoningParameters) AND reasoningEffort == "none"
	isNoneEffort := caps.SupportsNonReasoningParameters && reasoningEffortIsNone(chatOptions)
	stripTemperature := caps.StripsTemperatureAlways ||
		(caps.IsReasoningModel && !isNoneEffort)
	stripTopP := caps.IsReasoningModel && !isNoneEffort
	stripLogprobs := caps.IsReasoningModel
	stripLogitBias := caps.IsReasoningModel
	stripFreq := caps.IsReasoningModel
	stripPresence := caps.IsReasoningModel
	useMaxCompletion := caps.IsReasoningModel

	if !stripTemperature && opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if !stripTopP && opts.TopP != nil {
		body["top_p"] = *opts.TopP
	}
	if !stripFreq && opts.FrequencyPenalty != nil {
		body["frequency_penalty"] = *opts.FrequencyPenalty
	}
	if !stripPresence && opts.PresencePenalty != nil {
		body["presence_penalty"] = *opts.PresencePenalty
	}
	if !stripLogitBias {
		if v, ok := chatOptions["logitBias"]; ok {
			body["logit_bias"] = v
		}
	} else if v, ok := chatOptions["logitBias"]; ok {
		warnings = append(warnings, Warning{Type: "other", Message: "logitBias is not supported on reasoning models"})
		_ = v
	}

	// Max tokens: rename for reasoning models.
	if opts.MaxOutputTokens != nil {
		if useMaxCompletion {
			body["max_completion_tokens"] = *opts.MaxOutputTokens
		} else {
			body["max_tokens"] = *opts.MaxOutputTokens
		}
	}

	if opts.TopK != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "topK"})
	}
	if len(opts.StopSequences) > 0 {
		body["stop"] = append([]string(nil), opts.StopSequences...)
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}

	// OpenAI-specific provider options.
	if v, ok := chatOptions["user"]; ok {
		body["user"] = v
	}
	if v, ok := chatOptions["reasoningEffort"]; ok {
		body["reasoning_effort"] = v
	}
	if v, ok := chatOptions["textVerbosity"]; ok {
		body["verbosity"] = v
	}
	if v, ok := chatOptions["maxCompletionTokens"]; ok {
		body["max_completion_tokens"] = v
	}
	if v, ok := chatOptions["store"]; ok {
		body["store"] = v
	}
	if v, ok := chatOptions["metadata"]; ok {
		body["metadata"] = v
	}
	if v, ok := chatOptions["prediction"]; ok {
		body["prediction"] = v
	}
	if v, ok := chatOptions["serviceTier"]; ok {
		tier, _ := v.(string)
		if tier != "" {
			if tier == "flex" && !caps.SupportsFlexProcessing {
				warnings = append(warnings, Warning{Type: "other", Message: "service_tier \"flex\" is not supported on this model"})
			} else if tier == "priority" && !caps.SupportsPriorityProcessing {
				warnings = append(warnings, Warning{Type: "other", Message: "service_tier \"priority\" is not supported on this model"})
			} else {
				body["service_tier"] = tier
			}
		}
	}
	if v, ok := chatOptions["promptCacheKey"]; ok {
		body["prompt_cache_key"] = v
	}
	if v, ok := chatOptions["promptCacheRetention"]; ok {
		retention, _ := v.(string)
		if retention != "" && !caps.SupportsNonReasoningParameters && retention == "24h" {
			warnings = append(warnings, Warning{Type: "other", Message: "promptCacheRetention \"24h\" is only supported on gpt-5.1 family"})
		} else if retention != "" {
			body["prompt_cache_retention"] = retention
		}
	}
	if v, ok := chatOptions["safetyIdentifier"]; ok {
		body["safety_identifier"] = v
	}
	if v, ok := chatOptions["systemMessageMode"]; ok {
		// Stored for downstream message conversion.
		body["__systemMessageMode"] = v
	}
	if v, ok := chatOptions["parallelToolCalls"]; ok {
		body["parallel_tool_calls"] = v
	}
	if v, ok := chatOptions["logprobs"]; ok {
		// OpenAI chat: top_logprobs is an integer, logprobs is a bool.
		// We accept any; bool -> logprobs, int -> top_logprobs.
		if stripLogprobs {
			warnings = append(warnings, Warning{Type: "other", Message: "logprobs is not supported on reasoning models"})
		} else {
			switch t := v.(type) {
			case bool:
				if t {
					body["logprobs"] = true
				}
			case int:
				if t > 0 {
					body["logprobs"] = true
					body["top_logprobs"] = t
				}
			case int64:
				if t > 0 {
					body["logprobs"] = true
					body["top_logprobs"] = int(t)
				}
			case float64:
				if t > 0 {
					body["logprobs"] = true
					body["top_logprobs"] = int(t)
				}
			}
		}
	}

	// Response format.
	responseFormat, formatWarnings := m.chatResponseFormat(opts, chatOptions)
	warnings = append(warnings, formatWarnings...)
	if responseFormat != nil {
		body["response_format"] = responseFormat
	}

	// Messages.
	messages, err := m.convertChatMessages(opts.Messages, chatOptions, body)
	if err != nil {
		return chatRequestConfig{}, err
	}
	delete(body, "__systemMessageMode")
	body["messages"] = messages

	// Tools.
	toolsIn := make([]Tool, 0, len(opts.Tools))
	for _, t := range opts.Tools {
		toolsIn = append(toolsIn, Tool{
			Type:        t.Type,
			ID:          t.ID,
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
			Strict:      t.Strict,
			Args:        t.Args,
		})
	}
	tools, toolWarnings, err := m.convertChatTools(toolsIn)
	if err != nil {
		return chatRequestConfig{}, err
	}
	warnings = append(warnings, toolWarnings...)
	if len(tools) > 0 {
		body["tools"] = tools
		toolChoice, err := m.convertChatToolChoice(opts.ToolChoice, toolsIn)
		if err != nil {
			return chatRequestConfig{}, err
		}
		if toolChoice != nil {
			body["tool_choice"] = toolChoice
		}
	}

	return chatRequestConfig{Body: body, Warnings: warnings}, nil
}

type chatRequestConfig struct {
	Body     map[string]any
	Warnings []Warning
}

// chatResponseFormat builds the response_format payload for chat.
func (m *openaiChatLanguageModel) chatResponseFormat(opts GenerateOptions, chatOptions map[string]any) (map[string]any, []Warning) {
	var warnings []Warning
	var schema any
	name := ""
	description := ""
	isJSON := false
	if opts.StructuredOutput != nil {
		isJSON = true
		schema = opts.StructuredOutput.Schema
		name = opts.StructuredOutput.Name
		description = opts.StructuredOutput.Description
		if opts.ResponseFormat != nil {
			warnings = append(warnings, Warning{Type: "other", Message: "StructuredOutput takes precedence over ResponseFormat."})
		}
	} else if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json" {
		isJSON = true
		schema = opts.ResponseFormat.Schema
		name = opts.ResponseFormat.Name
		description = opts.ResponseFormat.Description
	}
	if !isJSON {
		return nil, warnings
	}
	if schema != nil {
		if name == "" {
			name = "response"
		}
		strict := true
		if v, ok := chatOptions["strictJsonSchema"]; ok {
			if b, ok := v.(bool); ok {
				strict = b
			}
		}
		jsonSchema := map[string]any{"schema": schema, "strict": strict, "name": name}
		if description != "" {
			jsonSchema["description"] = description
		}
		return map[string]any{"type": "json_schema", "json_schema": jsonSchema}, warnings
	}
	return map[string]any{"type": "json_object"}, warnings
}

// reasoningEffortIsNone returns true when the merged chat provider options
// include reasoningEffort == "none".
func reasoningEffortIsNone(chatOptions map[string]any) bool {
	effort, ok := chatOptions["reasoningEffort"].(string)
	return ok && effort == "none"
}

// mergeChatOpenAIOptions returns a merged map of OpenAI chat provider
// options from the caller-supplied ProviderOptions.
func mergeChatOpenAIOptions(opts ProviderOptions) (map[string]any, []Warning) {
	merged := map[string]any{}
	var warnings []Warning
	for _, key := range []string{"openai", "openai-compatible", "openaiCompatible"} {
		if _, ok := opts[key]; !ok {
			continue
		}
		if key == "openai-compatible" {
			warnings = append(warnings, Warning{Type: "other", Message: "The 'openai-compatible' key in providerOptions is deprecated. Use 'openai' instead."})
		}
		for k, v := range opts[key] {
			merged[k] = v
		}
	}
	return merged, warnings
}

// convertChatMessages, convertChatTools, convertChatToolChoice, parseChatResponse,
// runChatStream are implemented in chat_messages.go, chat_tools.go, chat_stream.go.
func init() {
	// Surface a panic if the openaicompatible package is missing the
	// message types the chat model depends on.
	_ = (openaicompatible.Message)(nil)
	_ = regexp.MustCompile("")
}
