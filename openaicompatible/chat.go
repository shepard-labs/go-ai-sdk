package openaicompatible

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
)

var chatRecognizedOptions = map[string]struct{}{
	"user":             {},
	"reasoningEffort":  {},
	"textVerbosity":    {},
	"strictJsonSchema": {},
}

type chatRequestConfig struct {
	Body     map[string]any
	Warnings []Warning
}

func (m *openAICompatibleChatLanguageModel) doGenerateChat(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildChatRequest(opts)
	if err != nil {
		return nil, err
	}
	body := config.Body
	if m.provider.transformRequestBody != nil {
		body = m.provider.transformRequestBody(body)
		if body == nil {
			body = map[string]any{}
		}
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointChatCompletions, bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	result, err := m.parseChatResponse(resp, bodyBytes, opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(config.Warnings, result.Warnings...)
	return result, nil
}

func (m *openAICompatibleChatLanguageModel) buildChatRequest(opts GenerateOptions) (chatRequestConfig, error) {
	providerOptions := cloneProviderOptions(opts.ProviderOptions)
	chatOptions, deprecatedWarnings := mergeChatProviderOptions(m.provider.name, providerOptions)
	passthrough := chatPassthroughOptions(m.provider.name, providerOptions)
	warnings := append([]Warning(nil), deprecatedWarnings...)
	body := map[string]any{"model": m.modelID}
	for k, v := range passthrough {
		body[k] = v
	}
	if value, ok := stringOption(chatOptions, "user"); ok {
		body["user"] = value
	}
	if opts.MaxOutputTokens != nil {
		body["max_tokens"] = *opts.MaxOutputTokens
	}
	if opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if opts.TopK != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "topK"})
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
	if len(opts.StopSequences) > 0 {
		body["stop"] = append([]string(nil), opts.StopSequences...)
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if value, ok := stringOption(chatOptions, "reasoningEffort"); ok {
		body["reasoning_effort"] = value
	}
	if value, ok := stringOption(chatOptions, "textVerbosity"); ok {
		body["verbosity"] = value
	}
	responseFormat, formatWarnings := m.chatResponseFormat(opts, chatOptions)
	warnings = append(warnings, formatWarnings...)
	if responseFormat != nil {
		body["response_format"] = responseFormat
	}
	messages, err := convertChatMessages(opts.Messages)
	if err != nil {
		return chatRequestConfig{}, err
	}
	body["messages"] = messages
	tools, toolWarnings := convertChatTools(opts.Tools)
	warnings = append(warnings, toolWarnings...)
	if len(tools) > 0 {
		body["tools"] = tools
		toolChoice, err := convertChatToolChoice(opts.ToolChoice)
		if err != nil {
			return chatRequestConfig{}, err
		}
		if toolChoice != nil {
			body["tool_choice"] = toolChoice
		}
	}
	return chatRequestConfig{Body: body, Warnings: warnings}, nil
}

func chatPassthroughOptions(name string, opts ProviderOptions) map[string]any {
	out := map[string]any{}
	merge := func(key string) {
		for k, v := range opts[key] {
			if _, recognized := chatRecognizedOptions[k]; recognized {
				continue
			}
			out[k] = v
		}
	}
	merge(name)
	merge(toCamelCase(name))
	return out
}

func stringOption(opts map[string]any, key string) (string, bool) {
	value, ok := opts[key]
	if !ok {
		return "", false
	}
	s, ok := value.(string)
	return s, ok && s != ""
}

func boolOption(opts map[string]any, key string, fallback bool) bool {
	value, ok := opts[key]
	if !ok {
		return fallback
	}
	if b, ok := value.(bool); ok {
		return b
	}
	if b, ok := value.(*bool); ok && b != nil {
		return *b
	}
	return fallback
}

func (m *openAICompatibleChatLanguageModel) chatResponseFormat(opts GenerateOptions, chatOptions map[string]any) (map[string]any, []Warning) {
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
	if schema != nil && m.provider.supportsStructuredOutputs {
		if name == "" {
			name = "response"
		}
		jsonSchema := map[string]any{"schema": schema, "strict": boolOption(chatOptions, "strictJsonSchema", true), "name": name}
		if description != "" {
			jsonSchema["description"] = description
		}
		return map[string]any{"type": "json_schema", "json_schema": jsonSchema}, warnings
	}
	if schema != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "responseFormat", Details: "JSON response format schema is only supported with structuredOutputs"})
	}
	return map[string]any{"type": "json_object"}, warnings
}

func convertChatMessages(messages []Message) ([]map[string]any, error) {
	converted := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		switch msg := message.(type) {
		case SystemMessage:
			out := providerMetadataForPrompt(msg.ProviderOptions)
			out["role"] = "system"
			out["content"] = msg.Content
			converted = append(converted, out)
		case UserMessage:
			out, err := convertUserMessage(msg)
			if err != nil {
				return nil, err
			}
			converted = append(converted, out)
		case AssistantMessage:
			out, err := convertAssistantMessage(msg)
			if err != nil {
				return nil, err
			}
			converted = append(converted, out)
		case ToolMessage:
			outs, err := convertToolMessage(msg)
			if err != nil {
				return nil, err
			}
			converted = append(converted, outs...)
		default:
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported message type %T", message)}
		}
	}
	return converted, nil
}

func providerMetadataForPrompt(metadata ProviderMetadata) map[string]any {
	out := map[string]any{}
	if metadata == nil {
		return out
	}
	if values, ok := metadata["openaiCompatible"].(map[string]any); ok {
		for k, v := range values {
			out[k] = v
		}
	}
	return out
}

func convertUserMessage(msg UserMessage) (map[string]any, error) {
	if len(msg.Content) == 1 {
		if text, ok := msg.Content[0].(TextContent); ok {
			out := providerMetadataForPrompt(text.ProviderOptions)
			out["role"] = "user"
			out["content"] = text.Text
			return out, nil
		}
	}
	out := providerMetadataForPrompt(msg.ProviderOptions)
	out["role"] = "user"
	parts := make([]map[string]any, 0, len(msg.Content))
	for _, content := range msg.Content {
		part, err := convertUserContent(content)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	out["content"] = parts
	return out, nil
}

func convertUserContent(content UserContent) (map[string]any, error) {
	switch part := content.(type) {
	case TextContent:
		out := providerMetadataForPrompt(part.ProviderOptions)
		out["type"] = "text"
		out["text"] = part.Text
		return out, nil
	case FileContent:
		return convertUserFileContent(part)
	default:
		return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported user content type %T", content)}
	}
}

func convertUserFileContent(part FileContent) (map[string]any, error) {
	metadata := providerMetadataForPrompt(part.ProviderOptions)
	mediaType := part.MediaType
	switch {
	case strings.HasPrefix(mediaType, "image/"):
		urlValue, err := fileDataURL(part.Data, mediaType)
		if err != nil {
			return nil, err
		}
		metadata["type"] = "image_url"
		metadata["image_url"] = map[string]any{"url": urlValue}
		return metadata, nil
	case strings.HasPrefix(mediaType, "audio/"):
		if isURLData(part.Data) {
			return nil, UnsupportedFunctionalityError{Functionality: "audio file parts with URLs"}
		}
		format := ""
		switch mediaType {
		case "audio/wav":
			format = "wav"
		case "audio/mp3", "audio/mpeg":
			format = "mp3"
		default:
			return nil, UnsupportedFunctionalityError{Functionality: "audio media type " + mediaType}
		}
		data, err := base64Data(part.Data)
		if err != nil {
			return nil, err
		}
		metadata["type"] = "input_audio"
		metadata["input_audio"] = map[string]any{"data": data, "format": format}
		return metadata, nil
	case mediaType == "application/pdf":
		if isURLData(part.Data) {
			return nil, UnsupportedFunctionalityError{Functionality: "PDF file parts with URLs"}
		}
		data, err := base64Data(part.Data)
		if err != nil {
			return nil, err
		}
		filename := part.Filename
		if filename == "" {
			filename = "document.pdf"
		}
		metadata["type"] = "file"
		metadata["file"] = map[string]any{"filename": filename, "file_data": "data:application/pdf;base64," + data}
		return metadata, nil
	case strings.HasPrefix(mediaType, "text/"):
		text, err := textFileData(part.Data)
		if err != nil {
			return nil, err
		}
		metadata["type"] = "text"
		metadata["text"] = text
		return metadata, nil
	default:
		return nil, UnsupportedFunctionalityError{Functionality: "file part media type " + mediaType}
	}
}

func isURLData(data any) bool {
	_, ok := data.(*url.URL)
	return ok
}

func fileDataURL(data any, mediaType string) (string, error) {
	if u, ok := data.(*url.URL); ok {
		return u.String(), nil
	}
	if mediaType == "image/*" {
		mediaType = "image/jpeg"
	}
	encoded, err := base64Data(data)
	if err != nil {
		return "", err
	}
	return "data:" + mediaType + ";base64," + encoded, nil
}

func base64Data(data any) (string, error) {
	switch v := data.(type) {
	case []byte:
		return base64.StdEncoding.EncodeToString(v), nil
	case string:
		return v, nil
	default:
		return "", InvalidPromptError{Message: fmt.Sprintf("unsupported file data type %T", data)}
	}
}

func textFileData(data any) (string, error) {
	switch v := data.(type) {
	case *url.URL:
		return v.String(), nil
	case []byte:
		return string(v), nil
	case string:
		decoded, err := base64.StdEncoding.DecodeString(v)
		if err != nil {
			return "", err
		}
		return string(decoded), nil
	default:
		return "", InvalidPromptError{Message: fmt.Sprintf("unsupported file data type %T", data)}
	}
}

func convertAssistantMessage(msg AssistantMessage) (map[string]any, error) {
	out := providerMetadataForPrompt(msg.ProviderOptions)
	out["role"] = "assistant"
	var text strings.Builder
	var reasoning strings.Builder
	var toolCalls []map[string]any
	for _, content := range msg.Content {
		switch part := content.(type) {
		case TextContent:
			text.WriteString(part.Text)
		case ReasoningContent:
			reasoning.WriteString(part.Text)
		case ToolCallContent:
			toolCalls = append(toolCalls, convertAssistantToolCall(part))
		default:
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported assistant content type %T", content)}
		}
	}
	if len(toolCalls) > 0 {
		if text.Len() == 0 {
			out["content"] = nil
		} else {
			out["content"] = text.String()
		}
		out["tool_calls"] = toolCalls
	} else {
		out["content"] = text.String()
	}
	if reasoning.Len() > 0 {
		out["reasoning_content"] = reasoning.String()
	}
	return out, nil
}

func convertAssistantToolCall(part ToolCallContent) map[string]any {
	out := providerMetadataForPrompt(part.ProviderOptions)
	out["id"] = part.ToolCallID
	out["type"] = "function"
	out["function"] = map[string]any{"name": part.ToolName, "arguments": string(part.Input)}
	if google, ok := part.ProviderOptions["google"].(map[string]any); ok {
		if signature, exists := google["thoughtSignature"]; exists {
			out["extra_content"] = map[string]any{"google": map[string]any{"thought_signature": fmt.Sprint(signature)}}
		}
	}
	return out
}

func convertToolMessage(msg ToolMessage) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(msg.Content))
	for _, content := range msg.Content {
		part, ok := content.(ToolResultContent)
		if !ok {
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported tool content type %T", content)}
		}
		if part.Output.Type == "tool-approval-response" {
			continue
		}
		value, err := convertToolResultOutput(part.Output)
		if err != nil {
			return nil, err
		}
		message := providerMetadataForPrompt(part.ProviderOptions)
		message["role"] = "tool"
		message["tool_call_id"] = part.ToolCallID
		message["content"] = value
		out = append(out, message)
	}
	return out, nil
}

func convertToolResultOutput(output ToolResultOutput) (string, error) {
	switch output.Type {
	case "text", "error-text":
		if value, ok := output.Value.(string); ok {
			return value, nil
		}
		return fmt.Sprint(output.Value), nil
	case "execution-denied":
		if output.Reason != "" {
			return output.Reason, nil
		}
		return "Tool execution denied.", nil
	case "content", "json", "error-json":
		bytes, err := json.Marshal(output.Value)
		return string(bytes), err
	default:
		return "", InvalidPromptError{Message: "unsupported tool result output type " + output.Type}
	}
}

func convertChatTools(tools []Tool) ([]map[string]any, []Warning) {
	converted := make([]map[string]any, 0, len(tools))
	var warnings []Warning
	for _, tool := range tools {
		switch tool.Type {
		case "function", "":
			function := map[string]any{"name": tool.Name, "parameters": tool.InputSchema}
			if tool.Description != "" {
				function["description"] = tool.Description
			}
			if tool.Strict != nil {
				function["strict"] = *tool.Strict
			}
			converted = append(converted, map[string]any{"type": "function", "function": function})
		case "provider":
			warnings = append(warnings, Warning{Type: "unsupported", Feature: "provider-defined tool " + tool.ID})
		}
	}
	return converted, warnings
}

func convertChatToolChoice(choice *ToolChoice) (any, error) {
	if choice == nil {
		return nil, nil
	}
	switch choice.Type {
	case "auto", "none", "required":
		return choice.Type, nil
	case "tool":
		return map[string]any{"type": "function", "function": map[string]any{"name": choice.ToolName}}, nil
	default:
		return nil, UnsupportedFunctionalityError{Functionality: "tool choice type: " + choice.Type}
	}
}

func (m *openAICompatibleChatLanguageModel) parseChatResponse(resp *apiResponse, requestBody []byte, providerOptions ProviderOptions) (*GenerateResult, error) {
	var decoded chatCompletionResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, InvalidResponseDataError{Message: err.Error(), Data: string(resp.Body)}
	}
	var decodedMap map[string]any
	if err := json.Unmarshal(resp.Body, &decodedMap); err != nil {
		decodedMap = map[string]any{}
	}
	var usage *OpenAICompatibleTokenUsage
	if len(decoded.Usage) > 0 && string(decoded.Usage) != "null" {
		var usageShape chatUsageShape
		if err := json.Unmarshal(decoded.Usage, &usageShape); err != nil {
			return nil, InvalidResponseDataError{Message: err.Error(), Data: string(decoded.Usage)}
		}
		usage = usageShape.toPublic()
		usage.Raw = cloneRawMessage(decoded.Usage)
	}
	metadata := predictionTokenMetadata(m.provider.name, providerOptions, usage)
	if m.provider.metadataExtractor != nil {
		extracted, err := m.provider.metadataExtractor.ExtractMetadata(append([]byte(nil), resp.Body...), decodedMap)
		if err != nil {
			return nil, err
		}
		mergeProviderMetadata(metadata, extracted)
	}
	choice := chatChoice{}
	if len(decoded.Choices) > 0 {
		choice = decoded.Choices[0]
	}
	content := chatResponseContent(choice.Message, metadataKeyForProviderOptions(m.provider.name, providerOptions), m.provider.generateID)
	convertedUsage := defaultChatUsage(usage)
	if usage != nil && m.provider.convertUsage != nil {
		convertedUsage = m.provider.convertUsage(*usage)
	}
	return &GenerateResult{
		Content:          content,
		FinishReason:     finishReasonFromOpenAI(choice.FinishReason),
		Usage:            convertedUsage,
		ProviderMetadata: metadata,
		Request:          RequestMetadata{Body: append([]byte(nil), requestBody...)},
		Response:         responseMetadata(decoded.ID, decoded.Model, decoded.Created, resp.Headers, resp.Body),
	}, nil
}

func chatResponseContent(message chatResponseMessage, metadataKey string, generateID IDGenerator) []Content {
	var content []Content
	if message.Content != "" {
		content = append(content, TextContent{Text: message.Content})
	}
	reasoning := message.ReasoningContent
	if reasoning == "" {
		reasoning = message.Reasoning
	}
	if reasoning != "" {
		content = append(content, ReasoningContent{Text: reasoning})
	}
	for _, toolCall := range message.ToolCalls {
		id := toolCall.ID
		if id == "" && generateID != nil {
			id = generateID()
		}
		providerMetadata := ProviderMetadata(nil)
		if toolCall.ExtraContent.Google.ThoughtSignature != nil {
			providerMetadata = ProviderMetadata{metadataKey: map[string]any{"thoughtSignature": *toolCall.ExtraContent.Google.ThoughtSignature}}
		}
		content = append(content, ToolCallContent{ToolCallID: id, ToolName: toolCall.Function.Name, Input: json.RawMessage(toolCall.Function.Arguments), ProviderMetadata: providerMetadata})
	}
	return content
}

func mergeProviderMetadata(dst, src ProviderMetadata) {
	if src == nil {
		return
	}
	for k, v := range src {
		if dstInner, ok := dst[k].(map[string]any); ok {
			if srcInner, ok := v.(map[string]any); ok {
				for innerK, innerV := range srcInner {
					dstInner[innerK] = innerV
				}
				continue
			}
		}
		dst[k] = v
	}
}

type chatCompletionResponse struct {
	ID      string          `json:"id"`
	Created *int64          `json:"created"`
	Model   string          `json:"model"`
	Choices []chatChoice    `json:"choices"`
	Usage   json.RawMessage `json:"usage"`
}

type chatUsageShape struct {
	PromptTokens        *int `json:"prompt_tokens"`
	CompletionTokens    *int `json:"completion_tokens"`
	TotalTokens         *int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens *int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens          *int `json:"reasoning_tokens"`
		AcceptedPredictionTokens *int `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens *int `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
}

func (u *chatUsageShape) toPublic() *OpenAICompatibleTokenUsage {
	if u == nil {
		return nil
	}
	out := &OpenAICompatibleTokenUsage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.TotalTokens}
	if u.PromptTokensDetails != nil {
		out.PromptTokensDetails = &struct{ CachedTokens *int }{CachedTokens: u.PromptTokensDetails.CachedTokens}
	}
	if u.CompletionTokensDetails != nil {
		out.CompletionTokensDetails = &struct {
			ReasoningTokens          *int
			AcceptedPredictionTokens *int
			RejectedPredictionTokens *int
		}{ReasoningTokens: u.CompletionTokensDetails.ReasoningTokens, AcceptedPredictionTokens: u.CompletionTokensDetails.AcceptedPredictionTokens, RejectedPredictionTokens: u.CompletionTokensDetails.RejectedPredictionTokens}
	}
	return out
}

type chatChoice struct {
	Message      chatResponseMessage `json:"message"`
	FinishReason string              `json:"finish_reason"`
}

type chatResponseMessage struct {
	Content          string              `json:"content"`
	ReasoningContent string              `json:"reasoning_content"`
	Reasoning        string              `json:"reasoning"`
	ToolCalls        []chatToolCallShape `json:"tool_calls"`
}

type chatToolCallShape struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
	ExtraContent struct {
		Google struct {
			ThoughtSignature *string `json:"thought_signature"`
		} `json:"google"`
	} `json:"extra_content"`
}
