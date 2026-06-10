package cohere

import (
	"context"
	"encoding/json"
)

type chatRequestConfig struct {
	Body     map[string]any
	Warnings []Warning
}

func (m *cohereChatLanguageModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	config, err := m.buildChatRequest(opts)
	if err != nil {
		return nil, err
	}
	bodyBytes, err := json.Marshal(config.Body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointChat, bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	result, err := m.parseGenerateResponse(resp, bodyBytes)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(config.Warnings, result.Warnings...)
	return result, nil
}

func (m *cohereChatLanguageModel) buildChatRequest(opts GenerateOptions) (chatRequestConfig, error) {
	prompt, err := convertToCohereChatPrompt(opts.Messages)
	if err != nil {
		return chatRequestConfig{}, err
	}
	tools, toolChoice, toolWarnings, err := prepareTools(opts.Tools, opts.ToolChoice)
	if err != nil {
		return chatRequestConfig{}, err
	}
	lang, err := parseLanguageOptions(opts.ProviderOptions)
	if err != nil {
		return chatRequestConfig{}, err
	}
	body := map[string]any{"model": m.modelID, "messages": prompt.Messages}
	if opts.FrequencyPenalty != nil {
		body["frequency_penalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		body["presence_penalty"] = *opts.PresencePenalty
	}
	if opts.MaxOutputTokens != nil {
		body["max_tokens"] = *opts.MaxOutputTokens
	}
	if opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if opts.TopP != nil {
		body["p"] = *opts.TopP
	}
	if opts.TopK != nil {
		body["k"] = *opts.TopK
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if len(opts.StopSequences) > 0 {
		body["stop_sequences"] = append([]string(nil), opts.StopSequences...)
	}
	warnings := append([]Warning(nil), toolWarnings...)
	warnings = append(warnings, prompt.Warnings...)
	if rf, rw := cohereResponseFormat(opts); rf != nil {
		body["response_format"] = rf
		warnings = append(warnings, rw...)
	} else {
		warnings = append(warnings, rw...)
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	if toolChoice != nil {
		body["tool_choice"] = *toolChoice
	}
	if len(prompt.Documents) > 0 {
		body["documents"] = prompt.Documents
	}
	if lang.Thinking != nil {
		th := map[string]any{"type": lang.Thinking.Type}
		if lang.Thinking.Type == "" {
			th["type"] = "enabled"
		}
		if lang.Thinking.TokenBudget != nil {
			th["token_budget"] = *lang.Thinking.TokenBudget
		}
		body["thinking"] = th
	}
	return chatRequestConfig{Body: body, Warnings: warnings}, nil
}

func cohereResponseFormat(opts GenerateOptions) (map[string]any, []Warning) {
	var warnings []Warning
	var schema any
	ok := false
	if opts.StructuredOutput != nil {
		ok = true
		schema = opts.StructuredOutput.Schema
		if opts.ResponseFormat != nil {
			warnings = append(warnings, Warning{Type: "other", Message: "StructuredOutput takes precedence over ResponseFormat."})
		}
	} else if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json" {
		ok = true
		schema = opts.ResponseFormat.Schema
	}
	if !ok {
		return nil, warnings
	}
	out := map[string]any{"type": "json_object"}
	if schema != nil {
		out["json_schema"] = schema
	}
	return out, warnings
}

func (m *cohereChatLanguageModel) parseGenerateResponse(resp *apiResponse, requestBody []byte) (*GenerateResult, error) {
	var decoded cohereChatResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, InvalidResponseDataError{Message: err.Error(), Data: string(resp.Body)}
	}
	content := make([]Content, 0)
	for _, item := range decoded.Message.Content {
		if item.Type == "text" && item.Text != "" {
			content = append(content, TextContent{Text: item.Text})
		}
		if item.Type == "thinking" && item.Thinking != "" {
			content = append(content, ReasoningContent{Text: item.Thinking})
		}
	}
	for _, citation := range decoded.Message.Citations {
		content = append(content, sourceFromCitation(citation, m.provider.generateID))
	}
	for _, call := range decoded.Message.ToolCalls {
		args := call.Function.Arguments
		if args == "null" {
			args = "{}"
		}
		content = append(content, ToolCallContent{ToolCallID: call.ID, ToolName: call.Function.Name, Input: json.RawMessage(args)})
	}
	id := ""
	if decoded.GenerationID != nil {
		id = *decoded.GenerationID
	}
	return &GenerateResult{Content: content, FinishReason: FinishReason{Unified: mapCohereFinishReason(decoded.FinishReason), Raw: decoded.FinishReason}, Usage: convertCohereUsage(decoded.Usage.Tokens), Request: RequestMetadata{Body: append([]byte(nil), requestBody...)}, Response: ResponseMetadata{ID: id, Headers: cloneHeader(resp.Headers), Body: append([]byte(nil), resp.Body...)}}, nil
}

func sourceFromCitation(c cohereCitation, gen IDGenerator) SourceContent {
	id := ""
	if gen != nil {
		id = gen()
	}
	title := "Document"
	if len(c.Sources) > 0 && c.Sources[0].Document.Title != "" {
		title = c.Sources[0].Document.Title
	}
	meta := map[string]any{"start": c.Start, "end": c.End, "text": c.Text, "sources": c.Sources}
	if c.Type != nil {
		meta["citationType"] = *c.Type
	}
	return SourceContent{SourceType: "document", ID: id, MediaType: "text/plain", Title: title, ProviderMetadata: ProviderMetadata{"cohere": meta}}
}
