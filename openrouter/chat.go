package openrouter

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type chatModel struct {
	provider *openRouterProvider
	modelID  string
	options  ChatOptions
}

func (m *chatModel) ModelID() string  { return m.modelID }
func (m *chatModel) Provider() string { return "openrouter.chat" }
func (m *chatModel) SupportURLs() map[string][]*regexp.Regexp {
	return cloneRegexpMap(chatSupportURLs)
}

func (m *chatModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	body, err := m.buildBody(opts, false)
	if err != nil {
		return nil, err
	}
	var resp chatResponse
	raw, headers, err := m.provider.postJSON(ctx, "/chat/completions", body, opts.Headers, &resp)
	if err != nil {
		return nil, err
	}
	return m.parseResponse(resp, raw, headers, body)
}

func (m *chatModel) buildBody(opts GenerateOptions, stream bool) (map[string]any, error) {
	if m.modelID == "" {
		return nil, ErrMissingModelID
	}
	messages, err := convertChatMessages(opts.Messages)
	if err != nil {
		return nil, err
	}
	body := map[string]any{"model": m.modelID, "messages": messages}
	applyChatOptions(body, m.options)
	applyGenerateOverrides(body, opts)
	if len(opts.Tools) > 0 {
		tools := make([]map[string]any, 0, len(opts.Tools))
		for _, t := range opts.Tools {
			mt, err := mapTool(t)
			if err != nil {
				return nil, err
			}
			tools = append(tools, mt)
		}
		body["tools"] = tools
	}
	if choice, err := mapToolChoice(opts.ToolChoice); err != nil {
		return nil, err
	} else if choice != nil {
		body["tool_choice"] = choice
	}
	if opts.ResponseFormat != nil {
		if rf := chatResponseFormat(*opts.ResponseFormat, m.options.StructuredOutputs); rf != nil {
			body["response_format"] = rf
		}
	}
	if stream {
		body["stream"] = true
		if m.provider.compatibility == CompatibilityStrict {
			body["stream_options"] = map[string]any{"include_usage": true}
		}
	}
	mergeBody(body, m.provider.extraBody)
	mergeBody(body, m.options.ExtraBody)
	mergeOpenRouterOptions(body, opts.ProviderOptions)
	return body, nil
}

func applyChatOptions(body map[string]any, o ChatOptions) {
	if len(o.Models) > 0 {
		body["models"] = o.Models
	}
	if len(o.LogitBias) > 0 {
		body["logit_bias"] = o.LogitBias
	}
	if o.Logprobs != nil && ((o.Logprobs.Enabled != nil && *o.Logprobs.Enabled) || o.Logprobs.Top != nil) {
		body["logprobs"] = true
		if o.Logprobs.Top != nil {
			body["top_logprobs"] = *o.Logprobs.Top
		} else {
			body["top_logprobs"] = 0
		}
	}
	if o.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *o.ParallelToolCalls
	}
	if o.User != "" {
		body["user"] = o.User
	}
	if len(o.Plugins) > 0 {
		plugins := make([]map[string]any, 0, len(o.Plugins))
		for _, p := range o.Plugins {
			mp, err := marshalPlugin(p)
			if err == nil {
				plugins = append(plugins, mp)
			}
		}
		body["plugins"] = plugins
	}
	if o.WebSearchOptions != nil {
		body["web_search_options"] = o.WebSearchOptions
	}
	if o.CacheControl != nil {
		body["cache_control"] = o.CacheControl
	}
	if o.Debug != nil {
		body["debug"] = o.Debug
	}
	if o.Provider != nil {
		body["provider"] = o.Provider
	}
	if o.IncludeReasoning != nil {
		body["include_reasoning"] = *o.IncludeReasoning
	}
	if o.Reasoning != nil {
		body["reasoning"] = o.Reasoning
	}
	if o.Usage != nil {
		body["usage"] = o.Usage
	}
	if o.Temperature != nil {
		body["temperature"] = *o.Temperature
	}
	if o.TopP != nil {
		body["top_p"] = *o.TopP
	}
	if o.TopK != nil {
		body["top_k"] = *o.TopK
	}
	if o.FrequencyPenalty != nil {
		body["frequency_penalty"] = *o.FrequencyPenalty
	}
	if o.PresencePenalty != nil {
		body["presence_penalty"] = *o.PresencePenalty
	}
	if o.MaxTokens != nil {
		body["max_tokens"] = *o.MaxTokens
	}
}

func applyGenerateOverrides(body map[string]any, opts GenerateOptions) {
	if opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if opts.TopP != nil {
		body["top_p"] = *opts.TopP
	}
	if opts.TopK != nil {
		body["top_k"] = *opts.TopK
	}
	if opts.FrequencyPenalty != nil {
		body["frequency_penalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		body["presence_penalty"] = *opts.PresencePenalty
	}
	if opts.MaxTokens != nil {
		body["max_tokens"] = *opts.MaxTokens
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if len(opts.Stop) > 0 {
		body["stop"] = opts.Stop
	}
}

func chatResponseFormat(r ResponseFormat, so *StructuredOutputsOptions) any {
	if r.Type != "json" {
		return nil
	}
	if r.Schema == nil {
		return map[string]any{"type": "json_object"}
	}
	strict := true
	if so != nil && so.Strict != nil {
		strict = *so.Strict
	}
	if r.Strict != nil {
		strict = *r.Strict
	}
	name := r.Name
	if name == "" {
		name = "response"
	}
	js := map[string]any{"schema": r.Schema, "strict": strict, "name": name}
	if r.Description != "" {
		js["description"] = r.Description
	}
	return map[string]any{"type": "json_schema", "json_schema": js}
}

type chatResponse struct {
	ID       string       `json:"id"`
	Model    string       `json:"model"`
	Provider string       `json:"provider"`
	Choices  []chatChoice `json:"choices"`
	Usage    *apiUsage    `json:"usage"`
}

type chatChoice struct {
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatMessage struct {
	Content          any               `json:"content"`
	Reasoning        string            `json:"reasoning"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details"`
	Images           []chatImage       `json:"images"`
	ToolCalls        []apiToolCall     `json:"tool_calls"`
	Annotations      []annotation      `json:"annotations"`
}

type chatImage struct {
	Type     string        `json:"type"`
	ImageURL apiURLPayload `json:"image_url"`
}

type annotation struct {
	Type         string          `json:"type"`
	URLCitation  *urlCitation    `json:"url_citation,omitempty"`
	FileCitation *FileAnnotation `json:"file_citation,omitempty"`
}

type urlCitation struct {
	URL        string `json:"url"`
	Title      string `json:"title"`
	Content    string `json:"content"`
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
}

func (m *chatModel) parseResponse(resp chatResponse, raw []byte, headers http.Header, requestBody map[string]any) (*GenerateResult, error) {
	if len(resp.Choices) == 0 {
		return nil, NoContentGeneratedError{Message: "OpenRouter returned no choices"}
	}
	choice := resp.Choices[0]
	var content []Content
	reasoning := filterResponseReasoning(choice.Message.ReasoningDetails)
	if len(reasoning) > 0 {
		for _, d := range reasoning {
			if d.Type == "reasoning.text" && d.Text != "" {
				content = append(content, ReasoningContent{Text: d.Text, ProviderMetadata: ProviderMetadata{"openrouter": map[string]any{"reasoning_details": []ReasoningDetail{d}}}})
			}
			if d.Type == "reasoning.summary" && d.Summary != "" {
				content = append(content, ReasoningContent{Text: d.Summary, ProviderMetadata: ProviderMetadata{"openrouter": map[string]any{"reasoning_details": []ReasoningDetail{d}}}})
			}
		}
	} else if choice.Message.Reasoning != "" {
		content = append(content, ReasoningContent{Text: choice.Message.Reasoning})
	}
	if text := messageText(choice.Message.Content); text != "" {
		content = append(content, TextContent{Text: text})
	}
	seenIDs := map[string]struct{}{}
	for i, tc := range choice.Message.ToolCalls {
		id := m.uniqueToolID(tc.ID, seenIDs)
		var md ProviderMetadata
		if i == 0 && len(reasoning) > 0 {
			md = ProviderMetadata{"openrouter": map[string]any{"reasoning_details": reasoning}}
		}
		content = append(content, ToolCallContent{ToolCallID: id, ToolName: tc.Function.Name, Input: jsonRawOrObject(tc.Function.Arguments), ProviderMetadata: md})
	}
	for _, img := range choice.Message.Images {
		media, data := dataURLParts(img.ImageURL.URL, "image/jpeg")
		content = append(content, FileContent{Data: data, MediaType: media})
	}
	var files []FileAnnotation
	for _, a := range choice.Message.Annotations {
		if a.URLCitation != nil {
			content = append(content, SourceContent{SourceType: "url", ID: a.URLCitation.URL, URL: a.URLCitation.URL, Title: a.URLCitation.Title, ProviderMetadata: ProviderMetadata{"openrouter": map[string]any{"content": a.URLCitation.Content, "startIndex": a.URLCitation.StartIndex, "endIndex": a.URLCitation.EndIndex}}})
		}
		if a.FileCitation != nil {
			files = append(files, *a.FileCitation)
		}
	}
	finish := mapFinishReason(choice.FinishReason, len(choice.Message.ToolCalls) > 0, hasEncrypted(reasoning))
	body, _ := json.Marshal(requestBody)
	return &GenerateResult{Content: content, FinishReason: finish, Usage: usagePtr(resp.Usage), ProviderMetadata: providerMetadata(resp.Provider, resp.Usage, reasoning, files), Request: RequestMetadata{Body: body}, Response: ResponseMetadata{ID: resp.ID, ModelID: resp.Model, Headers: headers, RawBody: raw, Timestamp: time.Now()}}, nil
}

func (m *chatModel) uniqueToolID(id string, seen map[string]struct{}) string {
	if id == "" {
		id = m.provider.nextID()
	}
	base := id
	n := 1
	for {
		if _, ok := seen[id]; !ok {
			seen[id] = struct{}{}
			return id
		}
		id = base + "-" + strconv.Itoa(n)
		n++
	}
}

func messageText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return ""
	default:
		b, _ := json.Marshal(x)
		return string(b)
	}
}

func jsonRawOrObject(s string) any {
	if s == "" {
		return map[string]any{}
	}
	var v any
	if json.Unmarshal([]byte(s), &v) == nil {
		return v
	}
	return s
}

func filterResponseReasoning(in []ReasoningDetail) []ReasoningDetail {
	return filterReasoningDetails(in, map[string]struct{}{})
}

func hasEncrypted(in []ReasoningDetail) bool {
	for _, d := range in {
		if d.Type == "reasoning.encrypted" {
			return true
		}
	}
	return false
}

func usagePtr(u *apiUsage) Usage {
	if u == nil {
		return Usage{}
	}
	return standardUsage(*u)
}

func mapFinishReason(raw string, hasTools, encrypted bool) FinishReason {
	if hasTools && encrypted && raw == "stop" {
		return FinishReasonToolCalls
	}
	switch raw {
	case "stop":
		if hasTools {
			return FinishReasonToolCalls
		}
		return FinishReasonStop
	case "length":
		return FinishReasonLength
	case "content_filter":
		return FinishReasonContentFilter
	case "function_call", "tool_calls":
		return FinishReasonToolCalls
	default:
		if hasTools {
			return FinishReasonToolCalls
		}
		return FinishReasonOther
	}
}

func dataURLParts(url, fallback string) (string, string) {
	if strings.HasPrefix(url, "data:") {
		header, data, ok := strings.Cut(url, ",")
		if ok {
			media := strings.TrimPrefix(strings.TrimSuffix(header, ";base64"), "data:")
			return media, data
		}
	}
	return fallback, url
}
