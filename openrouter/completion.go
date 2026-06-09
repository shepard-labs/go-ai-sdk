package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"
)

type completionModel struct {
	provider *openRouterProvider
	modelID  string
	options  CompletionOptions
}

func (m *completionModel) ModelID() string  { return m.modelID }
func (m *completionModel) Provider() string { return "openrouter.completion" }
func (m *completionModel) SupportURLs() map[string][]*regexp.Regexp {
	return cloneRegexpMap(completionSupportURLs)
}

func (m *completionModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	body, err := m.buildBody(opts, false)
	if err != nil {
		return nil, err
	}
	var resp completionResponse
	raw, h, err := m.provider.postJSON(ctx, "/completions", body, opts.Headers, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, NoContentGeneratedError{Message: "OpenRouter returned no completion choices"}
	}
	b, _ := json.Marshal(body)
	return &GenerateResult{Content: []Content{TextContent{Text: resp.Choices[0].Text}}, FinishReason: mapFinishReason(resp.Choices[0].FinishReason, false, false), Usage: usagePtr(resp.Usage), ProviderMetadata: providerMetadata(resp.Provider, resp.Usage, nil, nil), Request: RequestMetadata{Body: b}, Response: ResponseMetadata{ID: resp.ID, ModelID: resp.Model, Headers: h, RawBody: raw}}, nil
}
func (m *completionModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	body, err := m.buildBody(opts.GenerateOptions, true)
	if err != nil {
		return nil, err
	}
	rb, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.provider.baseURL+"/completions", bytes.NewReader(rb))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	mergeHeader(req.Header, m.provider.requestHeaders(opts.Headers))
	resp, err := m.provider.fetch.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := readLimited(resp.Body, m.provider.maxErrorResponseBytes)
		_ = resp.Body.Close()
		return nil, parseAPIError(resp.StatusCode, raw)
	}
	ch := make(chan StreamPart, 32)
	go m.consumeCompletionStream(resp.Body, ch, opts.IncludeRawChunks)
	return &StreamResult{Stream: ch, Parts: ch, Request: RequestMetadata{Body: rb}, Response: &StreamResponse{Headers: resp.Header}}, nil
}
func (m *completionModel) buildBody(opts GenerateOptions, stream bool) (map[string]any, error) {
	prompt, err := completionPrompt(opts.Messages)
	if err != nil {
		return nil, err
	}
	body := map[string]any{"model": m.modelID, "prompt": prompt}
	if len(m.options.Models) > 0 {
		body["models"] = m.options.Models
	}
	if m.options.Suffix != "" {
		body["suffix"] = m.options.Suffix
	}
	if m.options.User != "" {
		body["user"] = m.options.User
	}
	if m.options.IncludeReasoning != nil {
		body["include_reasoning"] = *m.options.IncludeReasoning
	}
	if m.options.Reasoning != nil {
		body["reasoning"] = m.options.Reasoning
	}
	applyGenerateOverrides(body, opts)
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
func completionPrompt(messages []Message) (string, error) {
	var b strings.Builder
	for i, msg := range messages {
		switch m := msg.(type) {
		case SystemMessage:
			if i != 0 {
				return "", InvalidPromptError{Message: "system messages are only supported at the beginning"}
			}
			b.WriteString(m.Content)
			if len(messages) > 1 {
				b.WriteString("\n")
			}
		case UserMessage:
			text, err := textOnlyUser(m.Content)
			if err != nil {
				return "", err
			}
			if len(messages) == 1 {
				return text, nil
			}
			b.WriteString("user: " + text + "\n")
		case AssistantMessage:
			text := assistantText(m.Content)
			if text == "" {
				return "", InvalidPromptError{Message: "completion prompts do not support assistant non-text content"}
			}
			b.WriteString("assistant: " + text + "\n")
		default:
			return "", InvalidPromptError{Message: "completion prompt form is unsupported"}
		}
	}
	return strings.TrimSuffix(b.String(), "\n"), nil
}
func textOnlyUser(in []UserContent) (string, error) {
	var b strings.Builder
	for _, c := range in {
		t, ok := c.(TextContent)
		if !ok {
			return "", InvalidPromptError{Message: "completion prompts do not support files or tool results"}
		}
		b.WriteString(t.Text)
	}
	return b.String(), nil
}
func assistantText(in []AssistantContent) string {
	var b strings.Builder
	for _, c := range in {
		t, ok := c.(TextContent)
		if !ok {
			return ""
		}
		b.WriteString(t.Text)
	}
	return b.String()
}

type completionResponse struct {
	ID       string `json:"id"`
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Choices  []struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *apiUsage `json:"usage"`
}

func (m *completionModel) consumeCompletionStream(r io.ReadCloser, ch chan<- StreamPart, raw bool) {
	defer close(ch)
	defer r.Close()
	var usage *apiUsage
	finish := FinishReasonOther
	id := m.provider.nextID()
	parseSSE(r, func(ev sseEvent) error {
		if ev.Data == "[DONE]" {
			return nil
		}
		if raw {
			ch <- StreamRaw{Chunk: ev.Data}
		}
		var c completionChunk
		if err := json.Unmarshal([]byte(ev.Data), &c); err != nil {
			ch <- StreamError{Err: err}
			finish = FinishReasonError
			return nil
		}
		if c.Error != nil {
			ch <- StreamError{Err: &APICallError{StatusCode: 200, Message: c.Error.Message}}
			finish = FinishReasonError
			return nil
		}
		if c.Usage != nil {
			usage = c.Usage
		}
		if len(c.Choices) > 0 {
			if c.Choices[0].Text != "" {
				ch <- StreamTextStart{ID: id}
				ch <- StreamTextDelta{ID: id, Delta: c.Choices[0].Text}
			}
			if c.Choices[0].FinishReason != "" {
				finish = mapFinishReason(c.Choices[0].FinishReason, false, false)
			}
		}
		return nil
	})
	ch <- StreamTextEnd{ID: id}
	ch <- StreamFinish{FinishReason: finish, Usage: usagePtr(usage), ProviderMetadata: providerMetadata("", usage, nil, nil)}
}

type completionChunk struct {
	Choices []struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *apiUsage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}
