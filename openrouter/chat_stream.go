package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
)

func (m *chatModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	body, err := m.buildBody(opts.GenerateOptions, true)
	if err != nil {
		return nil, err
	}
	requestBody, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.provider.baseURL+"/chat/completions", bytes.NewReader(requestBody))
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
	go m.consumeChatStream(resp.Body, ch, opts.IncludeRawChunks)
	return &StreamResult{Stream: ch, Parts: ch, Request: RequestMetadata{Body: requestBody}, Response: &StreamResponse{Headers: resp.Header}}, nil
}

func (m *chatModel) consumeChatStream(r io.ReadCloser, ch chan<- StreamPart, raw bool) {
	defer close(ch)
	defer r.Close()
	var usage *apiUsage
	var provider string
	var finish FinishReason = FinishReasonOther
	reasoning := []ReasoningDetail{}
	textID := ""
	textStarted := false
	err := parseSSE(r, func(ev sseEvent) error {
		if ev.Data == "[DONE]" {
			return nil
		}
		if raw {
			ch <- StreamRaw{Chunk: ev.Data}
		}
		var c chatChunk
		if err := json.Unmarshal([]byte(ev.Data), &c); err != nil {
			ch <- StreamError{Err: InvalidResponseDataError{Message: err.Error()}}
			finish = FinishReasonError
			return nil
		}
		if c.Error != nil {
			ch <- StreamError{Err: &APICallError{StatusCode: 200, Message: c.Error.Message, Code: c.Error.Code, Type: c.Error.Type}}
			finish = FinishReasonError
			return nil
		}
		if c.ID != "" || c.Model != "" {
			ch <- StreamResponseMetadata{ID: c.ID, ModelID: c.Model}
		}
		if c.Provider != "" {
			provider = c.Provider
		}
		if c.Usage != nil {
			usage = c.Usage
		}
		if len(c.Choices) == 0 {
			return nil
		}
		d := c.Choices[0].Delta
		if len(d.ReasoningDetails) > 0 {
			reasoning = append(reasoning, filterResponseReasoning(d.ReasoningDetails)...)
		}
		if d.Reasoning != "" && !textStarted {
			if textID == "" {
				textID = m.provider.nextID()
				ch <- StreamReasoningStart{ID: textID}
			}
			ch <- StreamReasoningDelta{ID: textID, Delta: d.Reasoning}
		}
		if d.Content != "" {
			if !textStarted {
				if textID != "" {
					ch <- StreamReasoningEnd{ID: textID, ProviderMetadata: ProviderMetadata{"openrouter": map[string]any{"reasoning_details": reasoning}}}
				}
				textStarted = true
				if textID == "" {
					textID = c.ID
					if textID == "" {
						textID = m.provider.nextID()
					}
				}
				ch <- StreamTextStart{ID: textID}
			}
			ch <- StreamTextDelta{ID: textID, Delta: d.Content}
		}
		for _, img := range d.Images {
			media, data := dataURLParts(img.ImageURL.URL, "image/jpeg")
			ch <- StreamFile{MediaType: media, Data: data}
		}
		for _, tc := range d.ToolCalls {
			ch <- StreamToolInputStart{ID: tc.ID, ToolName: tc.Function.Name}
			ch <- StreamToolInputDelta{ID: tc.ID, Delta: tc.Function.Arguments}
			ch <- StreamToolInputEnd{ID: tc.ID}
			ch <- StreamToolCall{ToolCallContent: ToolCallContent{ToolCallID: tc.ID, ToolName: tc.Function.Name, Input: jsonRawOrObject(tc.Function.Arguments)}}
		}
		if c.Choices[0].FinishReason != "" {
			finish = mapFinishReason(c.Choices[0].FinishReason, len(d.ToolCalls) > 0, hasEncrypted(reasoning))
		}
		return nil
	})
	if err != nil {
		ch <- StreamError{Err: err}
		finish = FinishReasonError
	}
	if textStarted {
		ch <- StreamTextEnd{ID: textID}
	} else if textID != "" {
		ch <- StreamReasoningEnd{ID: textID, ProviderMetadata: ProviderMetadata{"openrouter": map[string]any{"reasoning_details": reasoning}}}
	}
	ch <- StreamFinish{FinishReason: finish, Usage: usagePtr(usage), ProviderMetadata: providerMetadata(provider, usage, reasoning, nil)}
}

type chatChunk struct {
	ID       string            `json:"id"`
	Model    string            `json:"model"`
	Provider string            `json:"provider"`
	Choices  []chatChunkChoice `json:"choices"`
	Usage    *apiUsage         `json:"usage"`
	Error    *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
		Type    string `json:"type"`
	} `json:"error"`
}
type chatChunkChoice struct {
	Delta        chatDelta `json:"delta"`
	FinishReason string    `json:"finish_reason"`
}
type chatDelta struct {
	Content          string            `json:"content"`
	Reasoning        string            `json:"reasoning"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details"`
	ToolCalls        []apiToolCall     `json:"tool_calls"`
	Images           []chatImage       `json:"images"`
}
