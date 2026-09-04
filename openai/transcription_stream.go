package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"strings"
)

// DoStream performs a streaming transcription call. The request is a
// multipart POST (mirroring DoGenerate) with stream enabled; SSE events
// carrying partial transcript deltas become StreamTextDelta parts, and
// stream close becomes StreamTextEnd + StreamFinish. Malformed events
// become a StreamError part and terminate the stream.
func (m *openaiTranscriptionModel) DoStream(ctx context.Context, opts TranscriptionOptions) (*StreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	warnings := []Warning{}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	build := func(w *multipart.Writer) error {
		if err := w.WriteField("model", m.modelID); err != nil {
			return err
		}
		if err := w.WriteField("response_format", responseFormatForModel(m.modelID)); err != nil {
			return err
		}
		if opts.Language != nil {
			if err := w.WriteField("language", *opts.Language); err != nil {
				return err
			}
		}
		if opts.Prompt != nil {
			if err := w.WriteField("prompt", *opts.Prompt); err != nil {
				return err
			}
		}
		if opts.Temperature != nil {
			if err := w.WriteField("temperature", jsonFloat(*opts.Temperature)); err != nil {
				return err
			}
		}
		if opts.ChunkingStrategy != nil {
			if err := w.WriteField("chunking_strategy", *opts.ChunkingStrategy); err != nil {
				return err
			}
		}
		if err := w.WriteField("stream", "true"); err != nil {
			return err
		}
		filename := opts.Filename
		if filename == "" {
			ext := mediaTypeToExtension(opts.MediaType)
			filename = "audio." + ext
		}
		part, err := w.CreateFormFile("file", filename)
		if err != nil {
			return err
		}
		_, err = part.Write(opts.Audio)
		return err
	}
	if err := build(writer); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	resp, err := m.provider.executeMultipartStream(ctx, endpointAudioTranscriptions, opts.Headers, body.Bytes(), writer.FormDataContentType())
	if err != nil {
		return nil, err
	}
	parts := make(chan StreamPart)
	sresp := &StreamResponse{Headers: resp.Headers.Clone()}
	result := &StreamResult{Stream: parts, Parts: parts, Request: RequestMetadata{}, Response: sresp}
	go m.runTranscriptionStream(ctx, resp, parts, sresp, warnings)
	return result, nil
}

// transcriptionStreamEvent is a single SSE event from the streaming
// transcription API (transcript.text.delta / transcript.text.done).
type transcriptionStreamEvent struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
	Text  string `json:"text"`
}

// runTranscriptionStream multiplexes partial transcript deltas into stream
// parts. It runs on its own goroutine; close and error terminate the parts
// channel with end / error parts.
func (m *openaiTranscriptionModel) runTranscriptionStream(ctx context.Context, resp *httpStreamResponse, parts chan<- StreamPart, sresp *StreamResponse, startWarnings []Warning) {
	defer close(parts)
	defer resp.Body.Close()
	parts <- StreamStart{Warnings: startWarnings}
	headers := resp.Headers.Clone()
	sresp.Headers = headers
	const partID = "transcript-0"
	started := false
	failed := false
	var text strings.Builder
	processSSEStream(resp.Body, func(raw []byte) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}
		var evt transcriptionStreamEvent
		if err := json.Unmarshal(raw, &evt); err != nil {
			parts <- StreamError{Err: InvalidResponseDataError{Message: "failed to parse: " + err.Error()}}
			failed = true
			return false
		}
		switch evt.Type {
		case "transcript.text.delta":
			if !started {
				parts <- StreamTextStart{ID: partID}
				started = true
			}
			parts <- StreamTextDelta{ID: partID, Text: evt.Delta}
			text.WriteString(evt.Delta)
		case "transcript.text.done":
			if evt.Text != "" && text.Len() == 0 {
				text.WriteString(evt.Text)
			}
		}
		return true
	})
	// An error part is terminal: no end or finish parts follow it.
	if failed {
		return
	}
	if started {
		parts <- StreamTextEnd{ID: partID}
	}
	parts <- StreamFinish{FinishReason: FinishReason{Unified: "stop", Raw: "stop"}}
}

// executeMultipartStream performs a multipart POST and returns the response
// as a stream. Unlike executeMultipart it does not buffer the body.
func (p *openaiProvider) executeMultipartStream(ctx context.Context, path string, perCall http.Header, body []byte, contentType string) (*httpStreamResponse, error) {
	if p.err != nil {
		return nil, p.err
	}
	requestURL, err := p.requestURL(path)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	for k, values := range p.headersForCall(perCall) {
		for _, v := range values {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Content-Type", contentType)
	if p.generateID != nil {
		req.Header.Set("x-request-id", p.generateID())
	}
	resp, err := p.fetch.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, p.readAPIError(resp)
	}
	return &httpStreamResponse{Body: resp.Body, Headers: resp.Header.Clone()}, nil
}
