package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func terminalAPIError(t *testing.T, reader io.Reader) (*APICallError, []StreamPart) {
	t.Helper()
	out := make(chan StreamPart, 32)
	parseSSEStreamWithLimit(context.Background(), reader, out, nil, 1024)
	close(out)
	var parts []StreamPart
	for part := range out {
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		t.Fatal("no stream parts")
	}
	streamErr, ok := parts[len(parts)-1].(StreamError)
	if !ok {
		t.Fatalf("terminal part = %#v, want StreamError", parts[len(parts)-1])
	}
	var apiErr *APICallError
	if !errors.As(streamErr.Err, &apiErr) {
		t.Fatalf("error = %T %v, want APICallError", streamErr.Err, streamErr.Err)
	}
	return apiErr, parts
}

func TestPhase4ProviderErrorMappingAndDiagnostics(t *testing.T) {
	cases := []struct {
		typ       string
		status    int
		retryable bool
	}{
		{"invalid_request_error", 400, false},
		{"authentication_error", 401, false},
		{"permission_error", 403, false},
		{"not_found_error", 404, false},
		{"request_too_large", 413, false},
		{"rate_limit_error", 429, true},
		{"api_error", 500, true},
		{"overloaded_error", 529, true},
	}
	for _, tc := range cases {
		t.Run(tc.typ, func(t *testing.T) {
			raw := event("error", `{"error":{"type":"`+tc.typ+`","message":"failed"}}`)
			err, parts := terminalAPIError(t, strings.NewReader(raw))
			if len(parts) != 1 || err.Type != tc.typ || err.Message != "failed" || err.Status != tc.status || !err.SyntheticStatus || err.Retryable != tc.retryable || string(err.Body) != raw {
				t.Fatalf("parts = %#v error = %#v", parts, err)
			}
		})
	}
}

func TestPhase4UnknownMalformedAndProtocolErrors(t *testing.T) {
	for _, raw := range []string{
		event("error", `{"error":{"type":"new_error","message":"failed"}}`),
		event("error", `{"error":{"type":"api_error"}}`),
		event("error", `not json`),
		event("content_block_start", `{"index":0}`),
	} {
		err, parts := terminalAPIError(t, strings.NewReader(raw+event("message_start", `{"message":{"id":"later","model":"later"}}`)))
		if len(parts) != 1 || err.Status != 0 || err.SyntheticStatus || err.Retryable || len(err.Body) == 0 {
			t.Fatalf("parts = %#v error = %#v", parts, err)
		}
	}
}

func TestPhase4MalformedErrorPreservesDecodeCauseAndRawRecord(t *testing.T) {
	raw := event("error", `{"error":`)
	err, parts := terminalAPIError(t, strings.NewReader(raw))
	var syntaxErr *json.SyntaxError
	if len(parts) != 1 || err.Message != "decode error event" || !errors.As(err, &syntaxErr) || string(err.Body) != raw || err.Truncated {
		t.Fatalf("parts = %#v error = %#v", parts, err)
	}
}

func TestPhase4IncompleteFinalRecordPreservesDecodeCause(t *testing.T) {
	partial := "event: message_start\ndata: {\"message\":{\"id\":\"m\"}"
	err, parts := terminalAPIError(t, strings.NewReader(partial))
	var syntaxErr *json.SyntaxError
	if len(parts) != 1 || err.Message != "decode message_start" || !errors.As(err, &syntaxErr) || string(err.Body) != partial || err.Truncated {
		t.Fatalf("parts = %#v error = %#v", parts, err)
	}
}

func TestPhase4ErrorDiagnosticTruncation(t *testing.T) {
	raw := event("error", `{"error":{"type":"api_error","message":"`+strings.Repeat("x", 128)+`"}}`)
	out := make(chan StreamPart, 1)
	parseSSEStreamWithLimit(context.Background(), strings.NewReader(raw), out, nil, 32)
	close(out)
	streamErr := (<-out).(StreamError)
	var err *APICallError
	if !errors.As(streamErr.Err, &err) || err.Type != "api_error" || err.Status != http.StatusInternalServerError || !err.SyntheticStatus || !err.Truncated || len(err.Body) != 32 || string(err.Body) != raw[:32] {
		t.Fatalf("error = %#v", err)
	}
}

type bytesThenErrorReader struct {
	data []byte
	err  error
	done bool
}

func (r *bytesThenErrorReader) Read(p []byte) (int, error) {
	if r.done {
		return 0, r.err
	}
	r.done = true
	return copy(p, r.data), r.err
}

func TestPhase4PartialReaderErrorIsTerminalAndPreserved(t *testing.T) {
	cause := errors.New("reader failed")
	partial := []byte("event: message_start\ndata: {\"message\":{\"id\":\"m\",\"model\":\"c\"}}")
	out := make(chan StreamPart, 8)
	parseSSEStreamWithLimit(context.Background(), &bytesThenErrorReader{data: partial, err: cause}, out, nil, 64)
	close(out)
	var parts []StreamPart
	for part := range out {
		parts = append(parts, part)
	}
	streamErr := parts[0].(StreamError)
	var err *APICallError
	if !errors.As(streamErr.Err, &err) {
		t.Fatalf("error = %v, want APICallError", streamErr.Err)
	}
	if len(parts) != 1 || !errors.Is(err, cause) || err.Status != 0 || err.Retryable || string(err.Body) != string(partial) || err.Truncated {
		t.Fatalf("parts = %#v error = %#v", parts, err)
	}
}

func TestPhase4CompleteEventsBeforeReaderError(t *testing.T) {
	cause := errors.New("reader failed")
	raw := event("message_start", `{"message":{"id":"m","model":"c"}}`) +
		event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) +
		event("content_block_delta", `{"index":0,"delta":{"type":"text_delta","text":"hello"}}`)
	out := make(chan StreamPart, 8)
	parseSSEStreamWithLimit(context.Background(), &bytesThenErrorReader{data: []byte(raw), err: cause}, out, nil, 1024)
	close(out)
	var parts []StreamPart
	for part := range out {
		parts = append(parts, part)
	}
	if len(parts) != 5 {
		t.Fatalf("parts = %#v", parts)
	}
	if _, ok := parts[0].(StreamStart); !ok {
		t.Fatalf("part 0 = %#v", parts[0])
	}
	if _, ok := parts[1].(StreamResponseMetadata); !ok {
		t.Fatalf("part 1 = %#v", parts[1])
	}
	if _, ok := parts[2].(StreamTextStart); !ok {
		t.Fatalf("part 2 = %#v", parts[2])
	}
	if delta, ok := parts[3].(StreamTextDelta); !ok || delta.Text != "hello" {
		t.Fatalf("part 3 = %#v", parts[3])
	}
	streamErr, ok := parts[4].(StreamError)
	if !ok || !errors.Is(streamErr.Err, cause) {
		t.Fatalf("terminal = %#v", parts[4])
	}
}

func TestPhase4StreamingErrorResponseMetadata(t *testing.T) {
	for _, headerName := range []string{"x-request-id", "request-id"} {
		t.Run(headerName, func(t *testing.T) {
			headers := make(http.Header)
			headers.Set(headerName, "req-stream")
			metadata := streamErrorMetadata{headers: cloneHeader(headers), requestID: responseRequestID(headers)}
			out := make(chan StreamPart, 1)
			parseSSEStreamWithMetadata(context.Background(), strings.NewReader(event("error", `{"error":{"type":"api_error","message":"failed"}}`)), out, nil, 1024, metadata)
			close(out)
			streamErr := (<-out).(StreamError)
			var apiErr *APICallError
			if !errors.As(streamErr.Err, &apiErr) || apiErr.RequestID != "req-stream" || apiErr.Headers.Get(headerName) != "req-stream" {
				t.Fatalf("error = %#v", apiErr)
			}
			headers.Set(headerName, "mutated")
			if apiErr.Headers.Get(headerName) != "req-stream" {
				t.Fatalf("headers were not cloned: %#v", apiErr.Headers)
			}
		})
	}
}

func TestPhase4DoStreamErrorPreservesSuccessfulResponseMetadata(t *testing.T) {
	headers := http.Header{"Request-Id": []string{"req-stream"}, "X-Trace": []string{"trace"}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: roundTripFetcher{fn: func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: headers, Body: io.NopCloser(strings.NewReader(event("error", `{"error":{"type":"api_error","message":"failed"}}`))), Request: req}, nil
	}}}).Model("claude-3-haiku-20240307")
	result, callErr := model.DoStream(context.Background(), StreamOptions{})
	if callErr != nil {
		t.Fatal(callErr)
	}
	streamErr, ok := (<-result.Stream).(StreamError)
	if !ok {
		t.Fatalf("part = %#v", streamErr)
	}
	var err *APICallError
	if !errors.As(streamErr.Err, &err) || result.Response.Headers.Get("X-Trace") != "trace" || err.RequestID != "req-stream" || err.Headers.Get("Request-Id") != "req-stream" || err.Headers.Get("X-Trace") != "trace" {
		t.Fatalf("response = %#v error = %#v", result.Response, err)
	}
	headers.Set("X-Trace", "mutated")
	if result.Response.Headers.Get("X-Trace") != "trace" || err.Headers.Get("X-Trace") != "trace" {
		t.Fatalf("response headers = %#v error headers = %#v", result.Response.Headers, err.Headers)
	}
	if _, open := <-result.Stream; open {
		t.Fatal("stream emitted output after terminal error")
	}
}

func TestPhase4CancelledTerminalSendDoesNotBlock(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan StreamPart)
	done := make(chan struct{})
	cancel()
	go func() {
		parseSSEStreamWithLimit(ctx, &bytesThenErrorReader{err: errors.New("reader failed")}, out, nil, 1024)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("terminal send blocked after cancellation")
	}
}

func TestPhase4HTTPSetupErrorPreservesResponseMetadataAndPartialBody(t *testing.T) {
	cause := errors.New("body failed")
	resp := &http.Response{StatusCode: 503, Header: http.Header{"X-Request-Id": []string{"req-1"}}, Body: io.NopCloser(&bytesThenErrorReader{data: []byte(`{"error":{"type":"api_error","message":"partial"}}`), err: cause})}
	body, truncated, readErr := readLimited(resp.Body, 32)
	var apiErr *APICallError
	if !errors.As(apiCallError(resp, body, truncated, readErr), &apiErr) {
		t.Fatal("want APICallError")
	}
	if apiErr.Status != 503 || apiErr.SyntheticStatus || apiErr.RequestID != "req-1" || apiErr.Headers.Get("X-Request-Id") != "req-1" || !errors.Is(apiErr, cause) || len(apiErr.Body) != 32 || !apiErr.Truncated {
		t.Fatalf("error = %#v", apiErr)
	}
}
