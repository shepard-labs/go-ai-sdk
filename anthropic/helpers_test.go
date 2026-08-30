package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

type roundTripFetcher struct {
	fn func(*http.Request) (*http.Response, error)
}

func (f roundTripFetcher) Do(req *http.Request) (*http.Response, error) { return f.fn(req) }

func jsonResponse(status int, body string) *http.Response {
	resp := textResponse(status, body)
	resp.Header.Set("Content-Type", "application/json")
	return resp
}
func textResponse(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}
func event(name, data string) string { return "event: " + name + "\ndata: " + data + "\n\n" }
func parseStream(t *testing.T, sse string) []StreamPart {
	t.Helper()
	out := make(chan StreamPart, 32)
	parseSSEStream(context.Background(), strings.NewReader(sse), out, nil)
	close(out)
	var parts []StreamPart
	for part := range out {
		parts = append(parts, part)
	}
	return parts
}
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

type testReadCloser struct {
	data []byte
	err  error
	done bool
}

func (r *testReadCloser) Read(p []byte) (int, error) {
	if r.done {
		return 0, r.err
	}
	r.done = true
	return copy(p, r.data), r.err
}
func (*testReadCloser) Close() error { return nil }

type captureLogger struct{ calls []string }

func (l *captureLogger) Debug(msg string, _ ...any) { l.calls = append(l.calls, "debug:"+msg) }
func (l *captureLogger) Info(msg string, _ ...any)  { l.calls = append(l.calls, "info:"+msg) }
func (l *captureLogger) Warn(msg string, _ ...any)  { l.calls = append(l.calls, "warn:"+msg) }
func (l *captureLogger) Error(msg string, _ ...any) { l.calls = append(l.calls, "error:"+msg) }

type captureFetcher struct{ request *http.Request }

func (f *captureFetcher) Do(req *http.Request) (*http.Response, error) {
	f.request = req
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"id":"id","content":[],"usage":{}}`))}, nil
}
func intPtr(value int) *int { return &value }
