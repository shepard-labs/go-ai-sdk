package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

// transcriptionHeaderServer records the last request headers and serves a
// done-only SSE stream.
type transcriptionHeaderServer struct {
	server      *httptest.Server
	lastHeaders http.Header
	hits        int
}

func newTranscriptionHeaderServer(t *testing.T) *transcriptionHeaderServer {
	t.Helper()
	ths := &transcriptionHeaderServer{}
	ths.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ths.lastHeaders = r.Header.Clone()
		ths.hits++
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"transcript.text.done\",\"text\":\"hi\"}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(ths.server.Close)
	return ths
}

func drainStreamParts(res *StreamResult) []StreamPart {
	var out []StreamPart
	for p := range res.Parts {
		out = append(out, p)
	}
	return out
}

func streamerForServer(t *testing.T, ths *transcriptionHeaderServer) *openaiTranscriptionModel {
	t.Helper()
	prov, ok := CreateOpenAI(ProviderSettings{APIKey: "provider-key", BaseURL: ths.server.URL}).(*openaiProvider)
	if !ok {
		t.Fatal("CreateOpenAI did not return *openaiProvider")
	}
	return newTranscriptionStreamer(prov, "gpt-4o-transcribe")
}

// TestTranscriptionStreamSendsDefaultAcceptHeader verifies the SSE
// subprotocol default Accept header is sent.
func TestTranscriptionStreamSendsDefaultAcceptHeader(t *testing.T) {
	ths := newTranscriptionHeaderServer(t)
	m := streamerForServer(t, ths)
	res, err := m.DoStream(context.Background(), TranscriptionOptions{
		Audio: []byte{0x00}, MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	drainStreamParts(res)
	if got := ths.lastHeaders.Get("Accept"); got != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", got)
	}
	if got := ths.lastHeaders.Get("Authorization"); got != "Bearer provider-key" {
		t.Errorf("Authorization = %q, want provider default", got)
	}
}

// TestTranscriptionStreamPerCallAuthOverridesDefault verifies dual-auth
// handling: a per-call Authorization header wins over the provider default.
func TestTranscriptionStreamPerCallAuthOverridesDefault(t *testing.T) {
	ths := newTranscriptionHeaderServer(t)
	m := streamerForServer(t, ths)
	res, err := m.DoStream(context.Background(), TranscriptionOptions{
		Audio: []byte{0x00}, MediaType: "audio/wav",
		Headers: http.Header{"Authorization": []string{"Bearer caller-key"}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	drainStreamParts(res)
	if got := ths.lastHeaders.Get("Authorization"); got != "Bearer caller-key" {
		t.Errorf("Authorization = %q, want per-call override", got)
	}
}

// TestTranscriptionStreamCustomAcceptPreserved verifies a per-call Accept
// header is not clobbered by the SSE default.
func TestTranscriptionStreamCustomAcceptPreserved(t *testing.T) {
	ths := newTranscriptionHeaderServer(t)
	m := streamerForServer(t, ths)
	res, err := m.DoStream(context.Background(), TranscriptionOptions{
		Audio: []byte{0x00}, MediaType: "audio/wav",
		Headers: http.Header{"Accept": []string{"custom"}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	drainStreamParts(res)
	if got := ths.lastHeaders.Get("Accept"); got != "custom" {
		t.Errorf("Accept = %q, want custom", got)
	}
}

// TestTranscriptionStreamCancelledContextFailsPreOpen verifies a cancelled
// context fails before any stream is opened.
func TestTranscriptionStreamCancelledContextFailsPreOpen(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := newTranscriptionStreamer(p, "gpt-4o-transcribe").DoStream(ctx, TranscriptionOptions{
		Audio: []byte{0x00}, MediaType: "audio/wav",
	})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	if f.calls != 0 {
		t.Errorf("calls = %d, want 0 (no stream opened)", f.calls)
	}
}
