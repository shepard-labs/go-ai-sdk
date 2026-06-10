package google

import (
	"context"
	"io"
	"net/http"
	"testing"
	"time"
)

func newSSEHandler(t *testing.T, chunks []string) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	})
}

func collectStream(t *testing.T, parts <-chan StreamPart, timeout time.Duration) []StreamPart {
	t.Helper()
	out := []StreamPart{}
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	for {
		select {
		case p, ok := <-parts:
			if !ok {
				return out
			}
			out = append(out, p)
		case <-deadline.C:
			t.Fatalf("stream channel did not close within %v (collected %d parts)", timeout, len(out))
		}
	}
}

func TestDoStream_TextOnly(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"Hello"}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, err := lm.DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	parts := collectStream(t, res.Stream, 2*time.Second)
	if len(parts) < 4 {
		t.Fatalf("parts = %d, want >=4: %+v", len(parts), parts)
	}
	if _, ok := parts[0].(StreamStart); !ok {
		t.Errorf("parts[0] = %T, want StreamStart", parts[0])
	}
	if _, ok := parts[len(parts)-1].(StreamFinish); !ok {
		t.Errorf("last = %T, want StreamFinish", parts[len(parts)-1])
	}
	sawText := false
	for _, p := range parts {
		if _, ok := p.(StreamTextDelta); ok {
			sawText = true
		}
	}
	if !sawText {
		t.Errorf("no StreamTextDelta in %+v", parts)
	}
}
