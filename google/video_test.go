package google

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// mockFetcher is a Fetcher that routes requests to a test server handler.
type mockFetcher struct {
	server *httptest.Server
}

func newMockFetcher(handler func(http.ResponseWriter, *http.Request)) *mockFetcher {
	m := &mockFetcher{}
	m.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler(w, r)
	}))
	return m
}

func (f *mockFetcher) Do(req *http.Request) (*http.Response, error) {
	return http.DefaultClient.Do(req)
}

func (f *mockFetcher) URL() string { return f.server.URL }

func (f *mockFetcher) Close() { f.server.Close() }

// ---- Test cases ----

// TestVideo_DoGenerate_PollsUntilDone verifies that DoGenerate polls the
// operation endpoint until done=true and returns the video URIs.
func TestVideo_DoGenerate_PollsUntilDone(t *testing.T) {
	t.Parallel()
	pollCount := 0

	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Write([]byte(`{"done":false,"name":"operations/12345"}`))
			return
		}
		// GET /v1/operations/12345
		pollCount++
		if pollCount == 1 {
			w.Write([]byte(`{"done":false,"name":"operations/12345"}`))
			return
		}
		w.Write([]byte(`{
			"done": true,
			"name": "operations/12345",
			"response": {
				"predictions": [
					{"video": {"uri": "[image]"}},
					{"video": {"uri": "[image]"}}
				]
			}
		}`))
	})
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	m := &googleVideoModel{provider: p, modelID: "veo-3.0-generate-001"}

	result, err := m.DoGenerate(context.Background(), VideoGenerateOptions{Prompt: "a cat"})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if len(result.Videos) != 2 {
		t.Fatalf("expected 2 videos, got %d", len(result.Videos))
	}
	// ProviderMetadata captures the operation name.
	pm := result.ProviderMetadata["google"]
	if pm == nil {
		t.Fatal("ProviderMetadata missing 'google' key")
	}
	if pm.(map[string]any)["operationName"] != "operations/12345" {
		t.Errorf("unexpected operationName: %v", pm)
	}
	if pollCount < 1 {
		t.Errorf("expected at least 1 poll call, got %d", pollCount)
	}
}

// TestVideo_DoGenerate_Timeout verifies that the context deadline triggers
// GOOGLE_VIDEO_GENERATION_TIMEOUT.
func TestVideo_DoGenerate_Timeout(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		// Always return not-done so the poll loop runs until context deadline.
		w.Write([]byte(`{"done":false,"name":"op"}`))
	})
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	m := &googleVideoModel{provider: p, modelID: "veo-3.0-generate-001"}

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	_, err := m.DoGenerate(ctx, VideoGenerateOptions{Prompt: "a dog"})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_VIDEO_GENERATION_ABORTED" {
		t.Errorf("Type = %q, want GOOGLE_VIDEO_GENERATION_ABORTED", apiErr.Type)
	}
}

// TestVideo_DoGenerate_Abort verifies that ctx cancellation returns
// GOOGLE_VIDEO_GENERATION_ABORTED.
func TestVideo_DoGenerate_Abort(t *testing.T) {
	t.Parallel()
	calls := 0
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Method == http.MethodPost {
			w.Write([]byte(`{"done":false,"name":"op"}`))
			return
		}
		// Slow poll response.
		time.Sleep(300 * time.Millisecond)
		w.Write([]byte(`{"done":false,"name":"op"}`))
	})
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	m := &googleVideoModel{provider: p, modelID: "veo-3.0-generate-001"}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	_, err := m.DoGenerate(ctx, VideoGenerateOptions{
		Prompt:          "a bird",
		ProviderOptions: ProviderOptions{"google": map[string]any{"pollIntervalMs": 50}},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %T: %v", err, err)
	}
}

// TestVideo_DoGenerate_AppendsKeyToURIs verifies that ?key=<apiKey> is appended
// to video URIs, and that &key= is used when the URI already has a query string.
func TestVideo_DoGenerate_AppendsKeyToURIs(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Write([]byte(`{"done":false,"name":"op"}`))
			return
		}
		w.Write([]byte(`{
			"done": true,
			"name": "op",
			"response": {
				"predictions": [
					{"video": {"uri": "[image]?alt=media"}},
					{"video": {"uri": "[image]"}}
				]
			}
		}`))
	})
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	m := &googleVideoModel{provider: p, modelID: "veo-3.0-generate-001"}

	result, err := m.DoGenerate(context.Background(), VideoGenerateOptions{Prompt: "test"})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	// URI with existing ?alt=media should get &key= appended.
	if result.Videos[0] != "[image]?alt=media&key=test-key" {
		t.Errorf("URI[0] unexpected: %s", result.Videos[0])
	}
	// URI without existing query should get ?key= appended.
	if result.Videos[1] != "[image]?key=test-key" {
		t.Errorf("URI[1] unexpected: %s", result.Videos[1])
	}
}

// TestVideo_DoGenerate_MissingOperationName verifies that a POST response
// missing the operation name returns GOOGLE_VIDEO_GENERATION_ERROR.
func TestVideo_DoGenerate_MissingOperationName(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"done":false,"name":""}`))
	})
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	m := &googleVideoModel{provider: p, modelID: "veo-3.0-generate-001"}

	_, err := m.DoGenerate(context.Background(), VideoGenerateOptions{Prompt: "test"})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_VIDEO_GENERATION_ERROR" {
		t.Errorf("Type = %q, want GOOGLE_VIDEO_GENERATION_ERROR", apiErr.Type)
	}
	if apiErr.Retryable {
		t.Error("Retryable should be false")
	}
}

// TestVideo_DoGenerate_EmptySamples verifies that a completed LRO with no
// predictions returns GOOGLE_VIDEO_GENERATION_ERROR.
func TestVideo_DoGenerate_EmptySamples(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"done":true,"name":"op","response":{"predictions":[]}}`))
	})
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	m := &googleVideoModel{provider: p, modelID: "veo-3.0-generate-001"}

	_, err := m.DoGenerate(context.Background(), VideoGenerateOptions{Prompt: "test"})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_VIDEO_GENERATION_ERROR" {
		t.Errorf("Type = %q, want GOOGLE_VIDEO_GENERATION_ERROR", apiErr.Type)
	}
}

// TestVideo_MaxVideosPerCall verifies that MaxVideosPerCall returns 4.
func TestVideo_MaxVideosPerCall(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(http.ResponseWriter, *http.Request) {})
	defer mf.Close()
	p := &googleProvider{baseURL: mf.URL(), name: "google", headers: http.Header{}, fetch: mf, apiKey: "k", logger: noopLogger{}}
	m := &googleVideoModel{provider: p, modelID: "veo-3.0-generate-001"}
	if m.MaxVideosPerCall() != defaultMaxVideosPerCall {
		t.Errorf("MaxVideosPerCall() = %d, want %d", m.MaxVideosPerCall(), defaultMaxVideosPerCall)
	}
}

// TestVideo_MapResolution verifies the resolution label mapping table.
func TestVideo_MapResolution(t *testing.T) {
	t.Parallel()
	tests := []struct {
		input string
		want  string
	}{
		{"1280x720", "720p"},
		{"1920x1080", "1080p"},
		{"3840x2160", "4k"},
		{"2560x1440", "2560x1440"}, // passthrough
		{"1024x768", "1024x768"},
		{"", ""},
	}
	for _, tc := range tests {
		if got := mapResolution(tc.input); got != tc.want {
			t.Errorf("mapResolution(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

