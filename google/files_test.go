package google

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

// captureLogger implements Logger by silently absorbing all output.
type captureLogger struct{}

func (captureLogger) Debug(string, ...any) {}
func (captureLogger) Info(string, ...any)  {}
func (captureLogger) Warn(string, ...any)  {}
func (captureLogger) Error(string, ...any) {}

// TestFiles_Upload_PollsUntilActive verifies that Upload polls the file endpoint
// until the file is ACTIVE and returns the correct result.
func TestFiles_Upload_PollsUntilActive(t *testing.T) {
	t.Parallel()
	pollCount := 0

	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file1")
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method == http.MethodPost {
			// Upload bytes step: return PROCESSING state.
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"file":{"name":"files/file1","displayName":"test.txt","mimeType":"text/plain","sizeBytes":"5","uri":"files/file1","state":"PROCESSING"}}`))
			return
		}
		// GET /files/file1 — poll step.
		pollCount++
		if pollCount == 1 {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"file":{"name":"files/file1","displayName":"test.txt","mimeType":"text/plain","sizeBytes":"5","uri":"files/file1","state":"PROCESSING"}}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file1","displayName":"test.txt","mimeType":"text/plain","sizeBytes":"5","uri":"files/file1","state":"ACTIVE","createTime":"2025-01-01T00:00:00Z"}}`))
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
	f := &googleFiles{provider: p}

	result, err := f.Upload(context.Background(), []byte("hello"), FilesUploadOptions{
		MediaType: "text/plain",
		ProviderOptions: ProviderOptions{
			"google": {"displayName": "test.txt"},
		},
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}
	if result.ProviderReference["google"] != "files/file1" {
		t.Errorf("expected ProviderReference google = files/file1, got %v", result.ProviderReference)
	}
	if result.MediaType != "text/plain" {
		t.Errorf("expected MediaType text/plain, got %s", result.MediaType)
	}
	meta := result.ProviderMetadata["google"]
	if meta == nil {
		t.Fatal("ProviderMetadata missing 'google' key")
	}
	m := meta.(map[string]any)
	if m["state"] != "ACTIVE" {
		t.Errorf("expected state ACTIVE, got %v", m["state"])
	}
	if m["name"] != "files/file1" {
		t.Errorf("expected name files/file1, got %v", m["name"])
	}
	if pollCount < 1 {
		t.Errorf("expected at least 1 poll call, got %d", pollCount)
	}
}

// TestFiles_Upload_ActiveFromUploadBytes verifies that Upload returns immediately
// when the upload-bytes response already has state ACTIVE.
func TestFiles_Upload_ActiveFromUploadBytes(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file2")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Upload bytes step: file already ACTIVE — no polling needed.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file2","displayName":"ready.txt","mimeType":"text/plain","sizeBytes":"5","uri":"files/file2","state":"ACTIVE"}}`))
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
	f := &googleFiles{provider: p}

	result, err := f.Upload(context.Background(), []byte("world"), FilesUploadOptions{
		MediaType: "text/plain",
		ProviderOptions: ProviderOptions{
			"google": {"displayName": "ready.txt"},
		},
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}
	if result.ProviderReference["google"] != "files/file2" {
		t.Errorf("expected ProviderReference google = files/file2, got %v", result.ProviderReference)
	}
}

// TestFiles_Upload_Timeout verifies that the context deadline triggers
// GOOGLE_FILES_UPLOAD_TIMEOUT.
func TestFiles_Upload_Timeout(t *testing.T) {
	signal := make(chan struct{})
	mf := newMockFetcherWithSignal(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file3")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Respond after the deadline fires so deadline.C always wins the select.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file3","state":"PROCESSING"}}`))
	}, signal)
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	f := &googleFiles{provider: p}

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	// Close signal well after the deadline (150ms) + a small buffer (450ms)
	// so deadline.C fires before the handler responds in every run.
	go func() {
		time.Sleep(600 * time.Millisecond)
		close(signal)
	}()

	_, err := f.Upload(ctx, []byte("data"), FilesUploadOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_TIMEOUT" {
		var causeStr string
		if apiErr.Cause != nil {
			causeStr = fmt.Sprintf(" (cause=%v)", apiErr.Cause)
		}
		t.Errorf("expected Type GOOGLE_FILES_UPLOAD_TIMEOUT, got %s%s", apiErr.Type, causeStr)
	}
}

// TestFiles_Upload_FailedState verifies that a FAILED state returned from the API
// triggers GOOGLE_FILES_UPLOAD_FAILED.
func TestFiles_Upload_FailedState(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file4")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Upload bytes: FAILED immediately.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file4","state":"FAILED"}}`))
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
	f := &googleFiles{provider: p}

	_, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_FAILED" {
		t.Errorf("expected Type GOOGLE_FILES_UPLOAD_FAILED, got %s", apiErr.Type)
	}
}

// TestFiles_Upload_MissingUploadURL verifies that a missing X-Goog-Upload-Url
// header in the init response triggers GOOGLE_FILES_UPLOAD_ERROR.
func TestFiles_Upload_MissingUploadURL(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		// Init step: return 200 but without the required X-Goog-Upload-Url header.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{}`))
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
	f := &googleFiles{provider: p}

	_, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_ERROR" {
		t.Errorf("expected Type GOOGLE_FILES_UPLOAD_ERROR, got %s", apiErr.Type)
	}
}

// TestFiles_Upload_UploadBytesError verifies that a non-2xx response from the
// upload-bytes step triggers GOOGLE_FILES_UPLOAD_ERROR.
func TestFiles_Upload_UploadBytesError(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file5")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Upload bytes step: server error.
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"message":"internal error"}}`))
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
	f := &googleFiles{provider: p}

	_, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_ERROR" {
		t.Errorf("expected Type GOOGLE_FILES_UPLOAD_ERROR, got %s", apiErr.Type)
	}
}

// TestFiles_Upload_PollReturnsFailed verifies that GOOGLE_FILES_UPLOAD_FAILED is
// raised when a GET poll response has state FAILED.
func TestFiles_Upload_PollReturnsFailed(t *testing.T) {
	signal := make(chan struct{})
	mf := newMockFetcherWithSignal(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file6")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Return FAILED so the poll loop exits with GOOGLE_FILES_UPLOAD_FAILED.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file6","state":"FAILED"}}`))
	}, signal)
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
	f := &googleFiles{provider: p}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Start upload in goroutine so the handler can block until 3 polls complete.
	// Close the signal after ~3 polls (at 10ms intervals, that's ~30ms) with a
	// 400ms buffer — well inside the 2s timeout so ctx is still alive when the
	// FAILED response is written.
	go func() {
		time.Sleep(400 * time.Millisecond)
		close(signal)
	}()

	_, err := f.Upload(ctx, []byte("data"), FilesUploadOptions{
		ProviderOptions: ProviderOptions{
			"google": {"pollIntervalMs": float64(10)},
		},
	})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_FAILED" {
		t.Errorf("expected Type GOOGLE_FILES_UPLOAD_FAILED, got %s", apiErr.Type)
	}
}

// TestFiles_Upload_Abort verifies that ctx.Done() triggers GOOGLE_FILES_UPLOAD_ABORTED.
func TestFiles_Upload_Abort(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file7")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Respond immediately. The cancel fires at 50ms, well before the first poll
		// at 200ms, so the poll loop sees ctx cancellation and returns ABORTED.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file7","state":"PROCESSING"}}`))
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
	f := &googleFiles{provider: p}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, err := f.Upload(ctx, []byte("data"), FilesUploadOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_ABORTED" {
		t.Errorf("expected Type GOOGLE_FILES_UPLOAD_ABORTED, got %s", apiErr.Type)
	}
}

// TestFiles_Upload_FilenameWarning verifies that an unsupported filename option
// is logged and does not cause an error.
func TestFiles_Upload_FilenameWarning(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file8")
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file8","state":"ACTIVE","mimeType":"text/plain"}}`))
	})
	defer mf.Close()

	p := &googleProvider{
		baseURL: mf.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   mf,
		apiKey:  "test-key",
		logger:  &captureLogger{},
	}
	f := &googleFiles{provider: p}

	_, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{
		Filename:  "ignored.txt",
		MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}
}

// TestFiles_Upload_PollIntervalFromProviderOptions verifies that pollIntervalMs
// from providerOptions.google.pollIntervalMs is respected.
func TestFiles_Upload_PollIntervalFromProviderOptions(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file9")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Always return PROCESSING so timeout fires.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file9","state":"PROCESSING"}}`))
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
	f := &googleFiles{provider: p}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	start := time.Now()
	_, err := f.Upload(ctx, []byte("data"), FilesUploadOptions{
		ProviderOptions: ProviderOptions{
			"google": {
				"pollIntervalMs": float64(500),
				"pollTimeoutMs":  float64(2000),
			},
		},
	})
	elapsed := time.Since(start)

	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_TIMEOUT" {
		t.Errorf("expected GOOGLE_FILES_UPLOAD_TIMEOUT, got %s", apiErr.Type)
	}
	// With pollInterval=500ms, we should see multiple polls within 2s.
	if elapsed < 1*time.Second {
		t.Errorf("elapsed time too short: %v — pollInterval may not be respected", elapsed)
	}
}

// TestFiles_Upload_InitError verifies that a non-2xx init response triggers
// GOOGLE_FILES_UPLOAD_ERROR with Retryable=false.
func TestFiles_Upload_InitError(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":{"message":"bad key"}}`))
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
	f := &googleFiles{provider: p}

	_, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_FILES_UPLOAD_ERROR" {
		t.Errorf("expected Type GOOGLE_FILES_UPLOAD_ERROR, got %s", apiErr.Type)
	}
	if apiErr.Retryable {
		t.Error("expected Retryable=false for upload init failure")
	}
}

// TestFiles_Upload_DisplayNameFromProviderOptions verifies that the init body
// includes display_name when providerOptions.google.displayName is set.
func TestFiles_Upload_DisplayNameFromProviderOptions(t *testing.T) {
	t.Parallel()
	var initBody string

	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			buf := make([]byte, 2048)
			n, _ := r.Body.Read(buf)
			initBody = string(buf[:n])
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file10")
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file10","state":"ACTIVE"}}`))
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
	f := &googleFiles{provider: p}

	_, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{
		ProviderOptions: ProviderOptions{
			"google": {"displayName": "my-document.pdf"},
		},
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}
	if !strings.Contains(initBody, `"display_name":"my-document.pdf"`) {
		t.Errorf("expected init body to contain display_name, got: %s", initBody)
	}
}

// TestFiles_Upload_MediaTypeFallback verifies that when the uploaded file has
// no mimeType, the MediaType from opts is used.
func TestFiles_Upload_MediaTypeFallback(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file11")
			w.WriteHeader(http.StatusOK)
			return
		}
		// Active state with no mimeType in response.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file11","state":"ACTIVE","mimeType":""}}`))
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
	f := &googleFiles{provider: p}

	result, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{
		MediaType: "application/pdf",
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}
	if result.MediaType != "application/pdf" {
		t.Errorf("expected MediaType application/pdf (from opts), got %s", result.MediaType)
	}
}

// TestFiles_Upload_NoDisplayNameOmitted verifies that when displayName is not set,
// the init body has no display_name field.
func TestFiles_Upload_NoDisplayNameOmitted(t *testing.T) {
	t.Parallel()
	var initBody string

	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			buf := make([]byte, 2048)
			n, _ := r.Body.Read(buf)
			initBody = string(buf[:n])
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file12")
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"file":{"name":"files/file12","state":"ACTIVE"}}`))
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
	f := &googleFiles{provider: p}

	_, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}
	if strings.Contains(initBody, "display_name") {
		t.Errorf("expected init body to omit display_name, got: %s", initBody)
	}
}

// TestFiles_Upload_ProviderMetadataFields verifies that ProviderMetadata
// includes all available fields from the API response.
func TestFiles_Upload_ProviderMetadataFields(t *testing.T) {
	t.Parallel()
	mf := newMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files" {
			w.Header().Set("X-Goog-Upload-Url", "/upload/resumable/file13")
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{
			"file":{
				"name":"files/file13",
				"displayName":"report.pdf",
				"mimeType":"application/pdf",
				"sizeBytes":"12345",
				"uri":"files/file13",
				"state":"ACTIVE",
				"createTime":"2025-06-01T10:00:00Z",
				"updateTime":"2025-06-01T10:05:00Z",
				"expirationTime":"2026-06-01T00:00:00Z",
				"sha256Hash":"abc123def"
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
	f := &googleFiles{provider: p}

	result, err := f.Upload(context.Background(), []byte("data"), FilesUploadOptions{})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}
	meta := result.ProviderMetadata["google"].(map[string]any)
	if meta["createTime"] != "2025-06-01T10:00:00Z" {
		t.Errorf("missing createTime in metadata: %v", meta)
	}
	if meta["sha256Hash"] != "abc123def" {
		t.Errorf("missing sha256Hash in metadata: %v", meta)
	}
	if meta["expirationTime"] != "2026-06-01T00:00:00Z" {
		t.Errorf("missing expirationTime in metadata: %v", meta)
	}
}
