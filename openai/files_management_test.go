package openai

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// filesTestServer spins up an httptest server that records the last request
// and serves canned bodies per path.
type filesTestServer struct {
	server *httptest.Server
	last   *http.Request
	body   []byte
	// Parsed multipart upload fields, captured synchronously inside the
	// handler so tests never touch the server-side request body afterwards.
	uploadPurpose   string
	uploadFileBytes []byte
	uploadFilename  string
}

func newFilesTestServer(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) *filesTestServer {
	t.Helper()
	fts := &filesTestServer{}
	fts.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fts.last = r
		body, _ := io.ReadAll(r.Body)
		fts.body = body
		// Replay the body so handlers can still parse it (e.g. multipart).
		r.Body = io.NopCloser(bytes.NewReader(body))
		// Parse multipart uploads here, synchronously before the response,
		// so tests can assert on the stashed values instead of reading
		// fts.last's body after the handler returned (a data race with
		// net/http's finishRequest under -race).
		if mr, err := r.MultipartReader(); err == nil {
			for {
				part, err := mr.NextPart()
				if err == io.EOF {
					break
				}
				if err != nil {
					break
				}
				content, _ := io.ReadAll(part)
				if part.FileName() != "" {
					fts.uploadFileBytes = content
					fts.uploadFilename = part.FileName()
				} else if part.FormName() == "purpose" {
					fts.uploadPurpose = string(content)
				}
			}
			// Replay again so the handler still sees a fresh body.
			r.Body = io.NopCloser(bytes.NewReader(body))
		}
		handler(w, r)
	}))
	t.Cleanup(fts.server.Close)
	return fts
}

func filesTestProvider(fts *filesTestServer) Provider {
	return CreateOpenAI(ProviderSettings{APIKey: "test-key", BaseURL: fts.server.URL})
}

// TestFilesGetMetadataRoundTrip verifies GET /files/{id} maps the file
// object into the result.
func TestFilesGetMetadataRoundTrip(t *testing.T) {
	fts := newFilesTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"file-abc","object":"file","bytes":11,"created_at":1700000000,"filename":"hello.txt","purpose":"assistants","status":"processed"}`))
	})
	res, err := filesTestProvider(fts).Files().GetFileMetadata(context.Background(), "file-abc")
	if err != nil {
		t.Fatalf("GetFileMetadata: %v", err)
	}
	if fts.last.Method != http.MethodGet {
		t.Errorf("method = %s, want GET", fts.last.Method)
	}
	if fts.last.URL.Path != "/files/file-abc" {
		t.Errorf("path = %s", fts.last.URL.Path)
	}
	if res.ProviderReference["openai"] != "file-abc" {
		t.Errorf("ProviderReference: %v", res.ProviderReference)
	}
	if res.Filename != "hello.txt" {
		t.Errorf("Filename = %q", res.Filename)
	}
}

// TestFilesDownloadContentBytes verifies GET /files/{id}/content returns
// the raw bytes.
func TestFilesDownloadContentBytes(t *testing.T) {
	fts := newFilesTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("raw-bytes-123"))
	})
	data, err := filesTestProvider(fts).Files().DownloadFile(context.Background(), "file-abc")
	if err != nil {
		t.Fatalf("DownloadFile: %v", err)
	}
	if fts.last.Method != http.MethodGet {
		t.Errorf("method = %s, want GET", fts.last.Method)
	}
	if fts.last.URL.Path != "/files/file-abc/content" {
		t.Errorf("path = %s", fts.last.URL.Path)
	}
	if string(data) != "raw-bytes-123" {
		t.Errorf("data = %q", data)
	}
}

// TestFilesDeleteReportsFlag verifies DELETE /files/{id} surfaces the
// server's deleted flag.
func TestFilesDeleteReportsFlag(t *testing.T) {
	fts := newFilesTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"file-abc","object":"file","deleted":true}`))
	})
	deleted, err := filesTestProvider(fts).Files().DeleteFile(context.Background(), "file-abc")
	if err != nil {
		t.Fatalf("DeleteFile: %v", err)
	}
	if fts.last.Method != http.MethodDelete {
		t.Errorf("method = %s, want DELETE", fts.last.Method)
	}
	if fts.last.URL.Path != "/files/file-abc" {
		t.Errorf("path = %s", fts.last.URL.Path)
	}
	if !deleted {
		t.Error("deleted = false, want true")
	}
}

// TestFilesUploadFromReader verifies the reader overload sends the default
// purpose and the streamed bytes through the shared multipart path.
func TestFilesUploadFromReader(t *testing.T) {
	fts := newFilesTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"file-xyz","object":"file","filename":"note.txt","purpose":"assistants"}`))
	})
	res, err := filesTestProvider(fts).Files().Upload(
		context.Background(), strings.NewReader("reader-payload"), "note.txt", "text/plain", nil)
	if err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if fts.last.Method != http.MethodPost {
		t.Errorf("method = %s, want POST", fts.last.Method)
	}
	if fts.last.URL.Path != "/files" {
		t.Errorf("path = %s", fts.last.URL.Path)
	}
	if fts.uploadPurpose != "assistants" {
		t.Errorf("purpose = %q, want assistants", fts.uploadPurpose)
	}
	if string(fts.uploadFileBytes) != "reader-payload" {
		t.Errorf("file bytes = %q", fts.uploadFileBytes)
	}
	if fts.uploadFilename != "note.txt" {
		t.Errorf("filename = %q", fts.uploadFilename)
	}
	if res.ProviderReference["openai"] != "file-xyz" {
		t.Errorf("ProviderReference: %v", res.ProviderReference)
	}
	if res.Filename != "note.txt" {
		t.Errorf("Filename = %q", res.Filename)
	}
}

// TestFilesUploadFromReaderSendsExpiresAfter verifies provider options flow
// through the reader overload into the multipart brackets.
func TestFilesUploadFromReaderSendsExpiresAfter(t *testing.T) {
	fts := newFilesTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"file-1","object":"file"}`))
	})
	_, err := filesTestProvider(fts).Files().Upload(
		context.Background(),
		bytes.NewReader([]byte("x")),
		"a.bin",
		"application/octet-stream",
		ProviderOptions{"openai": map[string]any{"purpose": "vision", "expiresAfter": 60}},
	)
	if err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if !strings.Contains(string(fts.body), "expires_after[seconds]") {
		t.Error("multipart body missing expires_after[seconds]")
	}
	if !strings.Contains(string(fts.body), "vision") {
		t.Error("multipart body missing purpose vision")
	}
}

// TestFilesEmptyIDRejected verifies empty file IDs fail before any request.
func TestFilesEmptyIDRejected(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	if _, err := p.Files().GetFileMetadata(context.Background(), ""); err == nil {
		t.Error("GetFileMetadata with empty ID should error")
	}
	if _, err := p.Files().DownloadFile(context.Background(), ""); err == nil {
		t.Error("DownloadFile with empty ID should error")
	}
	if _, err := p.Files().DeleteFile(context.Background(), ""); err == nil {
		t.Error("DeleteFile with empty ID should error")
	}
	if f.calls != 0 {
		t.Errorf("calls = %d, want 0", f.calls)
	}
}

// TestFilesUploadCancelledContextReturnsPromptly verifies the streaming
// Upload path does not hang when the context is cancelled: the pipe writer
// observes ctx.Done and the reader side is released on early fetch failure.
func TestFilesUploadCancelledContextReturnsPromptly(t *testing.T) {
	fts := newFilesTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"file-cancelled","object":"file"}`))
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan error, 1)
	go func() {
		_, err := filesTestProvider(fts).Files().Upload(
			ctx, strings.NewReader("payload"), "note.txt", "text/plain", nil)
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Error("Upload with cancelled context should error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Upload with cancelled context hung")
	}
}
