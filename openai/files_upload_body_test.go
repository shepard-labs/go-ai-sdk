package openai

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

// multipartCapturingFetcher records the multipart body of the first
// request and returns a JSON body.
type multipartCapturingFetcher struct {
	captured string
	counter  int
}

func (m *multipartCapturingFetcher) Do(req *http.Request) (*http.Response, error) {
	if m.counter == 0 {
		body, _ := io.ReadAll(req.Body)
		m.captured = string(body)
	}
	m.counter++
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(`{"id":"file-1","object":"file","bytes":42,"created_at":1700000000,"filename":"foo.txt","purpose":"assistants","status":"processed"}`)),
	}, nil
}

func TestFilesUploadSendsExpiresAfterFields(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data:      []byte("hello"),
		Filename:  "foo.txt",
		MediaType: "text/plain",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{
				"purpose":      "vision",
				"expiresAfter": 3600,
			},
		},
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	if !strings.Contains(f.captured, "name=\"purpose\"") {
		t.Errorf("missing purpose field: %q", f.captured)
	}
	if !strings.Contains(f.captured, "vision") {
		t.Errorf("purpose value not found: %q", f.captured)
	}
	if !strings.Contains(f.captured, "expires_after[anchor]") {
		t.Errorf("missing expires_after[anchor]: %q", f.captured)
	}
	if !strings.Contains(f.captured, "expires_after[seconds]") {
		t.Errorf("missing expires_after[seconds]: %q", f.captured)
	}
	if !strings.Contains(f.captured, "3600") {
		t.Errorf("missing 3600 value: %q", f.captured)
	}
	if !strings.Contains(f.captured, "name=\"file\"") {
		t.Errorf("missing file field: %q", f.captured)
	}
}

func TestFilesUploadOmitsExpiresAfterWhenNotSet(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data: []byte("x"), Filename: "f.txt", MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	if strings.Contains(f.captured, "expires_after") {
		t.Errorf("expires_after should not be in body: %q", f.captured)
	}
}
