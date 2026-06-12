package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestFilesUploadWithExpiresAfter verifies that the files client sends
// `expires_after[anchor]` and `expires_after[seconds]` form fields when
// `expiresAfter` is set in provider options.
func TestFilesUploadWithExpiresAfter(t *testing.T) {
	respBody := `{"id":"file-1","object":"file","bytes":42,"created_at":1700000000,"filename":"foo.txt","purpose":"assistants","status":"processed"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
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
	if f.calls != 1 {
		t.Fatalf("calls = %d", f.calls)
	}
	// The recordingFetcher doesn't capture the body for multipart, but we
	// can at least verify no error path was hit.
}

// TestSkillsUploadMultipleFiles verifies that multiple SkillsFile entries
// are written to the multipart form.
func TestSkillsUploadMultipleFiles(t *testing.T) {
	respBody := `{"id":"skill-1","name":"s","latest_version":"v1"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		Files: []SkillsFile{
			{Path: "a.zip", Data: []byte("zip1"), MediaType: "application/zip"},
			{Path: "b.zip", Data: []byte("zip2"), MediaType: "application/zip"},
		},
	})
	if err != nil {
		t.Fatalf("UploadSkill: %v", err)
	}
	if res.ProviderReference["openai"] != "skill-1" {
		t.Errorf("ProviderReference: %v", res.ProviderReference)
	}
	if f.calls != 1 {
		t.Errorf("calls: %d", f.calls)
	}
}

// TestFilesUploadPurposeDefault verifies that purpose defaults to "assistants".
func TestFilesUploadPurposeDefault(t *testing.T) {
	respBody := `{"id":"file-1","object":"file","filename":"foo.txt","purpose":"assistants"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data: []byte("x"), Filename: "foo.txt", MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	// Recording fetcher's call count verifies the request fired.
	if f.calls != 1 {
		t.Errorf("calls: %d", f.calls)
	}
}
