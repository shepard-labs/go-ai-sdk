package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// TestSkillsUploadEmptyFilesRejected verifies that a SkillsUploadOptions
// with no files is rejected with InvalidPromptError.
func TestSkillsUploadEmptyFilesRejected(t *testing.T) {
	respBody := `{"id":"skill-1","name":"s"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		Files: nil,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
	// Verify no request fired.
	if f.calls != 0 {
		t.Errorf("expected 0 calls, got %d", f.calls)
	}
}

// TestSkillsUploadDefaultFilename verifies that a file with no Path falls
// back to "skill.zip" in the multipart form.
func TestSkillsUploadDefaultFilename(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		Files: []SkillsFile{
			{Data: []byte("zipdata"), MediaType: "application/zip"},
		},
	})
	if err != nil {
		t.Fatalf("UploadSkill: %v", err)
	}
	if !strings.Contains(f.captured, "filename=\"skill.zip\"") {
		t.Errorf("default filename not found: %q", f.captured)
	}
}

// TestSkillsUploadDisplayTitleWarning verifies that providing
// DisplayTitle adds an "unsupported" warning to the result.
func TestSkillsUploadDisplayTitleWarning(t *testing.T) {
	respBody := `{"id":"skill-1","name":"s","latest_version":"v1"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		DisplayTitle: "My Skill",
		Files: []SkillsFile{
			{Path: "a.zip", Data: []byte("zip"), MediaType: "application/zip"},
		},
	})
	if err != nil {
		t.Fatalf("UploadSkill: %v", err)
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected warning for DisplayTitle, got none")
	} else {
		if !strings.Contains(res.Warnings[0].Message, "displayTitle") {
			t.Errorf("warning message: %v", res.Warnings[0])
		}
	}
}

// TestSkillsUploadInvalidJSONResponse verifies that a malformed JSON
// response throws InvalidResponseDataError.
func TestSkillsUploadInvalidJSONResponse(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, "not json")}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		Files: []SkillsFile{
			{Path: "a.zip", Data: []byte("zip"), MediaType: "application/zip"},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T: %v", err, err)
	}
}

// TestFilesUploadInvalidJSONResponse verifies that a malformed files
// response throws InvalidResponseDataError.
func TestFilesUploadInvalidJSONResponse(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, "not json")}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data:      []byte("x"),
		Filename:  "x.txt",
		MediaType: "text/plain",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T: %v", err, err)
	}
}

// TestFilesUploadDefaultFilename verifies that an empty filename falls
// back to "file" in the multipart form.
func TestFilesUploadDefaultFilename(t *testing.T) {
	f := &multipartCapturingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data:      []byte("x"),
		Filename:  "",
		MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	if !strings.Contains(f.captured, "filename=\"file\"") {
		t.Errorf("default filename not found: %q", f.captured)
	}
}

// TestFilesUploadResponseMappingStatusError verifies that a response with
// status="error" is captured into provider metadata.
func TestFilesUploadResponseMappingStatusError(t *testing.T) {
	respBody := `{"id":"file-1","object":"file","filename":"foo.txt","purpose":"assistants","status":"error"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data: []byte("x"), Filename: "foo.txt", MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	pm, _ := res.ProviderMetadata["openai"].(map[string]any)
	if pm["status"] != "error" {
		t.Errorf("status: %v", pm["status"])
	}
	if res.Filename != "foo.txt" {
		t.Errorf("Filename: %q", res.Filename)
	}
}

// TestFilesUploadResponseExpiresAt verifies that a response carrying
// expires_at populates that field in provider metadata.
func TestFilesUploadResponseExpiresAt(t *testing.T) {
	respBody := `{"id":"file-1","object":"file","filename":"f.txt","purpose":"assistants","expires_at":1700003600}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data: []byte("x"), Filename: "f.txt", MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	pm, _ := res.ProviderMetadata["openai"].(map[string]any)
	if v, _ := pm["expiresAt"].(int64); v != 1700003600 {
		t.Errorf("expiresAt: %v", pm["expiresAt"])
	}
}

// TestFilesUploadResponseUsesDefaultFilename verifies that the response
// filename wins when present, even if a different one was supplied.
func TestFilesUploadResponseUsesDefaultFilename(t *testing.T) {
	respBody := `{"id":"file-1","object":"file","filename":"from-server.txt","purpose":"assistants"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data: []byte("x"), Filename: "local.txt", MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	if res.Filename != "from-server.txt" {
		t.Errorf("Filename: %q, want from-server.txt", res.Filename)
	}
}

// TestSkillsUploadMetadata verifies that the response fields
// defaultVersion, updatedAt, description are surfaced.
func TestSkillsUploadMetadata(t *testing.T) {
	respBody := `{"id":"skill-1","name":"MySkill","description":"desc","default_version":"v3","latest_version":"v3","created_at":1700000000,"updated_at":1700001000}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		Files: []SkillsFile{
			{Path: "a.zip", Data: []byte("z"), MediaType: "application/zip"},
		},
	})
	if err != nil {
		t.Fatalf("UploadSkill: %v", err)
	}
	if res.Name != "MySkill" {
		t.Errorf("Name: %q", res.Name)
	}
	if res.Description != "desc" {
		t.Errorf("Description: %q", res.Description)
	}
	if res.LatestVersion != "v3" {
		t.Errorf("LatestVersion: %q", res.LatestVersion)
	}
	pm, _ := res.ProviderMetadata["openai"].(map[string]any)
	if v, _ := pm["defaultVersion"].(string); v != "v3" {
		t.Errorf("defaultVersion: %v", pm["defaultVersion"])
	}
	if v, _ := pm["updatedAt"].(int64); v != 1700001000 {
		t.Errorf("updatedAt: %v", pm["updatedAt"])
	}
}
