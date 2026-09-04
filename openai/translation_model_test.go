package openai

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

// pathCapturingFetcher records the request path and multipart body, then
// returns a canned JSON fixture.
type pathCapturingFetcher struct {
	path     string
	captured string
	body     string
}

func (f *pathCapturingFetcher) Do(req *http.Request) (*http.Response, error) {
	f.path = req.URL.Path
	buf := make([]byte, 0, 1024)
	tmp := make([]byte, 1024)
	for {
		n, err := req.Body.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
		}
		if err != nil {
			break
		}
	}
	f.captured = string(buf)
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(f.body)),
	}, nil
}

// TestTranslationDoTranslate verifies audio translates to English text with
// segments via the /audio/translations endpoint.
func TestTranslationDoTranslate(t *testing.T) {
	f := &pathCapturingFetcher{body: `{"text":"hello world","language":"english","duration":1.0,"segments":[{"text":"hello world","start":0.0,"end":1.0}]}`}
	p := newOpenAIForTest(f, "https://example.test/v1")
	m := newTranslationModel(p, "whisper-1")
	if m.ModelID() != "whisper-1" {
		t.Errorf("ModelID: %q", m.ModelID())
	}
	if m.Provider() != "openai.translation" {
		t.Errorf("Provider: %q", m.Provider())
	}
	res, err := m.DoTranslate(context.Background(), TranslationOptions{
		TranscriptionOptions: TranscriptionOptions{
			Audio:     []byte{0x00, 0x01},
			MediaType: "audio/wav",
		},
	})
	if err != nil {
		t.Fatalf("DoTranslate: %v", err)
	}
	if f.path != "/v1/audio/translations" {
		t.Errorf("path: %q", f.path)
	}
	if res.Text != "hello world" {
		t.Errorf("Text: %q", res.Text)
	}
	if len(res.Segments) != 1 || res.Segments[0].Text != "hello world" {
		t.Errorf("Segments: %+v", res.Segments)
	}
	if len(res.Warnings) != 0 {
		t.Errorf("warnings: %v", res.Warnings)
	}
}

// TestTranslationDoTranslateTargetLanguageWarns verifies TargetLanguage
// produces an unsupported warning since the REST endpoint always outputs
// English.
func TestTranslationDoTranslateTargetLanguageWarns(t *testing.T) {
	f := &pathCapturingFetcher{body: `{"text":"hola"}`}
	p := newOpenAIForTest(f, "https://example.test/v1")
	m := newTranslationModel(p, "whisper-1")
	res, err := m.DoTranslate(context.Background(), TranslationOptions{
		TranscriptionOptions: TranscriptionOptions{
			Audio:     []byte{0x00, 0x01},
			MediaType: "audio/wav",
		},
		TargetLanguage: strP("es"),
	})
	if err != nil {
		t.Fatalf("DoTranslate: %v", err)
	}
	if res.Text != "hola" {
		t.Errorf("Text: %q", res.Text)
	}
	if len(res.Warnings) == 0 {
		t.Fatal("expected targetLanguage warning")
	}
	if !strings.Contains(res.Warnings[0].Message, "target_language") {
		t.Errorf("warning: %v", res.Warnings[0])
	}
}
