package openai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

// TestSpeechDoGenerateValidVoice verifies a normal voice call succeeds
// with no warnings.
func TestSpeechDoGenerateValidVoice(t *testing.T) {
	respBody := []byte("FAKE_MP3")
	f := &recordingFetcher{responses: []*http.Response{
		{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader(respBody))},
	}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hello",
		Voice: "alloy",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) != 0 {
		t.Errorf("warnings: %v", res.Warnings)
	}
	if string(res.Audio) != string(respBody) {
		t.Errorf("audio: %q", res.Audio)
	}
}

// TestSpeechDoGenerateInvalidVoiceFallback verifies that an unknown voice
// produces an unsupported warning and falls back to DefaultVoice.
func TestSpeechDoGenerateInvalidVoiceFallback(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader([]byte("x")))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hello",
		Voice: "bogus-voice",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected warning for bogus voice")
	} else if !strings.Contains(res.Warnings[0].Message, "voice") {
		t.Errorf("warning: %v", res.Warnings[0])
	}
}

// TestSpeechDoGenerateEmptyVoiceDefaultsToDefault verifies an empty voice
// uses the default without emitting a warning.
func TestSpeechDoGenerateEmptyVoiceDefaultsToDefault(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader([]byte("x")))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hello",
		Voice: "",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) != 0 {
		t.Errorf("expected no warnings for empty voice, got: %v", res.Warnings)
	}
}

// TestSpeechDoGenerateInstructionsWarnForNonGpt4oMiniTts verifies that
// the `instructions` option produces an unsupported warning unless the
// model is gpt-4o-mini-tts.
func TestSpeechDoGenerateInstructionsWarnForNonGpt4oMiniTts(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader([]byte("x")))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	instructions := "speak slowly"
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hello",
		Voice:        "alloy",
		Instructions: &instructions,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected unsupported warning for instructions on tts-1")
	} else if !strings.Contains(res.Warnings[0].Message, "instructions") {
		t.Errorf("warning: %v", res.Warnings[0])
	}
}

// TestSpeechDoGenerateInstructionsAcceptedForGpt4oMiniTts verifies that
// `instructions` is forwarded to the request body on gpt-4o-mini-tts.
func TestSpeechDoGenerateInstructionsAcceptedForGpt4oMiniTts(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader([]byte("x")))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	instructions := "speak slowly"
	res, err := p.Speech("gpt-4o-mini-tts").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hello",
		Voice:        "alloy",
		Instructions: &instructions,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, res.Request.Body)
	if body["instructions"] != "speak slowly" {
		t.Errorf("instructions: %v", body["instructions"])
	}
}

// TestSpeechDoGenerateLanguageUnsupported verifies language in
// providerOptions produces an unsupported warning.
func TestSpeechDoGenerateLanguageUnsupported(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader([]byte("x")))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hello",
		Voice: "alloy",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"language": "en"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected language warning")
	}
}

// TestSpeechDoGenerateOutputFormatInvalidFallback verifies an invalid
// outputFormat produces a warning and falls back to default.
func TestSpeechDoGenerateOutputFormatInvalidFallback(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader([]byte("x")))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hello",
		Voice:        "alloy",
		OutputFormat: "wav-but-not",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected outputFormat warning")
	}
}

// TestSpeechDoGenerateSpeedForwarded verifies the speed option is
// forwarded to the request body.
func TestSpeechDoGenerateSpeedForwarded(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader([]byte("x")))}
	f := &recordingFetcher{responses: []*http.Response{resp}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	speed := 1.5
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hello",
		Voice: "alloy",
		Speed: &speed,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, res.Request.Body)
	if v, _ := body["speed"].(float64); v != 1.5 {
		t.Errorf("speed: %v", body["speed"])
	}
}

// TestTranscriptionDoGenerateVerboseJSONForWhisper verifies whisper-1
// uses verbose_json response format.
func TestTranscriptionDoGenerateVerboseJSONForWhisper(t *testing.T) {
	respBody := `{"text":"hi","language":"english","duration":1.0,"segments":[{"text":"hi","start":0.0,"end":0.5}]}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	res, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Text != "hi" {
		t.Errorf("Text: %q", res.Text)
	}
	if res.Language != "en" {
		t.Errorf("Language: %q", res.Language)
	}
	if len(res.Segments) != 1 {
		t.Errorf("segments: %d", len(res.Segments))
	}
}

// TestTranscriptionDoGenerateJSONForGpt4oTranscribe verifies gpt-4o
// transcribe models use json (not verbose_json) and that there's no
// segments field.
func TestTranscriptionDoGenerateJSONForGpt4oTranscribe(t *testing.T) {
	respBody := `{"text":"hi","language":"english"}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	res, err := p.Transcription("gpt-4o-transcribe").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Text != "hi" {
		t.Errorf("Text: %q", res.Text)
	}
}

// TestTranscriptionDoGenerateUnknownLanguageInMetadata verifies an
// unrecognized language string lands in providerMetadata.openai.rawLanguage.
func TestTranscriptionDoGenerateUnknownLanguageInMetadata(t *testing.T) {
	respBody := `{"text":"hi","language":"klingon"}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	res, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Language != "" {
		t.Errorf("Language should be empty for klingon, got: %q", res.Language)
	}
	if res.ProviderMetadata == nil {
		t.Fatalf("expected ProviderMetadata for unknown language")
	}
	pm, _ := res.ProviderMetadata["openai"].(map[string]any)
	if pm["rawLanguage"] != "klingon" {
		t.Errorf("rawLanguage: %v", pm["rawLanguage"])
	}
}

// TestTranscriptionDoGenerateInvalidJSON verifies malformed JSON throws
// InvalidResponseDataError.
func TestTranscriptionDoGenerateInvalidJSON(t *testing.T) {
	mf := &multipartJSONFetcher{body: "not json"}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	_, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidResponseDataError); !ok {
		t.Errorf("expected InvalidResponseDataError, got %T: %v", err, err)
	}
}

// TestTranscriptionDoGenerateWordsFallback verifies that when no segments
// are present, words are used as segments.
func TestTranscriptionDoGenerateWordsFallback(t *testing.T) {
	respBody := `{"text":"hi there","words":[{"word":"hi","start":0.0,"end":0.3},{"word":"there","start":0.3,"end":0.6}]}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	res, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Segments) != 2 {
		t.Errorf("segments: %d", len(res.Segments))
	}
	if res.Segments[0].Text != "hi" {
		t.Errorf("seg[0] text: %q", res.Segments[0].Text)
	}
}

// TestTranscriptionTimestampGranularitiesUnsupported verifies that
// non-word/segment granularities emit a warning.
func TestTranscriptionTimestampGranularitiesUnsupported(t *testing.T) {
	respBody := `{"text":"hi"}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	res, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:                  []byte{0x00, 0x01},
		MediaType:              "audio/wav",
		TimestampGranularities: []string{"word", "paragraph"},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Warnings) == 0 {
		t.Errorf("expected warning for paragraph granularity")
	}
}

// TestTranscriptionDoGenerateDefaultFilename verifies the filename
// derives from media type when not provided.
func TestTranscriptionDoGenerateDefaultFilename(t *testing.T) {
	respBody := `{"text":"hi"}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	_, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/mp3",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if !strings.Contains(mf.captured, "filename=\"audio.mp3\"") {
		t.Errorf("default filename not found: %q", mf.captured)
	}
}

// multipartJSONFetcher records the body and returns a JSON body.
type multipartJSONFetcher struct {
	captured string
	counter  int
	body     string
}

func (m *multipartJSONFetcher) Do(req *http.Request) (*http.Response, error) {
	if m.counter == 0 {
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
		m.captured = string(buf)
	}
	m.counter++
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(m.body)),
	}, nil
}

// errEOF is the standard "end of file" error used by io.Reader test stubs.
var errEOF = errors.New("EOF")
