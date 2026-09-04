package openai

import (
	"context"
	"strings"
	"testing"
)

// TestTranscriptionDoGenerateChunkingStrategy verifies ChunkingStrategy is
// forwarded as the multipart `chunking_strategy` field when set.
func TestTranscriptionDoGenerateChunkingStrategy(t *testing.T) {
	respBody := `{"text":"hi"}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	strategy := "auto"
	_, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:            []byte{0x00, 0x01},
		MediaType:        "audio/wav",
		ChunkingStrategy: &strategy,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if !strings.Contains(mf.captured, "chunking_strategy") || !strings.Contains(mf.captured, "auto") {
		t.Errorf("chunking_strategy not found in multipart body: %q", mf.captured)
	}
}

// TestTranscriptionDoGenerateNoChunkingStrategy verifies the
// `chunking_strategy` field is absent when ChunkingStrategy is unset.
func TestTranscriptionDoGenerateNoChunkingStrategy(t *testing.T) {
	respBody := `{"text":"hi"}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	_, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if strings.Contains(mf.captured, "chunking_strategy") {
		t.Errorf("chunking_strategy should be absent: %q", mf.captured)
	}
}

// TestTranscriptionDoGenerateDiarizedJSON verifies the diarize model uses
// the diarized_json response format and parses per-segment speakers.
func TestTranscriptionDoGenerateDiarizedJSON(t *testing.T) {
	respBody := `{"text":"hello world","language":"english","segments":[{"text":"hello","start":0.0,"end":0.5,"speaker":"SPEAKER_00"},{"text":"world","start":0.5,"end":1.0,"speaker":"SPEAKER_01"}]}`
	mf := &multipartJSONFetcher{body: respBody}
	p := newOpenAIForTest(mf, "https://example.test/v1")
	res, err := p.Transcription("gpt-4o-transcribe-diarize").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if !strings.Contains(mf.captured, "diarized_json") {
		t.Errorf("diarized_json not found in multipart body: %q", mf.captured)
	}
	if len(res.Segments) != 2 {
		t.Fatalf("segments: %d", len(res.Segments))
	}
	if res.Segments[0].Speaker == nil || *res.Segments[0].Speaker != "SPEAKER_00" {
		t.Errorf("seg[0] speaker: %v", res.Segments[0].Speaker)
	}
	if res.Segments[1].Speaker == nil || *res.Segments[1].Speaker != "SPEAKER_01" {
		t.Errorf("seg[1] speaker: %v", res.Segments[1].Speaker)
	}
}

// TestTranscriptionDoGenerateWordSpeakers verifies per-word `speaker`
// labels are parsed into segments when falling back to words.
func TestTranscriptionDoGenerateWordSpeakers(t *testing.T) {
	respBody := `{"text":"hi there","words":[{"word":"hi","start":0.0,"end":0.3,"speaker":"SPEAKER_00"},{"word":"there","start":0.3,"end":0.6}]}`
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
		t.Fatalf("segments: %d", len(res.Segments))
	}
	if res.Segments[0].Speaker == nil || *res.Segments[0].Speaker != "SPEAKER_00" {
		t.Errorf("seg[0] speaker: %v", res.Segments[0].Speaker)
	}
	if res.Segments[1].Speaker != nil {
		t.Errorf("seg[1] speaker should be nil: %v", *res.Segments[1].Speaker)
	}
}
