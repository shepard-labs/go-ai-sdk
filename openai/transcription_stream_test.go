package openai

import (
	"context"
	"testing"
)

// drainTranscriptionStream collects all parts from a transcription stream.
func drainTranscriptionStream(t *testing.T, m *openaiTranscriptionModel, opts TranscriptionOptions) []StreamPart {
	t.Helper()
	res, err := m.DoStream(context.Background(), opts)
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	var out []StreamPart
	for p := range res.Parts {
		out = append(out, p)
	}
	return out
}

func newTranscriptionStreamer(p *openaiProvider, modelID string) *openaiTranscriptionModel {
	return newTranscriptionModel(p, modelID).(*openaiTranscriptionModel)
}

// TestTranscriptionStreamDeltas verifies partial transcript deltas become
// text parts terminated by an end part and a finish part.
func TestTranscriptionStreamDeltas(t *testing.T) {
	stream := "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello\"}\n\n" +
		"data: {\"type\":\"transcript.text.delta\",\"delta\":\" world\"}\n\n" +
		"data: {\"type\":\"transcript.text.done\",\"text\":\"hello world\"}\n\n" +
		"data: [DONE]\n\n"
	p := newOpenAIForTest(&streamingFetcher{body: stream}, "https://example.test/v1")
	parts := drainTranscriptionStream(t, newTranscriptionStreamer(p, "gpt-4o-transcribe"), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	if len(parts) == 0 {
		t.Fatal("no parts")
	}
	if _, ok := parts[0].(StreamStart); !ok {
		t.Errorf("first part: %T", parts[0])
	}
	var deltas []string
	var sawTextStart, sawTextEnd, sawFinish bool
	for _, part := range parts {
		switch v := part.(type) {
		case StreamTextStart:
			sawTextStart = true
		case StreamTextDelta:
			deltas = append(deltas, v.Text)
		case StreamTextEnd:
			sawTextEnd = true
		case StreamFinish:
			sawFinish = true
		}
	}
	if !sawTextStart {
		t.Error("no StreamTextStart")
	}
	if len(deltas) != 2 || deltas[0] != "hello" || deltas[1] != " world" {
		t.Errorf("deltas: %q", deltas)
	}
	if !sawTextEnd {
		t.Error("no StreamTextEnd")
	}
	if !sawFinish {
		t.Error("no StreamFinish")
	}
}

// TestTranscriptionStreamDoneOnly verifies a stream with no deltas still
// surfaces the done text as a complete start/delta/end sequence followed
// by finish (done-only text must not be silently dropped).
func TestTranscriptionStreamDoneOnly(t *testing.T) {
	stream := "data: {\"type\":\"transcript.text.done\",\"text\":\"hi\"}\n\n" +
		"data: [DONE]\n\n"
	p := newOpenAIForTest(&streamingFetcher{body: stream}, "https://example.test/v1")
	parts := drainTranscriptionStream(t, newTranscriptionStreamer(p, "gpt-4o-transcribe"), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	var sawTextStart, sawTextEnd, sawFinish bool
	var deltas []string
	for _, part := range parts {
		switch v := part.(type) {
		case StreamTextStart:
			sawTextStart = true
		case StreamTextDelta:
			deltas = append(deltas, v.Text)
		case StreamTextEnd:
			sawTextEnd = true
		case StreamFinish:
			sawFinish = true
		}
	}
	if !sawTextStart {
		t.Error("no StreamTextStart for done-only text")
	}
	if len(deltas) != 1 || deltas[0] != "hi" {
		t.Errorf("deltas: %q, want [hi]", deltas)
	}
	if !sawTextEnd {
		t.Error("no StreamTextEnd for done-only text")
	}
	if !sawFinish {
		t.Error("no StreamFinish")
	}
}

// TestTranscriptionStreamError verifies a malformed event becomes a
// StreamError part and terminates the stream.
func TestTranscriptionStreamError(t *testing.T) {
	stream := "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hi\"}\n\n" +
		"data: not-json\n\n"
	p := newOpenAIForTest(&streamingFetcher{body: stream}, "https://example.test/v1")
	parts := drainTranscriptionStream(t, newTranscriptionStreamer(p, "gpt-4o-transcribe"), TranscriptionOptions{
		Audio:     []byte{0x00, 0x01},
		MediaType: "audio/wav",
	})
	var sawError bool
	var sawFinish bool
	for _, part := range parts {
		switch v := part.(type) {
		case StreamError:
			sawError = true
			if _, ok := v.Err.(InvalidResponseDataError); !ok {
				t.Errorf("error type: %T", v.Err)
			}
		case StreamFinish:
			sawFinish = true
		}
	}
	if !sawError {
		t.Error("no StreamError")
	}
	if sawFinish {
		t.Error("unexpected StreamFinish after error")
	}
}
