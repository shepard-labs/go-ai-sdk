package openai

import "testing"

// TestRealtimeSessionConfigDefaultTranscriptionModel verifies that when
// InputAudioTranscription is provided without a model, the SDK injects
// "gpt-realtime-whisper" as the default.
func TestRealtimeSessionConfigDefaultTranscriptionModel(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	sess := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").BuildSessionConfig(SessionConfig{
		InputAudioFormat:        &AudioFormat{Type: "audio/pcm"},
		InputAudioTranscription: &InputAudioTranscription{},
	})
	audio, _ := sess["audio"].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	trans, _ := input["transcription"].(map[string]any)
	if trans == nil {
		t.Fatal("transcription not set")
	}
	if trans["model"] != "gpt-realtime-whisper" {
		t.Errorf("default transcription model: %v, want gpt-realtime-whisper", trans["model"])
	}
}

// TestRealtimeSessionConfigRespectsProvidedTranscriptionModel verifies
// that an explicit transcription model is preserved (no override).
func TestRealtimeSessionConfigRespectsProvidedTranscriptionModel(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	custom := "whisper-1"
	sess := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").BuildSessionConfig(SessionConfig{
		InputAudioFormat:        &AudioFormat{Type: "audio/pcm"},
		InputAudioTranscription: &InputAudioTranscription{Model: &custom},
	})
	audio, _ := sess["audio"].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	trans, _ := input["transcription"].(map[string]any)
	if trans["model"] != "whisper-1" {
		t.Errorf("transcription model: %v, want whisper-1", trans["model"])
	}
}

// TestRealtimeSessionConfigDisabledTurnDetection verifies that
// turn_detection type "disabled" maps to nil per spec.
func TestRealtimeSessionConfigDisabledTurnDetection(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	sess := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").BuildSessionConfig(SessionConfig{
		InputAudioFormat: &AudioFormat{Type: "audio/pcm"},
		TurnDetection:    &TurnDetection{Type: "disabled"},
	})
	audio, _ := sess["audio"].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	td, has := input["turn_detection"]
	if !has {
		t.Fatal("turn_detection key missing")
	}
	if td != nil {
		t.Errorf("disabled should map to nil, got: %v", td)
	}
}

// TestRealtimeSessionConfigServerVADMapping verifies the type mapping
// from "server-vad" (kebab) to "server_vad" (snake).
func TestRealtimeSessionConfigServerVADMapping(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	threshold := 0.5
	silence := 200
	prefix := 300
	sess := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").BuildSessionConfig(SessionConfig{
		InputAudioFormat: &AudioFormat{Type: "audio/pcm"},
		TurnDetection: &TurnDetection{
			Type:             "server-vad",
			Threshold:        &threshold,
			SilenceDurationMs: &silence,
			PrefixPaddingMs:   &prefix,
		},
	})
	audio, _ := sess["audio"].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	td, _ := input["turn_detection"].(map[string]any)
	if td["type"] != "server_vad" {
		t.Errorf("type: %v, want server_vad", td["type"])
	}
	if td["threshold"] != 0.5 {
		t.Errorf("threshold: %v", td["threshold"])
	}
	if td["silence_duration_ms"] != 200 {
		t.Errorf("silence_duration_ms: %v", td["silence_duration_ms"])
	}
	if td["prefix_padding_ms"] != 300 {
		t.Errorf("prefix_padding_ms: %v", td["prefix_padding_ms"])
	}
}

// TestRealtimeSessionConfigSemanticVADMapping verifies the type mapping
// from "semantic-vad" (kebab) to "semantic_vad" (snake).
func TestRealtimeSessionConfigSemanticVADMapping(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	sess := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").BuildSessionConfig(SessionConfig{
		InputAudioFormat: &AudioFormat{Type: "audio/pcm"},
		TurnDetection:    &TurnDetection{Type: "semantic-vad"},
	})
	audio, _ := sess["audio"].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	td, _ := input["turn_detection"].(map[string]any)
	if td["type"] != "semantic_vad" {
		t.Errorf("type: %v, want semantic_vad", td["type"])
	}
}
