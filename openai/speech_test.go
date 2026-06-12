package openai

import (
	"context"
	"net/http"
	"testing"
)

func TestSpeechGenerateBasicRequestBody(t *testing.T) {
	respBody := `{}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hello world",
		Voice: "alloy",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, res.Request.Body)
	if body["model"] != "tts-1" {
		t.Errorf("model: %v", body["model"])
	}
	if body["input"] != "hello world" {
		t.Errorf("input: %v", body["input"])
	}
	if body["voice"] != "alloy" {
		t.Errorf("voice: %v", body["voice"])
	}
	if body["response_format"] != "mp3" {
		t.Errorf("response_format: %v", body["response_format"])
	}
}

func TestSpeechGenerateInstructionsDroppedForTts1(t *testing.T) {
	respBody := `{}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	instructions := "speak slowly"
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:        "hi",
		Voice:       "alloy",
		Instructions: &instructions,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, res.Request.Body)
	if _, has := body["instructions"]; has {
		t.Errorf("instructions should not be in body for tts-1")
	}
	hasWarn := false
	for _, w := range res.Warnings {
		if w.Feature == "instructions" {
			hasWarn = true
		}
	}
	if !hasWarn {
		t.Errorf("expected instructions warning for tts-1")
	}
}

func TestSpeechGenerateInstructionsSupportedForGpt4oMiniTts(t *testing.T) {
	respBody := `{}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	instructions := "speak slowly"
	res, err := p.Speech("gpt-4o-mini-tts").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hi",
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

func TestSpeechGenerateInvalidOutputFormatFallsBack(t *testing.T) {
	respBody := `{}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hi",
		Voice:        "alloy",
		OutputFormat: "wma",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, res.Request.Body)
	if body["response_format"] != "mp3" {
		t.Errorf("response_format: %v", body["response_format"])
	}
	hasWarn := false
	for _, w := range res.Warnings {
		if w.Feature == "outputFormat" {
			hasWarn = true
		}
	}
	if !hasWarn {
		t.Errorf("expected outputFormat warning")
	}
}

func TestSpeechGenerateSpeedPropagated(t *testing.T) {
	respBody := `{}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	speed := 1.5
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hi",
		Voice: "alloy",
		Speed: &speed,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, res.Request.Body)
	if body["speed"] != 1.5 {
		t.Errorf("speed: %v", body["speed"])
	}
}

func TestSpeechGenerateLanguageProviderOptionWarns(t *testing.T) {
	respBody := `{}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hi",
		Voice: "alloy",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"language": "en"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	hasWarn := false
	for _, w := range res.Warnings {
		if w.Feature == "language" {
			hasWarn = true
		}
	}
	if !hasWarn {
		t.Errorf("expected language warning")
	}
}
