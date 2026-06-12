package openai

import (
	"context"
	"net/http"
	"time"
)

// openaiSpeechModel implements SpeechModel.
type openaiSpeechModel struct {
	provider *openaiProvider
	modelID  string
}

func newSpeechModel(p *openaiProvider, modelID string) SpeechModel {
	return &openaiSpeechModel{provider: p, modelID: modelID}
}

func (m *openaiSpeechModel) ModelID() string  { return m.modelID }
func (m *openaiSpeechModel) Provider() string { return "openai.speech" }

func (m *openaiSpeechModel) DoGenerate(ctx context.Context, opts SpeechGenerateOptions) (*SpeechGenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	warnings := []Warning{}
	body := map[string]any{
		"model": m.modelID,
		"input": opts.Text,
		"voice": opts.Voice,
	}
	// Validate voice; fall back to default with a warning.
	if !isValidVoice(opts.Voice) {
		if opts.Voice != "" {
			warnings = append(warnings, Warning{Type: "unsupported", Feature: "voice", Message: "unsupported voice: " + opts.Voice})
		}
		body["voice"] = DefaultVoice
	}
	// Validate output format; fall back to default with a warning.
	outputFormat := opts.OutputFormat
	if outputFormat == "" {
		outputFormat = defaultOutputFormat
	}
	if !isValidOutputFormat(outputFormat) {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "outputFormat", Message: "unsupported outputFormat: " + outputFormat})
		outputFormat = defaultOutputFormat
	}
	body["response_format"] = outputFormat
	if opts.Speed != nil {
		body["speed"] = *opts.Speed
	}
	if opts.Instructions != nil {
		// instructions is supported only by gpt-4o-mini-tts.
		if m.modelID != "gpt-4o-mini-tts" {
			warnings = append(warnings, Warning{Type: "unsupported", Feature: "instructions", Message: "instructions is only supported by gpt-4o-mini-tts"})
		} else {
			body["instructions"] = *opts.Instructions
		}
	}
	// language is unsupported by the OpenAI TTS API.
	if v, ok := opts.ProviderOptions["openai"]; ok {
		if _, ok := v["language"]; ok {
			warnings = append(warnings, Warning{Type: "unsupported", Feature: "language", Message: "language is not supported by the OpenAI TTS API"})
		}
	}
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointAudioSpeech, encoded, opts.Headers)
	if err != nil {
		return nil, err
	}
	result := &SpeechGenerateResult{
		Audio:    resp.Body,
		Warnings: warnings,
		Request:  RequestMetadata{Body: encoded},
		Response: SpeechResponseMetadata{
			Timestamp: time.Now(),
			ModelID:   m.modelID,
			Headers:   http.Header{},
			Body:      resp.Body,
		},
	}
	return result, nil
}
