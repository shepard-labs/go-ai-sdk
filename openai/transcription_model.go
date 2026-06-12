package openai

import (
	"context"
	"encoding/json"
	"mime/multipart"
)

// openaiTranscriptionModel implements TranscriptionModel.
type openaiTranscriptionModel struct {
	provider *openaiProvider
	modelID  string
}

func newTranscriptionModel(p *openaiProvider, modelID string) TranscriptionModel {
	return &openaiTranscriptionModel{provider: p, modelID: modelID}
}

func (m *openaiTranscriptionModel) ModelID() string  { return m.modelID }
func (m *openaiTranscriptionModel) Provider() string { return "openai.transcription" }

// responseFormatForModel returns the response_format the SDK should inject
// for this model. whisper-1 + others use verbose_json (which yields
// segments/words); the gpt-4o transcribe models use json.
func responseFormatForModel(modelID string) string {
	switch modelID {
	case "gpt-4o-transcribe", "gpt-4o-mini-transcribe":
		return "json"
	}
	return "verbose_json"
}

func (m *openaiTranscriptionModel) DoGenerate(ctx context.Context, opts TranscriptionOptions) (*TranscriptionResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	warnings := []Warning{}
	responseFormat := responseFormatForModel(m.modelID)
	build := func(w *multipart.Writer) error {
		if err := w.WriteField("model", m.modelID); err != nil {
			return err
		}
		if err := w.WriteField("response_format", responseFormat); err != nil {
			return err
		}
		if opts.Language != nil {
			if err := w.WriteField("language", *opts.Language); err != nil {
				return err
			}
		}
		if opts.Prompt != nil {
			if err := w.WriteField("prompt", *opts.Prompt); err != nil {
				return err
			}
		}
		if opts.Temperature != nil {
			if err := w.WriteField("temperature", jsonFloat(*opts.Temperature)); err != nil {
				return err
			}
		}
		if len(opts.TimestampGranularities) > 0 {
			for _, tg := range opts.TimestampGranularities {
				if tg != "word" && tg != "segment" {
					warnings = append(warnings, Warning{Type: "unsupported", Feature: "timestampGranularities", Message: "unsupported value: " + tg})
				}
				if err := w.WriteField("timestamp_granularities[]", tg); err != nil {
					return err
				}
			}
		}
		if len(opts.Include) > 0 {
			for _, inc := range opts.Include {
				if err := w.WriteField("include[]", inc); err != nil {
					return err
				}
			}
		}
		// Filename derives from MediaType if not explicitly set.
		filename := opts.Filename
		if filename == "" {
			ext := mediaTypeToExtension(opts.MediaType)
			filename = "audio." + ext
		}
		part, err := w.CreateFormFile("file", filename)
		if err != nil {
			return err
		}
		_, err = part.Write(opts.Audio)
		return err
	}
	resp, err := m.provider.executeMultipart(ctx, endpointAudioTranscriptions, opts.Headers, build)
	if err != nil {
		return nil, err
	}
	var raw map[string]any
	if err := json.Unmarshal(resp.Body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse: " + err.Error()}
	}
	result := &TranscriptionResult{
		Warnings: warnings,
		Response: ResponseMetadata{Body: resp.Body, ModelID: m.modelID},
	}
	if text, ok := raw["text"].(string); ok {
		result.Text = text
	}
	if language, ok := raw["language"].(string); ok {
		// Map full name → ISO-639-1.
		result.Language = mapLanguageToISO(language)
		if result.Language == "" {
			// Unknown / unrecognized full name. Pass through the raw string
			// in provider metadata so the caller can still see what came
			// back; the public Language field stays empty.
			result.ProviderMetadata = ProviderMetadata{
				"openai": map[string]any{
					"rawLanguage": language,
				},
			}
		}
	}
	if segments, ok := raw["segments"].([]any); ok && len(segments) > 0 {
		for _, s := range segments {
			sm, ok := s.(map[string]any)
			if !ok {
				continue
			}
			seg := TranscriptionSegment{}
			if t, ok := sm["text"].(string); ok {
				seg.Text = t
			}
			if start, ok := sm["start"].(float64); ok {
				seg.StartSecond = start
			}
			if end, ok := sm["end"].(float64); ok {
				seg.EndSecond = end
			}
			result.Segments = append(result.Segments, seg)
		}
	} else if words, ok := raw["words"].([]any); ok {
		// Fall back to words when no segments are present.
		for _, w := range words {
			wm, ok := w.(map[string]any)
			if !ok {
				continue
			}
			seg := TranscriptionSegment{}
			if t, ok := wm["word"].(string); ok {
				seg.Text = t
			}
			if start, ok := wm["start"].(float64); ok {
				seg.StartSecond = start
			}
			if end, ok := wm["end"].(float64); ok {
				seg.EndSecond = end
			}
			result.Segments = append(result.Segments, seg)
		}
	}
	if d, ok := raw["duration"].(float64); ok {
		result.Duration = &d
	}
	return result, nil
}

func jsonFloat(v float64) string {
	b, _ := json.Marshal(v)
	return string(b)
}
