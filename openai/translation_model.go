package openai

import (
	"context"
	"mime/multipart"
)

// endpointAudioTranslations is the REST audio translations endpoint. There
// is no realtime translations endpoint, so translation stays REST-only.
const endpointAudioTranslations = "/audio/translations"

// TranslationOptions carries parameters for a translation call. It reuses
// TranscriptionOptions for the audio payload; TargetLanguage is accepted
// for API parity but unsupported by the REST endpoint (output is always
// English) and produces a warning when set.
type TranslationOptions struct {
	TranscriptionOptions
	TargetLanguage *string
}

// TranslationResult is the result of a translation call: translated text
// plus per-segment slices.
type TranslationResult = TranscriptionResult

// TranslationModel is the openai package's translation model interface.
type TranslationModel interface {
	ModelID() string
	Provider() string
	DoTranslate(ctx context.Context, opts TranslationOptions) (*TranslationResult, error)
}

// openaiTranslationModel implements TranslationModel.
type openaiTranslationModel struct {
	provider *openaiProvider
	modelID  string
}

func newTranslationModel(p *openaiProvider, modelID string) TranslationModel {
	return &openaiTranslationModel{provider: p, modelID: modelID}
}

func (m *openaiTranslationModel) ModelID() string  { return m.modelID }
func (m *openaiTranslationModel) Provider() string { return "openai.translation" }

// DoTranslate performs a REST audio translation call: audio in, translated
// (English) text and segments out.
func (m *openaiTranslationModel) DoTranslate(ctx context.Context, opts TranslationOptions) (*TranslationResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	// Language and ChunkingStrategy are transcription-only parameters with
	// no translations-endpoint equivalent; error instead of silently
	// dropping them. TargetLanguage stays a warning (output is English).
	if opts.Language != nil {
		return nil, InvalidPromptError{Message: "language is not supported for translation; the translations API always outputs English"}
	}
	if opts.ChunkingStrategy != nil {
		return nil, InvalidPromptError{Message: "chunking_strategy is not supported for translation"}
	}
	warnings := []Warning{}
	if opts.TargetLanguage != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "targetLanguage", Message: "target_language is not supported by the OpenAI translations API; output is English"})
	}
	build := func(w *multipart.Writer) error {
		if err := w.WriteField("model", m.modelID); err != nil {
			return err
		}
		if err := w.WriteField("response_format", "verbose_json"); err != nil {
			return err
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
	resp, err := m.provider.executeMultipart(ctx, endpointAudioTranslations, opts.Headers, build)
	if err != nil {
		return nil, err
	}
	return parseTranscriptionPayload(resp.Body, warnings, m.modelID)
}
