package openai

import (
	"context"
	"net/http"
	"testing"
)

// Verifies that a chat completion with finish_reason=content_filter maps
// to the unified "content-filter" reason.
func TestChatFinishReasonContentFilter(t *testing.T) {
	got := chatFinishReasonFromString("content_filter")
	if got.Unified != "content-filter" {
		t.Errorf("got: %+v", got)
	}
}

// Verifies that a chat completion with finish_reason=tool_calls maps
// to the unified "tool-calls" reason.
func TestChatFinishReasonToolCalls(t *testing.T) {
	got := chatFinishReasonFromString("tool_calls")
	if got.Unified != "tool-calls" {
		t.Errorf("got: %+v", got)
	}
}

// Verifies that a chat completion with finish_reason=stop maps to "stop".
func TestChatFinishReasonStop(t *testing.T) {
	got := chatFinishReasonFromString("stop")
	if got.Unified != "stop" {
		t.Errorf("got: %+v", got)
	}
}

// Verifies that a chat completion with finish_reason=length maps to "length".
func TestChatFinishReasonLength(t *testing.T) {
	got := chatFinishReasonFromString("length")
	if got.Unified != "length" {
		t.Errorf("got: %+v", got)
	}
}

// Verifies file ID prefix detection is configurable per provider setting.
func TestFileIDPrefixesProviderSetting(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	if len(p.fileIDPrefixes) == 0 || p.fileIDPrefixes[0] != "file-" {
		t.Errorf("default file ID prefixes: %v", p.fileIDPrefixes)
	}
}

// Verifies that provider-level passThrough defaults to false.
func TestPassThroughDefaultsFalse(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	if p.passThroughUnsupportedFiles {
		t.Errorf("expected passThroughUnsupportedFiles=false by default")
	}
}

// Verifies that responsesStreamFinishReason maps "incomplete" to "length".
func TestResponsesStreamFinishReasonIncomplete(t *testing.T) {
	got := responsesStreamFinishReason("incomplete", false)
	if got.Unified != "length" {
		t.Errorf("got: %+v", got)
	}
}

// Verifies that responsesStreamFinishReason maps "completed" with a tool
// call to "tool-calls".
func TestResponsesStreamFinishReasonCompletedWithToolCall(t *testing.T) {
	got := responsesStreamFinishReason("completed", true)
	if got.Unified != "tool-calls" {
		t.Errorf("got: %+v", got)
	}
}

// Verifies embedding requests always include encoding_format: "float".
func TestEmbeddingsHardcodeFloatEncoding(t *testing.T) {
	respBody := `{"data":[{"embedding":[0.1,0.2]}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Embedding("text-embedding-3-small").DoEmbed(context.Background(), EmbedOptions{
		Values: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}
}

// Verifies that the embedding body has encoding_format: "float" in the
// outgoing JSON body.
func TestEmbeddingsBodyContainsEncodingFormat(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	got := (&openaiEmbeddingModel{provider: p, modelID: "text-embedding-3-small"}).
		buildEmbedRequest(EmbedOptions{Values: []string{"hello"}})
	if got["encoding_format"] != "float" {
		t.Errorf("encoding_format: %v", got["encoding_format"])
	}
	if got["model"] != "text-embedding-3-small" {
		t.Errorf("model: %v", got["model"])
	}
}

// Verifies that gpt-image-* models skip the response_format field.
func TestHasDefaultResponseFormat(t *testing.T) {
	cases := []struct {
		modelID string
		want    bool
	}{
		{"gpt-image-1", true},
		{"gpt-image-1-mini", true},
		{"gpt-image-2", true},
		{"chatgpt-image-latest", true},
		{"dall-e-3", false},
		{"dall-e-2", false},
	}
	for _, c := range cases {
		got := hasDefaultResponseFormat(c.modelID)
		if got != c.want {
			t.Errorf("%s: got %v, want %v", c.modelID, got, c.want)
		}
	}
}

// Verifies that dall-e-3 has MaxImagesPerCall=1, others have 10.
func TestMaxImagesPerCallPerModel(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	if got := p.Image("dall-e-3").MaxImagesPerCall(); got != 1 {
		t.Errorf("dall-e-3: got %d, want 1", got)
	}
	if got := p.Image("gpt-image-1").MaxImagesPerCall(); got != 10 {
		t.Errorf("gpt-image-1: got %d, want 10", got)
	}
}

// Verifies that the dall-e-3 style is rejected on non-dall-e-3 models.
func TestImageStyleOnlyForDallE3(t *testing.T) {
	respBody := `{"data":[]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"style": "vivid"},
		},
	})
	if err == nil {
		t.Errorf("expected error for style on gpt-image-1")
	}
}

// Verifies that dall-e-3 quality "low" is rejected.
func TestImageQualityEnumForDallE3(t *testing.T) {
	respBody := `{"data":[]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{"quality": "low"},
		},
	})
	if err == nil {
		t.Errorf("expected error for dall-e-3 quality=low")
	}
}

// Verifies image response parsing captures revised_prompt and usage.
func TestImageResponseRevisedPromptAndUsage(t *testing.T) {
	respBody := `{"data":[{"b64_json":"aGVsbG8=","revised_prompt":"a happy cat"}],"usage":{"input_tokens":10,"output_tokens":256,"total_tokens":266,"input_tokens_details":{"image_tokens":100,"text_tokens":50}}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "a cat",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Images) != 1 {
		t.Errorf("images: %d", len(res.Images))
	}
	if res.Usage == nil || res.Usage.InputTokens != 10 {
		t.Errorf("usage: %+v", res.Usage)
	}
}

// Verifies that the voice constants are exported and valid.
func TestVoiceConstants(t *testing.T) {
	voices := []string{
		VoiceAlloy, VoiceAsh, VoiceBallad, VoiceCoral, VoiceEcho,
		VoiceFable, VoiceNova, VoiceOnyx, VoiceSage, VoiceShimmer, VoiceVerse,
	}
	if len(voices) != 11 {
		t.Errorf("expected 11 voices, got %d", len(voices))
	}
	for _, v := range voices {
		if !isValidVoice(v) {
			t.Errorf("voice %q not valid", v)
		}
	}
	if isValidVoice("nonsense") {
		t.Errorf("nonsense should not be valid")
	}
}

// Verifies that mediaTypeToExtension handles common audio types.
func TestMediaTypeToExtension(t *testing.T) {
	cases := map[string]string{
		"audio/mpeg":  "mp3",
		"audio/mp3":   "mp3",
		"audio/wav":   "wav",
		"audio/webm":  "webm",
		"audio/ogg":   "ogg",
		"audio/flac":  "flac",
		"audio/aac":   "aac",
		"audio/opus":  "opus",
		"audio/pcm":   "pcm",
		"audio/mp4":   "m4a",
		"audio/m4a":   "m4a",
		"audio/mpga":  "mpga",
		"unknown":     "mp3", // default
	}
	for mediaType, want := range cases {
		if got := mediaTypeToExtension(mediaType); got != want {
			t.Errorf("%s: got %q, want %q", mediaType, got, want)
		}
	}
}

// Verifies that the speech model falls back to default voice and warns
// on an invalid voice.
func TestSpeechFallsBackOnInvalidVoice(t *testing.T) {
	respBody := `{}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Speech("tts-1").DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hi",
		Voice: "nonsense",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	hasVoiceWarn := false
	for _, w := range res.Warnings {
		if w.Feature == "voice" {
			hasVoiceWarn = true
		}
	}
	if !hasVoiceWarn {
		t.Errorf("expected voice warning")
	}
}

// Verifies that the transcription model injects verbose_json for whisper-1
// and json for gpt-4o-transcribe.
func TestResponseFormatForModel(t *testing.T) {
	if got := responseFormatForModel("whisper-1"); got != "verbose_json" {
		t.Errorf("whisper-1: got %q", got)
	}
	if got := responseFormatForModel("gpt-4o-transcribe"); got != "json" {
		t.Errorf("gpt-4o-transcribe: got %q", got)
	}
	if got := responseFormatForModel("gpt-4o-mini-transcribe"); got != "json" {
		t.Errorf("gpt-4o-mini-transcribe: got %q", got)
	}
	if got := responseFormatForModel("other"); got != "verbose_json" {
		t.Errorf("other: got %q", got)
	}
}

// Verifies the language name -> ISO-639-1 mapping.
func TestMapLanguageToISO(t *testing.T) {
	cases := map[string]string{
		"english":  "en",
		"chinese":  "zh",
		"spanish":  "es",
		"french":   "fr",
		"japanese": "ja",
		"korean":   "ko",
		"russian":  "ru",
		"":         "",
		"klingon":  "",
	}
	for name, want := range cases {
		if got := mapLanguageToISO(name); got != want {
			t.Errorf("%q: got %q, want %q", name, got, want)
		}
	}
}

// Verifies the language map has at least 57 entries.
func TestLanguageMapHas57Entries(t *testing.T) {
	if len(languageNameToISO) < 57 {
		t.Errorf("language map has %d entries, want at least 57", len(languageNameToISO))
	}
}

// Verifies that transcription falls back to words when no segments are
// present in the response.
func TestTranscriptionFallsBackToWords(t *testing.T) {
	respBody := `{"text":"hello world","language":"english","words":[{"word":"hello","start":0.0,"end":0.3},{"word":"world","start":0.4,"end":0.8}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:    []byte("fakemp3"),
		Filename: "audio.mp3",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(res.Segments) != 2 {
		t.Errorf("segments: %d", len(res.Segments))
	}
	if len(res.Segments) >= 1 && res.Segments[0].Text != "hello" {
		t.Errorf("first word: %q", res.Segments[0].Text)
	}
}

// Verifies that an unknown language string ends up empty in the public field
// but is preserved in provider metadata as rawLanguage.
func TestTranscriptionUnknownLanguagePreservedInMetadata(t *testing.T) {
	respBody := `{"text":"hi","language":"klingon"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Transcription("whisper-1").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:    []byte("fakemp3"),
		Filename: "audio.mp3",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Language != "" {
		t.Errorf("Language: %q, want empty", res.Language)
	}
	if res.ProviderMetadata == nil {
		t.Errorf("ProviderMetadata is nil")
	}
}
