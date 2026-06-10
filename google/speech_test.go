package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// mockSpeechFetcher is a Fetcher that routes requests to a test server handler.
type mockSpeechFetcher struct {
	server *httptest.Server
}

func newSpeechMockFetcher(handler func(http.ResponseWriter, *http.Request)) *mockSpeechFetcher {
	m := &mockSpeechFetcher{}
	m.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler(w, r)
	}))
	return m
}

func (f *mockSpeechFetcher) Do(req *http.Request) (*http.Response, error) {
	return http.DefaultClient.Do(req)
}

func (f *mockSpeechFetcher) URL() string { return f.server.URL }

func (f *mockSpeechFetcher) Close() { f.server.Close() }

// stubProvider returns a googleProvider with the given Fetcher.
// The Fetcher's URL must match the provider's baseURL so requests are routed
// to the mock server, not the real internet.
func stubSpeechProvider(fetch *mockSpeechFetcher) *googleProvider {
	return &googleProvider{
		baseURL: fetch.URL(),
		name:    "google",
		headers: http.Header{},
		fetch:   fetch,
		apiKey:  "test-key",
		logger:  noopLogger{},
	}
}

// stubSpeechModel returns a googleSpeechModel for testing.
func stubSpeechModel(fetch *mockSpeechFetcher) *googleSpeechModel {
	return &googleSpeechModel{provider: stubSpeechProvider(fetch), modelID: "tts-001"}
}

// pcmData returns 1 second of 24000 Hz mono s16le silence as raw bytes.
func pcmData() []byte {
	// 24000 samples/s * 1 s * 2 bytes/sample = 48000 bytes
	b := make([]byte, 48000)
	return b
}

// pcmBase64 returns base64 of pcmData().
func pcmBase64() string {
	return base64.StdEncoding.EncodeToString(pcmData())
}

// ---- WAV header tests ----

func TestWAVHeader_44Bytes(t *testing.T) {
	t.Parallel()
	// The header itself (with empty PCM) is always exactly 44 bytes.
	hdr := wrapWAVHeader([]byte{}, 24000)
	if len(hdr) != 44 {
		t.Errorf("WAV header length = %d, want 44", len(hdr))
	}
}

func TestWAVHeader_RIFF(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	if string(hdr[0:4]) != "RIFF" {
		t.Errorf("hdr[0:4] = %q, want RIFF", string(hdr[0:4]))
	}
}

func TestWAVHeader_WAVE(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	if string(hdr[8:12]) != "WAVE" {
		t.Errorf("hdr[8:12] = %q, want WAVE", string(hdr[8:12]))
	}
}

func TestWAVHeader_Fmt(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	if string(hdr[12:16]) != "fmt " {
		t.Errorf("hdr[12:16] = %q, want 'fmt '", string(hdr[12:16]))
	}
}

func TestWAVHeader_Subchunk1Size(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	// 16 for PCM format
	if hdr[16] != 16 || hdr[17] != 0 || hdr[18] != 0 || hdr[19] != 0 {
		t.Errorf("subchunk1Size = %v, want 16", hdr[16:20])
	}
}

func TestWAVHeader_AudioFormatPCM(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	// 1 = PCM
	if hdr[20] != 1 || hdr[21] != 0 {
		t.Errorf("audioFormat = %v, want 1 (PCM)", hdr[20:22])
	}
}

func TestWAVHeader_Mono(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	// numChannels = 1
	if hdr[22] != 1 || hdr[23] != 0 {
		t.Errorf("numChannels = %v, want 1", hdr[22:24])
	}
}

func TestWAVHeader_SampleRate24000(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	// 24000 = 0x5DC0; 32-bit little-endian: [0xC0, 0x5D, 0x00, 0x00]
	want := [4]byte{192, 93, 0, 0}
	if hdr[24] != want[0] || hdr[25] != want[1] || hdr[26] != want[2] || hdr[27] != want[3] {
		t.Errorf("sampleRate bytes = %v, want %v", hdr[24:28], want)
	}
}

func TestWAVHeader_SampleRate44100(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 44100)
	// 44100 = 0xAC44; 32-bit little-endian: [0x44, 0xAC, 0x00, 0x00]
	want := [4]byte{68, 172, 0, 0}
	if hdr[24] != want[0] || hdr[25] != want[1] || hdr[26] != want[2] || hdr[27] != want[3] {
		t.Errorf("sampleRate bytes = %v, want %v", hdr[24:28], want)
	}
}

func TestWAVHeader_BytesPerSec(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	// 24000 * 1 * 16/8 = 48000 = 0xBB80; 32-bit little-endian: [0x80, 0xBB, 0x00, 0x00]
	want := [4]byte{128, 187, 0, 0}
	if hdr[28] != want[0] || hdr[29] != want[1] || hdr[30] != want[2] || hdr[31] != want[3] {
		t.Errorf("bytesPerSec = %v, want %v", hdr[28:32], want)
	}
}

func TestWAVHeader_BlockAlign(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	// 1 * 16/8 = 2
	if hdr[32] != 2 || hdr[33] != 0 {
		t.Errorf("blockAlign = %v, want 2", hdr[32:34])
	}
}

func TestWAVHeader_BitsPerSample16(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	// 16 bits
	if hdr[34] != 16 || hdr[35] != 0 {
		t.Errorf("bitsPerSample = %v, want 16", hdr[34:36])
	}
}

func TestWAVHeader_Data(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader(pcmData(), 24000)
	if string(hdr[36:40]) != "data" {
		t.Errorf("hdr[36:40] = %q, want 'data'", string(hdr[36:40]))
	}
}

func TestWAVHeader_DataSize(t *testing.T) {
	t.Parallel()
	raw := pcmData()
	hdr := wrapWAVHeader(raw, 24000)
	want := uint32(len(raw))
	got := uint32(hdr[40]) | uint32(hdr[41])<<8 | uint32(hdr[42])<<16 | uint32(hdr[43])<<24
	if got != want {
		t.Errorf("dataSize = %d, want %d", got, want)
	}
}

func TestWAVHeader_PCMFollowsHeader(t *testing.T) {
	t.Parallel()
	raw := pcmData()
	out := wrapWAVHeader(raw, 24000)
	if len(out) != 44+len(raw) {
		t.Fatalf("output length = %d, want %d", len(out), 44+len(raw))
	}
	for i, b := range raw {
		if out[44+i] != b {
			t.Errorf("out[44+%d] = %d, want %d (PCM byte mismatch)", i, out[44+i], b)
		}
	}
}

func TestWAVHeader_EmptyPCM(t *testing.T) {
	t.Parallel()
	hdr := wrapWAVHeader([]byte{}, 24000)
	if len(hdr) != 44 {
		t.Errorf("header length with empty PCM = %d, want 44", len(hdr))
	}
	// file size = 44 + 0 - 8 = 36
	want := uint32(36)
	got := uint32(hdr[4]) | uint32(hdr[5])<<8 | uint32(hdr[6])<<16 | uint32(hdr[7])<<24
	if got != want {
		t.Errorf("file size = %d, want %d", got, want)
	}
}

// ---- parseMimeRate tests ----

func TestParseMimeRate_24000(t *testing.T) {
	t.Parallel()
	got := parseMimeRate("audio/L16;rate=24000")
	if got != 24000 {
		t.Errorf("parseMimeRate = %d, want 24000", got)
	}
}

func TestParseMimeRate_44100(t *testing.T) {
	t.Parallel()
	got := parseMimeRate("audio/L16;rate=44100")
	if got != 44100 {
		t.Errorf("parseMimeRate = %d, want 44100", got)
	}
}

func TestParseMimeRate_NoRate(t *testing.T) {
	t.Parallel()
	got := parseMimeRate("audio/L16")
	if got != 0 {
		t.Errorf("parseMimeRate = %d, want 0", got)
	}
}

func TestParseMimeRate_Empty(t *testing.T) {
	t.Parallel()
	got := parseMimeRate("")
	if got != 0 {
		t.Errorf("parseMimeRate = %d, want 0", got)
	}
}

// ---- speechModelOptionsFromProviderOptions tests ----

func TestSpeechModelOptions_Default(t *testing.T) {
	t.Parallel()
	opts, recognized := speechModelOptionsFromProviderOptions(nil)
	if opts.MultiSpeakerVoiceConfig != nil {
		t.Error("MultiSpeakerVoiceConfig should be nil for nil merged")
	}
	if len(recognized) != 0 {
		t.Errorf("recognized = %v, want []", recognized)
	}
}

func TestSpeechModelOptions_MultiSpeakerVoiceConfig(t *testing.T) {
	t.Parallel()
	merged := map[string]any{
		"multiSpeakerVoiceConfig": map[string]any{
			"speakerVoiceConfigs": []any{
				map[string]any{
					"speaker": "speaker_1",
					"voiceConfig": map[string]any{
						"voiceName": "Kore",
					},
				},
				map[string]any{
					"speaker": "speaker_2",
					"voiceConfig": map[string]any{
						"voiceName": "Puck",
					},
				},
			},
		},
	}
	opts, recognized := speechModelOptionsFromProviderOptions(merged)
	if opts.MultiSpeakerVoiceConfig == nil {
		t.Fatal("MultiSpeakerVoiceConfig is nil")
	}
	svcs := opts.MultiSpeakerVoiceConfig.SpeakerVoiceConfigs
	if len(svcs) != 2 {
		t.Fatalf("len(svcs) = %d, want 2", len(svcs))
	}
	if svcs[0].Speaker != "speaker_1" || svcs[0].VoiceConfig.VoiceName != "Kore" {
		t.Errorf("svcs[0] = %+v, want Speaker=speaker_1 VoiceName=Kore", svcs[0])
	}
	if svcs[1].Speaker != "speaker_2" || svcs[1].VoiceConfig.VoiceName != "Puck" {
		t.Errorf("svcs[1] = %+v, want Speaker=speaker_2 VoiceName=Puck", svcs[1])
	}
	if len(recognized) != 1 || recognized[0] != "multiSpeakerVoiceConfig" {
		t.Errorf("recognized = %v, want [multiSpeakerVoiceConfig]", recognized)
	}
}

// ---- DoGenerate tests ----

// TestSpeech_DoGenerate_DefaultVoice_Kore verifies that DoGenerate uses Kore
// as the default voice when no voice is specified.
func TestSpeech_DoGenerate_DefaultVoice_Kore(t *testing.T) {
	t.Parallel()
	var capturedBody map[string]any
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			if err := decodeJSON(r.Body, &capturedBody); err != nil {
				t.Fatalf("failed to decode body: %v", err)
			}
			w.Write([]byte(`{
				"candidates": [{
					"content": {
						"parts": [{
							"inlineData": {
								"mimeType": "audio/L16;rate=24000",
								"data": "` + pcmBase64() + `"
							}
						}]
					}
				}]
			}`))
		}
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{Text: "hello"})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if len(result.Audio) == 0 {
		t.Error("Audio should not be empty")
	}

	// Verify Kore is used as default voice.
	genCfg, ok := capturedBody["generationConfig"].(map[string]any)
	if !ok {
		t.Fatal("generationConfig missing from body")
	}
	speechConfig, ok := genCfg["speechConfig"].(map[string]any)
	if !ok {
		t.Fatal("speechConfig missing from generationConfig")
	}
	voiceConfig, ok := speechConfig["voiceConfig"].(map[string]any)
	if !ok {
		t.Fatal("voiceConfig missing from speechConfig")
	}
	pvc, ok := voiceConfig["prebuiltVoiceConfig"].(map[string]any)
	if !ok {
		t.Fatal("prebuiltVoiceConfig missing from voiceConfig")
	}
	if pvc["voiceName"] != "Kore" {
		t.Errorf("voiceName = %q, want 'Kore'", pvc["voiceName"])
	}
}

// TestSpeech_DoGenerate_DefaultSampleRate_24000 verifies that the WAV header
// uses 24000 Hz when the API returns audio/L16;rate=24000.
func TestSpeech_DoGenerate_DefaultSampleRate_24000(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{Text: "hello"})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	pm := result.ProviderMetadata["google"]
	if pm == nil {
		t.Fatal("ProviderMetadata missing 'google' key")
	}
	meta := pm.(map[string]any)
	if meta["sampleRate"] != 24000 {
		t.Errorf("sampleRate = %v, want 24000", meta["sampleRate"])
	}
}

// TestSpeech_DoGenerate_WAVOutputFormat verifies that outputFormat="wav" wraps
// raw PCM in a 44-byte WAV header.
func TestSpeech_DoGenerate_WAVOutputFormat(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hello",
		OutputFormat: "wav",
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	// WAV header must be 44 bytes.
	if len(result.Audio) != 44+len(pcmData()) {
		t.Errorf("Audio length = %d, want %d (44-byte WAV header + PCM)", len(result.Audio), 44+len(pcmData()))
	}
	// First 4 bytes must be "RIFF".
	if string(result.Audio[0:4]) != "RIFF" {
		t.Errorf("Audio[0:4] = %q, want 'RIFF'", string(result.Audio[0:4]))
	}
	// Bytes 8-12 must be "WAVE".
	if string(result.Audio[8:12]) != "WAVE" {
		t.Errorf("Audio[8:12] = %q, want 'WAVE'", string(result.Audio[8:12]))
	}
}

// TestSpeech_DoGenerate_PCMOutputFormat verifies that outputFormat="pcm" returns
// raw PCM without a WAV header and includes a warning.
func TestSpeech_DoGenerate_PCMOutputFormat(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hello",
		OutputFormat: "pcm",
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	// PCM must not have a WAV header.
	if len(result.Audio) != len(pcmData()) {
		t.Errorf("Audio length = %d, want %d (raw PCM, no header)", len(result.Audio), len(pcmData()))
	}
	// Must include a pcm warning.
	found := false
	for _, w := range result.Warnings {
		if w.Feature == "outputFormat:pcm" {
			found = true
		}
	}
	if !found {
		t.Error("expected warning for pcm outputFormat")
	}
}

// TestSpeech_DoGenerate_MultiSpeakerVoiceConfig verifies that multiSpeakerVoiceConfig
// in ProviderOptions takes precedence over the voice field and omits the
// single voiceConfig from the wire body.
func TestSpeech_DoGenerate_MultiSpeakerVoiceConfig(t *testing.T) {
	t.Parallel()
	var capturedBody map[string]any
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			if err := decodeJSON(r.Body, &capturedBody); err != nil {
				t.Fatalf("failed to decode body: %v", err)
			}
			w.Write([]byte(`{
				"candidates": [{
					"content": {
						"parts": [{
							"inlineData": {
								"mimeType": "audio/L16;rate=24000",
								"data": "` + pcmBase64() + `"
							}
						}]
					}
				}]
			}`))
		}
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	_, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:            "hello",
		Voice:           "Femma", // should be ignored
		Instructions:    "narrate", // should be ignored
		ProviderOptions: ProviderOptions{"google": map[string]any{"multiSpeakerVoiceConfig": map[string]any{"speakerVoiceConfigs": []any{map[string]any{"speaker": "s1", "voiceConfig": map[string]any{"voiceName": "Kore"}}}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	genCfg, ok := capturedBody["generationConfig"].(map[string]any)
	if !ok {
		t.Fatal("generationConfig missing from body")
	}
	speechConfig, ok := genCfg["speechConfig"].(map[string]any)
	if !ok {
		t.Fatal("speechConfig missing from generationConfig")
	}
	// Must not have voiceConfig when multiSpeakerVoiceConfig is set.
	if _, ok := speechConfig["voiceConfig"]; ok {
		t.Error("voiceConfig should not be present when multiSpeakerVoiceConfig is set")
	}
	msvc, ok := speechConfig["multiSpeakerVoiceConfig"].(map[string]any)
	if !ok {
		t.Fatal("multiSpeakerVoiceConfig missing from speechConfig")
	}
	svcs, ok := msvc["speakerVoiceConfigs"].([]any)
	if !ok {
		t.Fatal("speakerVoiceConfigs missing from multiSpeakerVoiceConfig")
	}
	if len(svcs) != 1 {
		t.Fatalf("len(svcs) = %d, want 1", len(svcs))
	}
	svc := svcs[0].(map[string]any)
	if svc["speaker"] != "s1" {
		t.Errorf("speaker = %q, want 's1'", svc["speaker"])
	}
}

// TestSpeech_DoGenerate_Warnings_Speed verifies that a non-default speed
// produces an unsupported warning.
func TestSpeech_DoGenerate_Warnings_Speed(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:  "hello",
		Speed: 1.5,
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	found := false
	for _, w := range result.Warnings {
		if w.Feature == "speed" {
			found = true
		}
	}
	if !found {
		t.Error("expected warning for speed")
	}
}

// TestSpeech_DoGenerate_Warnings_Language verifies that a non-empty language
// field produces an unsupported warning.
func TestSpeech_DoGenerate_Warnings_Language(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:      "hello",
		Language:  "en",
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	found := false
	for _, w := range result.Warnings {
		if w.Feature == "language" {
			found = true
		}
	}
	if !found {
		t.Error("expected warning for language")
	}
}

// TestSpeech_DoGenerate_Warnings_UnknownOutputFormat verifies that an unknown
// outputFormat falls back to wav with a warning.
func TestSpeech_DoGenerate_Warnings_UnknownOutputFormat(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hello",
		OutputFormat: "mp3",
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	// Must still return WAV.
	if string(result.Audio[0:4]) != "RIFF" {
		t.Errorf("Audio[0:4] = %q, want 'RIFF' (fallback to wav)", string(result.Audio[0:4]))
	}
	// Must include a warning.
	found := false
	for _, w := range result.Warnings {
		if w.Feature == "outputFormat" {
			found = true
		}
	}
	if !found {
		t.Error("expected warning for unknown outputFormat")
	}
}

// TestSpeech_DoGenerate_Warnings_MultiSpeakerWithVoice verifies that setting
// both multiSpeakerVoiceConfig and Voice produces a warning.
func TestSpeech_DoGenerate_Warnings_MultiSpeakerWithVoice(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:            "hello",
		Voice:           "Femma",
		ProviderOptions: ProviderOptions{"google": map[string]any{"multiSpeakerVoiceConfig": map[string]any{"speakerVoiceConfigs": []any{map[string]any{"speaker": "s1", "voiceConfig": map[string]any{"voiceName": "Kore"}}}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	found := false
	for _, w := range result.Warnings {
		if w.Feature == "voice with multiSpeakerVoiceConfig" {
			found = true
		}
	}
	if !found {
		t.Error("expected warning for voice with multiSpeakerVoiceConfig")
	}
}

// TestSpeech_DoGenerate_Warnings_MultiSpeakerWithInstructions verifies that
// setting both multiSpeakerVoiceConfig and Instructions produces a warning.
func TestSpeech_DoGenerate_Warnings_MultiSpeakerWithInstructions(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"inlineData": {
							"mimeType": "audio/L16;rate=24000",
							"data": "` + pcmBase64() + `"
						}
					}]
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	result, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:            "hello",
		Instructions:    "narrate",
		ProviderOptions: ProviderOptions{"google": map[string]any{"multiSpeakerVoiceConfig": map[string]any{"speakerVoiceConfigs": []any{map[string]any{"speaker": "s1", "voiceConfig": map[string]any{"voiceName": "Kore"}}}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	found := false
	for _, w := range result.Warnings {
		if w.Feature == "instructions with multiSpeakerVoiceConfig" {
			found = true
		}
	}
	if !found {
		t.Error("expected warning for instructions with multiSpeakerVoiceConfig")
	}
}

// TestSpeech_DoGenerate_Instructions_Prepended verifies that Instructions is
// prepended to the text in the request body.
func TestSpeech_DoGenerate_Instructions_Prepended(t *testing.T) {
	t.Parallel()
	var capturedBody map[string]any
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			if err := decodeJSON(r.Body, &capturedBody); err != nil {
				t.Fatalf("failed to decode body: %v", err)
			}
			w.Write([]byte(`{
				"candidates": [{
					"content": {
						"parts": [{
							"inlineData": {
								"mimeType": "audio/L16;rate=24000",
								"data": "` + pcmBase64() + `"
							}
						}]
					}
				}]
			}`))
		}
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	_, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{
		Text:         "hello",
		Instructions: "read slowly",
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	contents, ok := capturedBody["contents"].([]any)
	if !ok || len(contents) == 0 {
		t.Fatal("contents missing from body")
	}
	parts, ok := contents[0].(map[string]any)["parts"].([]any)
	if !ok || len(parts) == 0 {
		t.Fatal("parts missing from content[0]")
	}
	text, ok := parts[0].(map[string]any)["text"].(string)
	if !ok {
		t.Fatal("text missing from parts[0]")
	}
	if text != "read slowly: hello" {
		t.Errorf("text = %q, want 'read slowly: hello'", text)
	}
}

// TestSpeech_DoGenerate_ResponseModalities_AUDIO verifies that the request body
// includes responseModalities: ["AUDIO"].
func TestSpeech_DoGenerate_ResponseModalities_AUDIO(t *testing.T) {
	t.Parallel()
	var capturedBody map[string]any
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			if err := decodeJSON(r.Body, &capturedBody); err != nil {
				t.Fatalf("failed to decode body: %v", err)
			}
			w.Write([]byte(`{
				"candidates": [{
					"content": {
						"parts": [{
							"inlineData": {
								"mimeType": "audio/L16;rate=24000",
								"data": "` + pcmBase64() + `"
							}
						}]
					}
				}]
			}`))
		}
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	_, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{Text: "hello"})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	genCfg, ok := capturedBody["generationConfig"].(map[string]any)
	if !ok {
		t.Fatal("generationConfig missing from body")
	}
	modalities, ok := genCfg["responseModalities"].([]any)
	if !ok {
		t.Fatal("responseModalities missing from generationConfig")
	}
	if len(modalities) != 1 || modalities[0] != "AUDIO" {
		t.Errorf("responseModalities = %v, want [AUDIO]", modalities)
	}
}

// TestSpeech_DoGenerate_EmptyAudio verifies that an empty audio response
// returns GOOGLE_SPEECH_GENERATION_ERROR.
func TestSpeech_DoGenerate_EmptyAudio(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": []
				}
			}]
		}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	_, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{Text: "hello"})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_SPEECH_GENERATION_ERROR" {
		t.Errorf("Type = %q, want GOOGLE_SPEECH_GENERATION_ERROR", apiErr.Type)
	}
}

// TestSpeech_DoGenerate_NoCandidates verifies that a response with no
// candidates returns GOOGLE_SPEECH_GENERATION_ERROR.
func TestSpeech_DoGenerate_NoCandidates(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"candidates": []}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	_, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{Text: "hello"})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Type != "GOOGLE_SPEECH_GENERATION_ERROR" {
		t.Errorf("Type = %q, want GOOGLE_SPEECH_GENERATION_ERROR", apiErr.Type)
	}
}

// TestSpeech_DoGenerate_APIError verifies that an HTTP error is returned as an APICallError.
func TestSpeech_DoGenerate_APIError(t *testing.T) {
	t.Parallel()
	mf := newSpeechMockFetcher(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": {"message": "internal error"}}`))
	})
	defer mf.Close()
	m := stubSpeechModel(mf)

	_, err := m.DoGenerate(context.Background(), SpeechGenerateOptions{Text: "hello"})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
}

// decodeJSON reads all bytes from r and unmarshals them into out.
func decodeJSON(r io.Reader, out *map[string]any) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, out)
}