package google

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"regexp"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// googleSpeechModel implements the Google Generative AI speech model interface.
type googleSpeechModel struct {
	provider *googleProvider
	modelID  string
}

// ModelID returns the model's ID string.
func (m *googleSpeechModel) ModelID() string { return m.modelID }

// Provider returns the provider name suffix.
func (m *googleSpeechModel) Provider() string { return m.provider.name + ".speech" }

// DoGenerate performs a speech synthesis call. It posts to :generateContent
// with responseModalities: ["AUDIO"] and speechConfig, then wraps the returned
// audio bytes in a WAV header if outputFormat is "wav".
func (m *googleSpeechModel) DoGenerate(ctx context.Context, opts SpeechGenerateOptions) (*SpeechGenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}

	rawOpts := cloneProviderOptions(opts.ProviderOptions)
	vertexLike := isVertexLike(m.provider.baseURL, m.provider.useVertexAIHeaders, rawOpts)
	merged := mergeGoogleNamespaces(rawOpts, vertexLike)
	speechOpts, _ := speechModelOptionsFromProviderOptions(merged)

	var warnings []Warning
	if speechOpts.MultiSpeakerVoiceConfig != nil {
		if opts.Voice != "" {
			warnings = append(warnings, Warning{
				Type:    "unsupported",
				Feature: "voice with multiSpeakerVoiceConfig",
				Details: "voice is ignored when multiSpeakerVoiceConfig is set.",
			})
		}
		if opts.Instructions != "" {
			warnings = append(warnings, Warning{
				Type:    "unsupported",
				Feature: "instructions with multiSpeakerVoiceConfig",
				Details: "instructions is ignored when multiSpeakerVoiceConfig is set.",
			})
		}
	}

	// Default voice: Kore, default sample rate: 24000 Hz.
	voiceName := "Kore"
	if opts.Voice != "" && speechOpts.MultiSpeakerVoiceConfig == nil {
		voiceName = opts.Voice
	}

	instructions := opts.Instructions

	if opts.Speed != 0 && opts.Speed != 1.0 {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "speed",
			Details: "speed is not supported by the Google Generative AI API.",
		})
	}
	if opts.Language != "" {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "language",
			Details: "language is not supported by the Google Generative AI API.",
		})
	}

	outputFormat := opts.OutputFormat
	if outputFormat == "" {
		outputFormat = "wav"
	}

	switch outputFormat {
	case "wav", "pcm":
		// ok
	default:
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "outputFormat",
			Details: "unknown outputFormat; falling back to wav.",
		})
		outputFormat = "wav"
	}

	// Build generationConfig.
	genConfig := internal.APIGenerationConfig{
		ResponseModalities: []string{"AUDIO"},
	}
	if speechOpts.MultiSpeakerVoiceConfig != nil {
		cfg := internal.APISpeechConfig{
			MultiSpeakerVoiceConfig: &internal.APIMultiSpeakerVoiceConfig{
				SpeakerVoiceConfigs: make([]internal.APISpeakerVoiceConfig, len(speechOpts.MultiSpeakerVoiceConfig.SpeakerVoiceConfigs)),
			},
		}
		for i, svc := range speechOpts.MultiSpeakerVoiceConfig.SpeakerVoiceConfigs {
			cfg.MultiSpeakerVoiceConfig.SpeakerVoiceConfigs[i] = internal.APISpeakerVoiceConfig{
				Speaker: svc.Speaker,
				VoiceConfig: internal.APIPrebuiltVoiceConfig{
					VoiceName: svc.VoiceConfig.VoiceName,
				},
			}
		}
		genConfig.SpeechConfig = &cfg
	} else {
		genConfig.SpeechConfig = &internal.APISpeechConfig{
			VoiceConfig: &internal.APISingleVoiceConfig{
				PrebuiltVoiceConfig: internal.APIPrebuiltVoiceConfig{
					VoiceName: voiceName,
				},
			},
		}
	}

	text := opts.Text
	if instructions != "" {
		text = instructions + ": " + text
	}

	body := internal.APIGenerateContentRequest{
		Contents: []internal.APIContent{
			{
				Role: "user",
				Parts: []internal.APIPart{
					{Text: text},
				},
			},
		},
		GenerationConfig: &genConfig,
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	resp, err := m.provider.executeJSON(ctx, "/"+getModelPath(m.modelID)+":generateContent", bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}

	var genResp internal.APIGenerateContentResponse
	if err := json.Unmarshal(resp.Body, &genResp); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse generateContent response", Data: string(resp.Body)}
	}
	if len(genResp.Candidates) == 0 {
		return nil, &APICallError{
			Message:   "generateContent response missing candidates",
			Type:      "GOOGLE_SPEECH_GENERATION_ERROR",
			Retryable: false,
			Status:    resp.Status,
			Headers:   resp.Headers,
			RequestID: resp.RequestID,
			Body:      resp.Body,
		}
	}

	candidate := genResp.Candidates[0]
	if len(candidate.Content.Parts) == 0 {
		return nil, &APICallError{
			Message:   "generateContent candidate missing parts",
			Type:      "GOOGLE_SPEECH_GENERATION_ERROR",
			Retryable: false,
			Status:    resp.Status,
			Headers:   resp.Headers,
			RequestID: resp.RequestID,
			Body:      resp.Body,
		}
	}

	var audioData string
	var mimeType string
	for _, part := range candidate.Content.Parts {
		if part.InlineData != nil {
			audioData = part.InlineData.Data
			mimeType = part.InlineData.MimeType
			break
		}
	}

	if audioData == "" {
		return nil, &APICallError{
			Message:   "generateContent response missing audio data",
			Type:      "GOOGLE_SPEECH_GENERATION_ERROR",
			Retryable: false,
			Status:    resp.Status,
			Headers:   resp.Headers,
			RequestID: resp.RequestID,
			Body:      resp.Body,
		}
	}

	// Decode base64 audio.
	rawAudio, err := base64.StdEncoding.DecodeString(audioData)
	if err != nil {
		return nil, err
	}

	var audio []byte
	var providerMetadata ProviderMetadata
	sampleRate := 24000

	switch outputFormat {
	case "wav":
		audio = wrapWAVHeader(rawAudio, sampleRate)
		providerMetadata = ProviderMetadata{
			"google": map[string]any{
				"mimeType":   mimeType,
				"sampleRate": sampleRate,
			},
		}
	case "pcm":
		audio = rawAudio
		if mimeType != "" {
			if rate := parseMimeRate(mimeType); rate > 0 {
				sampleRate = rate
			}
		}
		providerMetadata = ProviderMetadata{
			"google": map[string]any{
				"mimeType":   mimeType,
				"sampleRate": sampleRate,
			},
		}
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "outputFormat:pcm",
			Details: "pcm output is returned raw; most clients require wav encoding.",
		})
	}

	return &SpeechGenerateResult{
		Audio:            audio,
		Warnings:         warnings,
		ProviderMetadata: providerMetadata,
		Request:          RequestMetadata{Body: append([]byte(nil), bodyBytes...)},
		Response:         responseMetadata(resp.Headers, append([]byte(nil), resp.Body...), "", m.modelID),
	}, nil
}

// speechModelOptionsFromProviderOptions parses the typed SpeechModelOptions view
// from the merged google provider-options namespace. The list of recognized keys
// is returned so callers can strip them before forwarding passthrough keys.
func speechModelOptionsFromProviderOptions(merged map[string]any) (SpeechModelOptions, []string) {
	out := SpeechModelOptions{}
	var recognized []string
	if merged == nil {
		return out, recognized
	}
	if v, ok := merged["multiSpeakerVoiceConfig"]; ok {
		if m, ok := v.(map[string]any); ok {
			if svcsRaw, ok := m["speakerVoiceConfigs"].([]any); ok {
				svcs := make([]SpeakerVoiceConfig, 0, len(svcsRaw))
				for _, elem := range svcsRaw {
					if elemMap, ok := elem.(map[string]any); ok {
						svc := SpeakerVoiceConfig{}
						if s, ok := elemMap["speaker"].(string); ok {
							svc.Speaker = s
						}
						if vcMap, ok := elemMap["voiceConfig"].(map[string]any); ok {
							if vn, ok := vcMap["voiceName"].(string); ok {
								svc.VoiceConfig.VoiceName = vn
							}
						}
						svcs = append(svcs, svc)
					}
				}
				out.MultiSpeakerVoiceConfig = &MultiSpeakerVoiceConfig{SpeakerVoiceConfigs: svcs}
			}
		}
		recognized = append(recognized, "multiSpeakerVoiceConfig")
	}
	return out, recognized
}

// wrapWAVHeader prepends a 44-byte WAV header to raw PCM audio data.
// The header encodes pcm/s16le/mono at the given sampleRate.
func wrapWAVHeader(pcmData []byte, sampleRate int) []byte {
	const (
		wavHeaderSize  = 44
		numChannels    = 1
		bitsPerSample  = 16
		blockAlign     = numChannels * bitsPerSample / 8
	)
	dataSize := len(pcmData)
	bytesPerSec := sampleRate * numChannels * bitsPerSample / 8

	header := make([]byte, wavHeaderSize)
	// RIFF chunk descriptor
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(wavHeaderSize+dataSize-8))
	copy(header[8:12], "WAVE")
	// fmt sub-chunk
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)             // subchunk1Size
	binary.LittleEndian.PutUint16(header[20:22], 1)              // audioFormat (PCM)
	binary.LittleEndian.PutUint16(header[22:24], uint16(numChannels))
	binary.LittleEndian.PutUint32(header[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(header[28:32], uint32(bytesPerSec))
	binary.LittleEndian.PutUint16(header[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(header[34:36], uint16(bitsPerSample))
	// data sub-chunk
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(dataSize))

	out := make([]byte, wavHeaderSize+dataSize)
	copy(out, header)
	copy(out[wavHeaderSize:], pcmData)
	return out
}

// parseMimeRate extracts the sample rate from a MIME type string such as
// "audio/L16;rate=24000". Returns 0 if no rate is found.
func parseMimeRate(mime string) int {
	if mime == "" {
		return 0
	}
	m := mimeRateRegexp.FindStringSubmatch(mime)
	if m == nil {
		return 0
	}
	var rate int
	for _, c := range m[1] {
		rate = rate*10 + int(c-'0')
	}
	return rate
}

var mimeRateRegexp = regexp.MustCompile(`;rate=(\d+)`)