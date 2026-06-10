package google

import "strings"

// GoogleOptions is the recognized typed view of merged google ProviderOptions.
// The outer key in the ProviderOptions map is "google" (see ProviderName).
// Unrecognized keys in ProviderOptions["google"] are forwarded as-is in the
// request body.
type GoogleOptions struct {
	ResponseModalities          []string
	ThinkingConfig              *ThinkingConfig
	CachedContent               string // "cachedContents/{id}"
	StructuredOutputs           *bool
	SafetySettings              []SafetySetting
	AudioTimestamp              *bool
	Labels                      map[string]string
	MediaResolution             string
	ImageConfig                 *ImageConfig
	RetrievalConfig             *RetrievalConfig
	StreamFunctionCallArguments *bool
	ServiceTier                 string // "standard" | "flex" | "priority"
	SharedRequestType           string // Vertex only
	RequestType                 string // Vertex only
	// Threshold is a top-level default safety threshold applied to every
	// SafetySetting entry that has an empty Threshold field.
	Threshold string
}

// ThinkingConfig controls the model's thinking/reasoning behavior.
type ThinkingConfig struct {
	IncludeThoughts *bool
	ThinkingBudget  *int   // Gemini 2.5
	ThinkingLevel   string // Gemini 3: "minimal" | "low" | "medium" | "high"
}

// SafetySetting configures a harm category threshold.
type SafetySetting struct {
	Category  string // "HARM_CATEGORY_HATE_SPEECH" | ... | "HARM_CATEGORY_UNSPECIFIED"
	Threshold string // "BLOCK_NONE" | "OFF" | "BLOCK_LOW_AND_ABOVE" | ...
}

// ImageConfig configures image generation parameters.
type ImageConfig struct {
	AspectRatio string // "1:1" | "2:3" | "3:2" | ...
	ImageSize   string // "1K" | "2K" | "4K" | "512"
}

// LatLng is a geographic coordinate pair.
type LatLng struct {
	Latitude  float64
	Longitude float64
}

// RetrievalConfig configures the retrieval tool (e.g. Maps latLng).
type RetrievalConfig struct {
	LatLng *LatLng
}

// Safety threshold constants (same enum as SafetySetting.Threshold).
const (
	ThresholdHarmBlockUnspecified = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
	ThresholdBlockLowAndAbove     = "BLOCK_LOW_AND_ABOVE"
	ThresholdBlockMediumAndAbove  = "BLOCK_MEDIUM_AND_ABOVE"
	ThresholdBlockOnlyHigh        = "BLOCK_ONLY_HIGH"
	ThresholdBlockNone            = "BLOCK_NONE"
	ThresholdOff                  = "OFF"
)

// ImageModelOptions is the recognized typed view of google ImageModel ProviderOptions.
type ImageModelOptions struct {
	PersonGeneration string
	AspectRatio      string
	GoogleSearch     *GoogleSearchArgs
}

// VideoModelOptions is the recognized typed view of google VideoModel ProviderOptions.
type VideoModelOptions struct {
	PollIntervalMs   *int
	PollTimeoutMs    *int
	PersonGeneration string
	NegativePrompt   string
	ReferenceImages  []ReferenceImage
}

// SpeechModelOptions is the recognized typed view of google SpeechModel ProviderOptions.
type SpeechModelOptions struct {
	MultiSpeakerVoiceConfig *MultiSpeakerVoiceConfig
}

// MultiSpeakerVoiceConfig configures a multi-speaker TTS voice.
type MultiSpeakerVoiceConfig struct {
	SpeakerVoiceConfigs []SpeakerVoiceConfig
}

// SpeakerVoiceConfig binds a speaker label to a voice.
type SpeakerVoiceConfig struct {
	Speaker     string
	VoiceConfig PrebuiltVoiceConfig
}

// PrebuiltVoiceConfig names a prebuilt TTS voice.
type PrebuiltVoiceConfig struct {
	VoiceName string
}

// EmbeddingModelOptions is the recognized typed view of google EmbeddingModel
// ProviderOptions.
type EmbeddingModelOptions struct {
	OutputDimensionality *int
	TaskType             string // "SEMANTIC_SIMILARITY" | "CLASSIFICATION" | ...
	Content              [][]ContentPart
}

// isVertexProvider reports whether the base URL targets Vertex AI directly or
// whether the caller explicitly enabled Vertex header behavior.
//
// isVertexProvider is fixed at construction time.
func isVertexProvider(baseURL string, useVertexAIHeaders bool) bool {
	return strings.Contains(baseURL, "aiplatform.googleapis.com") || useVertexAIHeaders
}

// isVertexLike is a looser predicate than isVertexProvider: also returns true
// when the merged ProviderOptions contain a non-empty "googleVertex" or "vertex"
// key (used by the AI Gateway).
func isVertexLike(baseURL string, useVertexAIHeaders bool, opts ProviderOptions) bool {
	if isVertexProvider(baseURL, useVertexAIHeaders) {
		return true
	}
	if opts == nil {
		return false
	}
	if len(opts["googleVertex"]) > 0 || len(opts["vertex"]) > 0 {
		return true
	}
	return false
}

// googleOptionsFromProviderOptions extracts the recognized GoogleOptions fields
// from the raw ProviderOptions map (under the "google" key, with cross-namespace
// fallback to "googleVertex" / "vertex" when vertexLike is true).
// It returns the typed view; the caller is responsible for forwarding unrecognized
// keys as body passthrough.
func googleOptionsFromProviderOptions(opts ProviderOptions, vertexLike bool) GoogleOptions {
	if opts == nil {
		return GoogleOptions{}
	}
	raw := mergeGoogleNamespaces(opts, vertexLike)
	if raw == nil {
		return GoogleOptions{}
	}
	var out GoogleOptions

	if v, ok := raw["responseModalities"]; ok {
		if arr, ok := toStringSlice(v); ok {
			out.ResponseModalities = arr
		}
	}
	if v, ok := raw["cachedContent"]; ok {
		if s, ok := v.(string); ok {
			out.CachedContent = s
		}
	}
	if v, ok := raw["structuredOutputs"]; ok {
		if b, ok := v.(bool); ok {
			out.StructuredOutputs = &b
		}
	}
	if v, ok := raw["audioTimestamp"]; ok {
		if b, ok := v.(bool); ok {
			out.AudioTimestamp = &b
		}
	}
	if v, ok := raw["mediaResolution"]; ok {
		if s, ok := v.(string); ok {
			out.MediaResolution = s
		}
	}
	if v, ok := raw["serviceTier"]; ok {
		if s, ok := v.(string); ok {
			out.ServiceTier = s
		}
	}
	if v, ok := raw["sharedRequestType"]; ok {
		if s, ok := v.(string); ok {
			out.SharedRequestType = s
		}
	}
	if v, ok := raw["requestType"]; ok {
		if s, ok := v.(string); ok {
			out.RequestType = s
		}
	}
	if v, ok := raw["threshold"]; ok {
		if s, ok := v.(string); ok {
			out.Threshold = s
		}
	}
	if v, ok := raw["streamFunctionCallArguments"]; ok {
		if b, ok := v.(bool); ok {
			out.StreamFunctionCallArguments = &b
		}
	}
	if v, ok := raw["labels"]; ok {
		if m, ok := v.(map[string]any); ok {
			labels := make(map[string]string, len(m))
			for k, val := range m {
				if s, ok := val.(string); ok {
					labels[k] = s
				}
			}
			out.Labels = labels
		}
	}
	return out
}

// mergeGoogleNamespaces merges "google" options with optional cross-namespace
// fallback ("googleVertex" / "vertex") when vertexLike is true. The "google"
// key wins on conflicts.
func mergeGoogleNamespaces(opts ProviderOptions, vertexLike bool) map[string]any {
	var base map[string]any
	if vertexLike {
		if v, ok := opts["vertex"]; ok {
			base = shallowMerge(base, v)
		}
		if v, ok := opts["googleVertex"]; ok {
			base = shallowMerge(base, v)
		}
	}
	if v, ok := opts["google"]; ok {
		base = shallowMerge(base, v)
	}
	return base
}

func shallowMerge(dst, src map[string]any) map[string]any {
	if dst == nil {
		dst = make(map[string]any, len(src))
	}
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func toStringSlice(v any) ([]string, bool) {
	switch val := v.(type) {
	case []string:
		return val, true
	case []any:
		out := make([]string, 0, len(val))
		for _, elem := range val {
			s, ok := elem.(string)
			if !ok {
				return nil, false
			}
			out = append(out, s)
		}
		return out, true
	}
	return nil, false
}

// videoModelOptionsFromProviderOptions parses the typed VideoModelOptions view
// from the merged google provider-options namespace. The list of recognized keys
// is returned so callers can strip them before forwarding passthrough keys.
func videoModelOptionsFromProviderOptions(merged map[string]any) (VideoModelOptions, []string) {
	out := VideoModelOptions{}
	var recognized []string
	if v, ok := merged["pollIntervalMs"]; ok {
		if i, ok := v.(int); ok {
			out.PollIntervalMs = &i
			recognized = append(recognized, "pollIntervalMs")
		}
	}
	if v, ok := merged["pollTimeoutMs"]; ok {
		if i, ok := v.(int); ok {
			out.PollTimeoutMs = &i
			recognized = append(recognized, "pollTimeoutMs")
		}
	}
	if v, ok := merged["personGeneration"]; ok {
		if s, ok := v.(string); ok {
			out.PersonGeneration = s
			recognized = append(recognized, "personGeneration")
		}
	}
	if v, ok := merged["negativePrompt"]; ok {
		if s, ok := v.(string); ok {
			out.NegativePrompt = s
			recognized = append(recognized, "negativePrompt")
		}
	}
	if v, ok := merged["referenceImages"]; ok {
		if arr, ok := v.([]any); ok {
			images := make([]ReferenceImage, 0, len(arr))
			for _, elem := range arr {
				if m, ok := elem.(map[string]any); ok {
					ri := ReferenceImage{}
					if s, ok := m["bytesBase64Encoded"].(string); ok {
						ri.BytesBase64Encoded = s
					}
					if s, ok := m["gcsUri"].(string); ok {
						ri.GcsUri = s
					}
					images = append(images, ri)
				}
			}
			out.ReferenceImages = images
			recognized = append(recognized, "referenceImages")
		}
	}
	return out, recognized
}
