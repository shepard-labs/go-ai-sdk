package google

import "strings"

// ModelCapabilitiesForID returns a populated ModelCapabilities for the given
// model ID. Matching is by case-insensitive prefix (per upstream
// google-supported-file-url behavior). Unknown model IDs return the zero value
// (all false), which is still a usable value for callers that don't rely on
// the flags.
func ModelCapabilitiesForID(modelID string) ModelCapabilities {
	id := strings.ToLower(strings.TrimSpace(modelID))
	if id == "" {
		return ModelCapabilities{}
	}

	caps := ModelCapabilities{
		SupportsTools:             true,
		SupportsSystemInstruction: true,
		SupportsStructuredOutput:  true,
		MaxOutputTokens:           8192,
	}

	switch {
	case strings.HasPrefix(id, "gemini-3."):
		// Gemini 3.x (3.0, 3.1, 3.5, etc.) — level thinking + function-call
		// streaming + grounded search tools.
		caps.SupportsThinking = true
		caps.SupportsFunctionCallStreaming = true
		caps.SupportsGoogleSearch = true
		caps.SupportsUrlContext = true
		caps.SupportsCodeExecution = true
		caps.SupportsFileSearch = true
		caps.SupportsGrounding = true
		caps.SupportsImageOutput = isImageModelID(id)
		caps.SupportsAudioOutput = isAudioOutputID(id)
		caps.MaxOutputTokens = 65536
	case strings.HasPrefix(id, "gemini-3"):
		// Gemini 3.0 family.
		caps.SupportsThinking = true
		caps.SupportsFunctionCallStreaming = true
		caps.SupportsGoogleSearch = true
		caps.SupportsUrlContext = true
		caps.SupportsCodeExecution = true
		caps.SupportsFileSearch = true
		caps.SupportsGrounding = true
		caps.SupportsImageOutput = isImageModelID(id)
		caps.SupportsAudioOutput = isAudioOutputID(id)
		caps.MaxOutputTokens = 65536
	case strings.HasPrefix(id, "gemini-2.5"):
		// Gemini 2.5 family — budget thinking + cached content.
		caps.SupportsThinking = true
		caps.SupportsGoogleSearch = true
		caps.SupportsUrlContext = true
		caps.SupportsCodeExecution = true
		caps.SupportsFileSearch = true
		caps.SupportsGrounding = true
		caps.SupportsCachedContent = true
		caps.SupportsImageOutput = isImageModelID(id)
		caps.SupportsAudioOutput = isAudioOutputID(id)
		caps.MaxOutputTokens = 65536
	case strings.HasPrefix(id, "gemini-2.0"):
		caps.SupportsThinking = true
		caps.SupportsGoogleSearch = true
		caps.SupportsUrlContext = true
		caps.SupportsCodeExecution = true
		caps.SupportsGrounding = true
		caps.MaxOutputTokens = 8192
	case strings.HasPrefix(id, "gemini-"):
		// Generic Gemini 1.x or other.
		caps.SupportsGoogleSearch = true
		caps.MaxOutputTokens = 8192
	case strings.HasPrefix(id, "imagen-"):
		caps.SupportsImageOutput = true
		caps.MaxOutputTokens = 1024
	case strings.HasPrefix(id, "veo-"):
		caps.MaxOutputTokens = 1
	case strings.HasPrefix(id, "embedding"):
		caps.MaxOutputTokens = 0
	case strings.HasPrefix(id, "gemma-"):
		// Gemma does NOT support systemInstruction; the system text is
		// prepended to the first user message.
		caps.SupportsSystemInstruction = false
		caps.MaxOutputTokens = 8192
	case strings.HasPrefix(id, "deep-research-"):
		caps.SupportsGoogleSearch = true
		caps.SupportsGrounding = true
		caps.MaxOutputTokens = 65536
	}

	// Multimodal input flags apply across most families except image/video/
	// embedding-only families (which don't accept image/audio/video input).
	if !isNonInputFamily(id) {
		caps.SupportsImages = true
		caps.SupportsAudio = isAudioInputID(id)
		caps.SupportsVideo = true
	}

	return caps
}

// isNonInputFamily reports whether the model ID belongs to a non-input family
// (image generators, video generators, embedding models) where image/audio/
// video input is not supported.
func isNonInputFamily(id string) bool {
	return strings.HasPrefix(id, "imagen-") ||
		strings.HasPrefix(id, "veo-") ||
		strings.HasPrefix(id, "embedding") ||
		strings.HasPrefix(id, "gemini-embedding")
}

func isImageModelID(id string) bool {
	return strings.Contains(id, "image") || strings.HasPrefix(id, "nano-banana")
}

func isAudioOutputID(id string) bool {
	return strings.Contains(id, "tts") || strings.Contains(id, "native-audio")
}

func isAudioInputID(id string) bool {
	return !isNonInputFamily(id)
}

// isGemini3 reports whether the model is Gemini 3 or newer (drives the
// "include server-side tool invocations" mixed-tools behavior).
func isGemini3(modelID string) bool {
	id := strings.ToLower(modelID)
	return strings.HasPrefix(id, "gemini-3") || strings.HasPrefix(id, "gemini-3.")
}

// isGemini2Point5 reports whether the model is Gemini 2.5 (drives
// thinkingBudget vs thinkingLevel resolution).
func isGemini2Point5(modelID string) bool {
	id := strings.ToLower(modelID)
	return strings.HasPrefix(id, "gemini-2.5")
}

// isGemma reports whether the model ID belongs to the Gemma family.
func isGemma(modelID string) bool {
	id := strings.ToLower(modelID)
	return strings.HasPrefix(id, "gemma-")
}
