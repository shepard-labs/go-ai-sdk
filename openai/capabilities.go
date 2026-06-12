package openai

import "strings"

// Capabilities describes per-model behavior used to strip or rename sampling
// parameters and to drive system-message mode selection.
type Capabilities struct {
	// IsReasoningModel indicates the model uses OpenAI's reasoning endpoint
	// (o1, o3, o4-mini, gpt-5 family).
	IsReasoningModel bool
	// SystemMessageMode is one of "system", "developer", or "remove".
	SystemMessageMode string
	// SupportsFlexProcessing indicates the model accepts service_tier: "flex".
	SupportsFlexProcessing bool
	// SupportsPriorityProcessing indicates the model accepts service_tier: "priority".
	SupportsPriorityProcessing bool
	// SupportsNonReasoningParameters allows temperature / topP / logProbs when
	// reasoningEffort is "none". gpt-5.1 family only.
	SupportsNonReasoningParameters bool
	// StripsTemperatureAlways forces temperature to be dropped even when
	// reasoning models in this family would normally accept it.
	StripsTemperatureAlways bool
	// SearchPreview indicates gpt-4o-search-preview family.
	SearchPreview bool
	// AudioPreview indicates gpt-4o-audio-preview family.
	AudioPreview bool
}

// ModelCapabilitiesForID returns the capability record for a model ID. The
// function uses an allowlist: unknown / fine-tuned / proxy model IDs default
// to non-reasoning, system-mode behavior so they don't break.
func ModelCapabilitiesForID(modelID string) Capabilities {
	if modelID == "" {
		return Capabilities{SystemMessageMode: "system"}
	}
	caps := defaultCapabilities()

	lower := strings.ToLower(modelID)
	family := familyPrefix(lower)

	// Reasoning families.
	if isReasoningFamilyPrefix(family) && !isChatLatestVariant(lower) {
		caps.IsReasoningModel = true
		caps.SystemMessageMode = "developer"
	}

	// Flex processing.
	if family == "o3" || family == "o4-mini" || (family == "gpt-5" && !isChatLatestVariant(lower)) {
		caps.SupportsFlexProcessing = true
	}

	// Priority processing.
	if family == "gpt-4" || family == "o3" || family == "o4-mini" ||
		(family == "gpt-5" && !isChatLatestVariant(lower) && !isGpt5NanoFamily(lower) && !isGpt54NanoFamily(lower)) {
		caps.SupportsPriorityProcessing = true
	}

	// Non-reasoning parameters when reasoning effort is "none".
	if family == "gpt-5.1" || family == "gpt-5.2" || family == "gpt-5.3" || family == "gpt-5.4" || family == "gpt-5.5" {
		caps.SupportsNonReasoningParameters = true
	}

	// Search / audio preview.
	if family == "gpt-4o-search-preview" || lower == "gpt-4o-mini-search-preview" {
		caps.StripsTemperatureAlways = true
		caps.SearchPreview = true
	}
	if family == "gpt-4o-audio-preview" {
		caps.AudioPreview = true
	}
	return caps
}

// IsReasoningModel reports whether the model ID belongs to a reasoning
// family (o1, o3, o4-mini, gpt-5 family).
func IsReasoningModel(modelID string) bool {
	return ModelCapabilitiesForID(modelID).IsReasoningModel
}

// SystemMessageMode returns "developer" for reasoning models and "system"
// otherwise.
func SystemMessageMode(modelID string) string {
	return ModelCapabilitiesForID(modelID).SystemMessageMode
}

// MaxCompletionTokensForModel returns a best-effort maximum completion
// tokens override for known reasoning models; 0 means "no specific override".
func MaxCompletionTokensForModel(modelID string) int {
	switch strings.ToLower(modelID) {
	case "o1", "o1-2024-12-17":
		return 100000
	case "o3", "o3-2025-04-16", "o3-mini":
		return 100000
	case "o4-mini", "o4-mini-2025-04-16":
		return 100000
	case "gpt-5", "gpt-5-2025-08-07", "gpt-5-mini", "gpt-5-mini-2025-08-07",
		"gpt-5-nano", "gpt-5-nano-2025-08-07", "gpt-5-chat-latest",
		"gpt-5.1", "gpt-5.1-2025-11-13", "gpt-5.1-chat-latest",
		"gpt-5.2", "gpt-5.2-2025-12-11", "gpt-5.2-pro", "gpt-5.2-pro-2025-12-11",
		"gpt-5.3-chat-latest", "gpt-5.4", "gpt-5.5":
		return 128000
	}
	return 0
}

func defaultCapabilities() Capabilities {
	return Capabilities{SystemMessageMode: "system"}
}

func familyPrefix(lowerModelID string) string {
	// Returns the longest family prefix used for capability lookups. We try
	// the multi-segment prefixes first (gpt-5.4, gpt-4o-audio-preview, etc.)
	// before falling back to the basic families.
	candidates := []string{
		"gpt-5.5", "gpt-5.4", "gpt-5.3", "gpt-5.2", "gpt-5.1", "gpt-5",
		"gpt-4.1", "gpt-4o-mini-search-preview", "gpt-4o-search-preview",
		"gpt-4o-audio-preview", "gpt-4o-mini-audio-preview", "gpt-4o", "gpt-4",
		"gpt-3.5", "o4-mini", "o3-mini", "o3", "o1",
	}
	for _, c := range candidates {
		if strings.HasPrefix(lowerModelID, c) {
			return c
		}
	}
	return ""
}

func isReasoningFamilyPrefix(family string) bool {
	switch family {
	case "o1", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5":
		return true
	}
	return false
}

func isChatLatestVariant(lower string) bool {
	return strings.HasSuffix(lower, "-chat-latest") ||
		lower == "gpt-5-chat-latest" ||
		lower == "gpt-5.1-chat-latest" ||
		lower == "gpt-5.2-chat-latest" ||
		lower == "gpt-5.3-chat-latest"
}

func isGpt5NanoFamily(lower string) bool {
	return strings.HasPrefix(lower, "gpt-5-nano")
}

func isGpt54NanoFamily(lower string) bool {
	return strings.HasPrefix(lower, "gpt-5.4-nano")
}
