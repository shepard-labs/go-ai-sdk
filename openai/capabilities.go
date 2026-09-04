package openai

import (
	"strconv"
	"strings"
)

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
	gptMajor, gptMinor, isGpt := parseGptVersion(lower)
	oVersion, isOSeries := parseOSeriesVersion(lower)
	isChatLatest := isChatLatestVariant(lower)
	isNano := strings.Contains(lower, "-nano")

	// Reasoning families, including future anchored versions (gpt-5.7+,
	// o5+) that familyPrefix doesn't enumerate yet.
	if (isReasoningFamilyPrefix(family) || (isGpt && gptMajor >= 5) || isOSeries) && !isChatLatestVariant(lower) {
		caps.IsReasoningModel = true
		caps.SystemMessageMode = "developer"
	}

	// Flex processing: o-series >= 3, or gpt major >= 5 (including 5.x),
	// excluding chat-latest variants. There is no nano exclusion, matching
	// upstream getOpenAILanguageModelCapabilities.
	if (isOSeries && oVersion >= 3) || (isGpt && gptMajor >= 5 && !isChatLatest) {
		caps.SupportsFlexProcessing = true
	}

	// Priority processing: the gpt-4 family (including gpt-4o), gpt major
	// >= 5 excluding nano and chat-latest variants, or o-series >= 3.
	if strings.HasPrefix(lower, "gpt-4") ||
		(isGpt && gptMajor >= 5 && !isChatLatest && !isNano) ||
		(isOSeries && oVersion >= 3) {
		caps.SupportsPriorityProcessing = true
	}

	// Non-reasoning parameters when reasoning effort is "none".
	// Anchored on gpt-5.x with minor >= 1 so future versions (gpt-5.7+)
	// classify the same as the enumerated gpt-5.1..gpt-5.6 families.
	if isGpt && (gptMajor > 5 || (gptMajor == 5 && gptMinor >= 1)) {
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
		"gpt-5.3-chat-latest", "gpt-5.4", "gpt-5.5",
		"gpt-5.6", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra":
		return 128000
	}
	// Best-effort fallback for future anchored versions (gpt-5.7+, o5+).
	lower := strings.ToLower(modelID)
	if major, _, ok := parseGptVersion(lower); ok && major >= 5 {
		return 128000
	}
	if _, ok := parseOSeriesVersion(lower); ok {
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
		"gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.3", "gpt-5.2", "gpt-5.1", "gpt-5",
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
	case "o1", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6":
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

// parseGptVersion parses an anchored "gpt-<major>[.<minor>]" prefix out of
// a lowercased model ID, ignoring any trailing "-suffix" ("-nano",
// "-chat-latest", date snapshots) — mirroring upstream
// /^gpt-(\d+)(?:\.(\d+))?(?:-(.+))?$/. Anything else after the version
// (e.g. "gpt-5x-custom", "gpt-4o") reports ok=false. All indexing is
// bounds-checked so unrecognized IDs never panic.
func parseGptVersion(lower string) (major, minor int, ok bool) {
	rest, found := strings.CutPrefix(lower, "gpt-")
	if !found {
		return 0, 0, false
	}
	i := 0
	for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
		i++
	}
	if i == 0 {
		return 0, 0, false
	}
	major, _ = strconv.Atoi(rest[:i])
	rest = rest[i:]
	if rest == "" || rest[0] == '-' {
		return major, 0, true
	}
	if !strings.HasPrefix(rest, ".") {
		return 0, 0, false
	}
	j := 1
	for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
		j++
	}
	if j == 1 {
		return 0, 0, false
	}
	minor, _ = strconv.Atoi(rest[1:j])
	rest = rest[j:]
	if rest == "" || rest[0] == '-' {
		return major, minor, true
	}
	return 0, 0, false
}

// parseOSeriesVersion parses an anchored "o<digits>" prefix (o1, o3-mini,
// o4-mini, o5, ...) out of a lowercased model ID — mirroring upstream
// /^o(\d+)(?:-|$)/, so "o1x" reports ok=false. Non o-series IDs report
// ok=false; indexing is bounds-checked so unrecognized IDs never panic.
func parseOSeriesVersion(lower string) (n int, ok bool) {
	if !strings.HasPrefix(lower, "o") {
		return 0, false
	}
	rest := lower[1:]
	i := 0
	for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
		i++
	}
	if i == 0 {
		return 0, false
	}
	rest = rest[i:]
	if rest != "" && rest[0] != '-' {
		return 0, false
	}
	n, _ = strconv.Atoi(lower[1 : 1+i])
	return n, true
}

// Unknown model IDs (fine-tunes, proxies, future names outside the anchored
// parsers above) keep the conservative defaults from defaultCapabilities:
// non-reasoning, system message mode, no optional processing tiers. This is
// enforced structurally — ModelCapabilitiesForID only enables flags on
// positive family/version matches — so there is nothing to panic on.
