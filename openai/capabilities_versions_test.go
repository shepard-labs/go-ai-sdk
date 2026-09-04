package openai

import "testing"

// TestModelCapabilitiesFutureGptVersions verifies anchored version parsing:
// gpt-5.7+ classifies like the enumerated gpt-5.x reasoning family.
func TestModelCapabilitiesFutureGptVersions(t *testing.T) {
	for _, modelID := range []string{"gpt-5.7", "gpt-5.7-mini", "gpt-5.10", "GPT-5.8"} {
		caps := ModelCapabilitiesForID(modelID)
		if !caps.IsReasoningModel {
			t.Errorf("%s: IsReasoningModel should be true", modelID)
		}
		if caps.SystemMessageMode != "developer" {
			t.Errorf("%s: SystemMessageMode = %q, want developer", modelID, caps.SystemMessageMode)
		}
		if !caps.SupportsNonReasoningParameters {
			t.Errorf("%s: SupportsNonReasoningParameters should be true", modelID)
		}
		if got := MaxCompletionTokensForModel(modelID); got != 128000 {
			t.Errorf("%s: MaxCompletionTokensForModel = %d, want 128000", modelID, got)
		}
	}
}

// TestModelCapabilitiesFutureOSeries verifies future o-series versions
// (o5+) classify as reasoning models.
func TestModelCapabilitiesFutureOSeries(t *testing.T) {
	for _, modelID := range []string{"o5", "o5-mini", "o6"} {
		caps := ModelCapabilitiesForID(modelID)
		if !caps.IsReasoningModel {
			t.Errorf("%s: IsReasoningModel should be true", modelID)
		}
		if caps.SystemMessageMode != "developer" {
			t.Errorf("%s: SystemMessageMode = %q, want developer", modelID, caps.SystemMessageMode)
		}
		if got := MaxCompletionTokensForModel(modelID); got != 128000 {
			t.Errorf("%s: MaxCompletionTokensForModel = %d, want 128000", modelID, got)
		}
	}
}

// TestModelCapabilitiesUnknownIDsStayConservative verifies unrecognized IDs
// (fine-tunes, proxies, future names) keep safe defaults and never panic.
func TestModelCapabilitiesUnknownIDsStayConservative(t *testing.T) {
	for _, modelID := range []string{"custom-fine-tune-xyz", "my-proxy-model", "", "gpt-", "gpt-x", "o", "openai-odd"} {
		caps := ModelCapabilitiesForID(modelID)
		if caps.IsReasoningModel {
			t.Errorf("%q: IsReasoningModel should be false", modelID)
		}
		if caps.SystemMessageMode != "system" {
			t.Errorf("%q: SystemMessageMode = %q, want system", modelID, caps.SystemMessageMode)
		}
		if caps.SupportsFlexProcessing || caps.SupportsPriorityProcessing || caps.SupportsNonReasoningParameters {
			t.Errorf("%q: processing flags should all be false: %+v", modelID, caps)
		}
		if got := MaxCompletionTokensForModel(modelID); got != 0 {
			t.Errorf("%q: MaxCompletionTokensForModel = %d, want 0", modelID, got)
		}
	}
}

// TestAnchoredVersionParsing verifies the gpt/o-series parsers on exact,
// suffixed, and malformed inputs.
func TestAnchoredVersionParsing(t *testing.T) {
	gptCases := []struct {
		in           string
		major, minor int
		ok           bool
	}{
		{"gpt-5", 5, 0, true},
		{"gpt-5.6", 5, 6, true},
		{"gpt-5.6-sol", 5, 6, true},
		{"gpt-5-nano", 5, 0, true},
		{"gpt-4.1", 4, 1, true},
		{"gpt-3.5-turbo", 3, 5, true},
		{"gpt-4o", 0, 0, false},
		{"gpt-5x-custom", 0, 0, false},
		{"gpt-5.1x", 0, 0, false},
		{"gpt-5.", 0, 0, false},
		{"gpt-", 0, 0, false},
		{"gpt-x", 0, 0, false},
		{"o1", 0, 0, false},
		{"", 0, 0, false},
	}
	for _, c := range gptCases {
		major, minor, ok := parseGptVersion(c.in)
		if major != c.major || minor != c.minor || ok != c.ok {
			t.Errorf("parseGptVersion(%q) = (%d,%d,%v), want (%d,%d,%v)",
				c.in, major, minor, ok, c.major, c.minor, c.ok)
		}
	}
	oCases := []struct {
		in string
		n  int
		ok bool
	}{
		{"o1", 1, true},
		{"o3-mini", 3, true},
		{"o4-mini", 4, true},
		{"o5", 5, true},
		{"o1x", 0, false},
		{"o", 0, false},
		{"ox", 0, false},
		{"openai", 0, false},
		{"gpt-5", 0, false},
		{"", 0, false},
	}
	for _, c := range oCases {
		n, ok := parseOSeriesVersion(c.in)
		if n != c.n || ok != c.ok {
			t.Errorf("parseOSeriesVersion(%q) = (%d,%v), want (%d,%v)", c.in, n, ok, c.n, c.ok)
		}
	}
}
