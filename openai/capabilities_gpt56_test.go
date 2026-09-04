package openai

import (
	"testing"
)

// TestModelCapabilitiesGpt56 verifies gpt-5.6 family capabilities: the
// base model and its sol/terra/luna snapshots are reasoning models with
// flex/priority processing and non-reasoning parameter support.
func TestModelCapabilitiesGpt56(t *testing.T) {
	for _, modelID := range []string{"gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} {
		caps := ModelCapabilitiesForID(modelID)
		if !caps.IsReasoningModel {
			t.Errorf("%s: IsReasoningModel should be true", modelID)
		}
		if caps.SystemMessageMode != "developer" {
			t.Errorf("%s: SystemMessageMode = %q, want developer", modelID, caps.SystemMessageMode)
		}
		if !caps.SupportsFlexProcessing {
			t.Errorf("%s: SupportsFlexProcessing should be true", modelID)
		}
		if !caps.SupportsPriorityProcessing {
			t.Errorf("%s: SupportsPriorityProcessing should be true", modelID)
		}
		if !caps.SupportsNonReasoningParameters {
			t.Errorf("%s: SupportsNonReasoningParameters should be true", modelID)
		}
		if got := MaxCompletionTokensForModel(modelID); got != 128000 {
			t.Errorf("%s: MaxCompletionTokensForModel = %d, want 128000", modelID, got)
		}
	}
}

// TestModelCapabilitiesGpt56CaseInsensitive verifies capability lookup is
// case-insensitive for the new family.
func TestModelCapabilitiesGpt56CaseInsensitive(t *testing.T) {
	if !IsReasoningModel("GPT-5.6-SOL") {
		t.Error("GPT-5.6-SOL should be a reasoning model")
	}
}
