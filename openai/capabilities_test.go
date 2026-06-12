package openai

import "testing"

// Verifies the spec examples for ModelCapabilitiesForID.
func TestModelCapabilitiesForIDSpecExamples(t *testing.T) {
	cases := []struct {
		modelID string
		check   func(Capabilities) bool
		desc    string
	}{
		{"gpt-5", func(c Capabilities) bool { return c.IsReasoningModel && c.SystemMessageMode == "developer" }, "gpt-5 → IsReasoningModel: true, SystemMessageMode: developer"},
		{"gpt-5.1", func(c Capabilities) bool { return c.IsReasoningModel && c.SupportsNonReasoningParameters }, "gpt-5.1 → IsReasoningModel: true, SupportsNonReasoningParameters: true"},
		{"gpt-4o", func(c Capabilities) bool { return !c.IsReasoningModel && c.SystemMessageMode == "system" }, "gpt-4o → IsReasoningModel: false, SystemMessageMode: system"},
		{"custom-fine-tune", func(c Capabilities) bool { return !c.IsReasoningModel && c.SystemMessageMode == "system" }, "custom-fine-tune → IsReasoningModel: false (allowlist default)"},
		{"o1", func(c Capabilities) bool { return c.IsReasoningModel && c.SystemMessageMode == "developer" }, "o1 → reasoning"},
		{"o3", func(c Capabilities) bool { return c.IsReasoningModel && c.SupportsFlexProcessing && c.SupportsPriorityProcessing }, "o3 → reasoning, flex, priority"},
		{"o4-mini", func(c Capabilities) bool { return c.IsReasoningModel && c.SupportsFlexProcessing && c.SupportsPriorityProcessing }, "o4-mini → reasoning, flex, priority"},
		{"gpt-5-nano", func(c Capabilities) bool { return c.IsReasoningModel && !c.SupportsPriorityProcessing }, "gpt-5-nano → reasoning, no priority"},
		{"gpt-5-chat-latest", func(c Capabilities) bool { return !c.IsReasoningModel }, "gpt-5-chat-latest → not reasoning (chat variant)"},
		{"gpt-5.1-chat-latest", func(c Capabilities) bool { return !c.IsReasoningModel }, "gpt-5.1-chat-latest → not reasoning"},
		{"gpt-4o-search-preview", func(c Capabilities) bool { return c.StripsTemperatureAlways && c.SearchPreview }, "gpt-4o-search-preview → StripsTemperatureAlways, SearchPreview"},
		{"gpt-4o-mini-search-preview", func(c Capabilities) bool { return c.StripsTemperatureAlways && c.SearchPreview }, "gpt-4o-mini-search-preview → StripsTemperatureAlways, SearchPreview"},
		{"gpt-4o-audio-preview", func(c Capabilities) bool { return c.AudioPreview }, "gpt-4o-audio-preview → AudioPreview"},
		{"gpt-5.2", func(c Capabilities) bool { return c.IsReasoningModel && c.SupportsNonReasoningParameters }, "gpt-5.2 → reasoning, non-reasoning params allowed"},
		{"gpt-5.4-nano", func(c Capabilities) bool { return c.IsReasoningModel && !c.SupportsPriorityProcessing }, "gpt-5.4-nano → reasoning, no priority"},
	}
	for _, c := range cases {
		got := ModelCapabilitiesForID(c.modelID)
		if !c.check(got) {
			t.Errorf("%s (model=%q): got %+v", c.desc, c.modelID, got)
		}
	}
}

func TestMaxCompletionTokensForModel(t *testing.T) {
	if got := MaxCompletionTokensForModel("o1"); got != 100000 {
		t.Errorf("o1: got %d", got)
	}
	if got := MaxCompletionTokensForModel("gpt-5"); got != 128000 {
		t.Errorf("gpt-5: got %d", got)
	}
	if got := MaxCompletionTokensForModel("unknown"); got != 0 {
		t.Errorf("unknown: got %d", got)
	}
}

func TestIsReasoningModelHelper(t *testing.T) {
	if !IsReasoningModel("o1") {
		t.Error("o1 should be reasoning")
	}
	if !IsReasoningModel("gpt-5") {
		t.Error("gpt-5 should be reasoning")
	}
	if IsReasoningModel("gpt-4o") {
		t.Error("gpt-4o should not be reasoning")
	}
}

func TestSystemMessageModeHelper(t *testing.T) {
	if got := SystemMessageMode("gpt-5"); got != "developer" {
		t.Errorf("gpt-5: got %q", got)
	}
	if got := SystemMessageMode("gpt-4o"); got != "system" {
		t.Errorf("gpt-4o: got %q", got)
	}
}
