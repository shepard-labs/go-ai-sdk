package openaicompatible

import "testing"

func TestFinishReasonMapping(t *testing.T) {
	for raw, want := range map[string]string{
		"stop":           "stop",
		"length":         "length",
		"content_filter": "content-filter",
		"function_call":  "tool-calls",
		"tool_calls":     "tool-calls",
		"":               "other",
		"unknown":        "other",
	} {
		if got := finishReasonFromOpenAI(raw); got.Unified != want || got.Raw != raw {
			t.Fatalf("finishReasonFromOpenAI(%q) = %#v, want unified %q", raw, got, want)
		}
	}
	if got := errorFinishReason(); got.Unified != "error" {
		t.Fatalf("error finish = %#v", got)
	}
}
