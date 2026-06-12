package openai

import "testing"

func TestMapLanguageToISOSamplePairs(t *testing.T) {
	// Spot-check a subset of 57 entries.
	cases := map[string]string{
		"english":        "en",
		"chinese":        "zh",
		"japanese":       "ja",
		"korean":         "ko",
		"spanish":        "es",
		"portuguese":     "pt",
		"haitian creole": "ht",
		"haitian":        "ht",
		"nynorsk":        "nn",
		"norwegian":      "no",
		"german":         "de",
		"french":         "fr",
		"russian":        "ru",
		"arabic":         "ar",
		"hindi":          "hi",
	}
	for in, want := range cases {
		if got := mapLanguageToISO(in); got != want {
			t.Errorf("mapLanguageToISO(%q) = %q, want %q", in, got, want)
		}
	}
}
