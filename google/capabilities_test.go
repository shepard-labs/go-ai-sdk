package google

import "testing"

func TestModelCapabilitiesForID(t *testing.T) {
	cases := []struct {
		name string
		id   string
		// Subset of fields to assert.
		check func(t *testing.T, c ModelCapabilities)
	}{
		{
			name: "gemini-2.5-pro supports thinking + cached content",
			id:   "gemini-2.5-pro",
			check: func(t *testing.T, c ModelCapabilities) {
				if !c.SupportsThinking {
					t.Error("SupportsThinking = false, want true")
				}
				if !c.SupportsCachedContent {
					t.Error("SupportsCachedContent = false, want true")
				}
				if !c.SupportsGoogleSearch {
					t.Error("SupportsGoogleSearch = false, want true")
				}
				if c.MaxOutputTokens != 65536 {
					t.Errorf("MaxOutputTokens = %d, want 65536", c.MaxOutputTokens)
				}
			},
		},
		{
			name: "gemini-3-pro supports level thinking + function-call streaming",
			id:   "gemini-3-pro-preview",
			check: func(t *testing.T, c ModelCapabilities) {
				if !c.SupportsThinking {
					t.Error("SupportsThinking = false, want true")
				}
				if !c.SupportsFunctionCallStreaming {
					t.Error("SupportsFunctionCallStreaming = false, want true")
				}
				if !c.SupportsFileSearch {
					t.Error("SupportsFileSearch = false, want true")
				}
				if c.MaxOutputTokens != 65536 {
					t.Errorf("MaxOutputTokens = %d, want 65536", c.MaxOutputTokens)
				}
			},
		},
		{
			name: "gemini-3.1-pro supports function-call streaming",
			id:   "gemini-3.1-pro-preview",
			check: func(t *testing.T, c ModelCapabilities) {
				if !c.SupportsFunctionCallStreaming {
					t.Error("SupportsFunctionCallStreaming = false, want true")
				}
			},
		},
		{
			name: "gemini-2.0-flash supports thinking but no file search",
			id:   "gemini-2.0-flash",
			check: func(t *testing.T, c ModelCapabilities) {
				if !c.SupportsThinking {
					t.Error("SupportsThinking = false, want true")
				}
				if c.SupportsFunctionCallStreaming {
					t.Error("SupportsFunctionCallStreaming = true, want false")
				}
				if c.SupportsFileSearch {
					t.Error("SupportsFileSearch = true, want false")
				}
			},
		},
		{
			name: "gemma does not support system instruction",
			id:   "gemma-3-4b-it",
			check: func(t *testing.T, c ModelCapabilities) {
				if c.SupportsSystemInstruction {
					t.Error("SupportsSystemInstruction = true, want false")
				}
			},
		},
		{
			name: "imagen is image-output",
			id:   "imagen-4.0-generate-001",
			check: func(t *testing.T, c ModelCapabilities) {
				if !c.SupportsImageOutput {
					t.Error("SupportsImageOutput = false, want true")
				}
			},
		},
		{
			name: "veo is not multimodal-input",
			id:   "veo-3.1-generate",
			check: func(t *testing.T, c ModelCapabilities) {
				if c.SupportsImages {
					t.Error("SupportsImages = true, want false")
				}
				if c.SupportsAudio {
					t.Error("SupportsAudio = true, want false")
				}
			},
		},
		{
			name: "embedding-001 has no image support",
			id:   "gemini-embedding-001",
			check: func(t *testing.T, c ModelCapabilities) {
				if c.SupportsImages {
					t.Error("SupportsImages = true, want false")
				}
			},
		},
		{
			name: "unknown model returns zero value (no flags)",
			id:   "totally-unknown-model",
			check: func(t *testing.T, c ModelCapabilities) {
				// Just confirm it doesn't panic and is usable.
				_ = c
			},
		},
		{
			name: "case-insensitive prefix",
			id:   "GEMINI-2.5-PRO",
			check: func(t *testing.T, c ModelCapabilities) {
				if !c.SupportsThinking {
					t.Error("SupportsThinking = false, want true (case-insensitive)")
				}
			},
		},
		{
			name: "empty id returns zero value",
			id:   "",
			check: func(t *testing.T, c ModelCapabilities) {
				if c.SupportsThinking || c.SupportsImages {
					t.Error("empty id should not match any prefix")
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			caps := ModelCapabilitiesForID(tc.id)
			tc.check(t, caps)
		})
	}
}

func TestIsGemini3(t *testing.T) {
	cases := []struct {
		id   string
		want bool
	}{
		{"gemini-3-pro-preview", true},
		{"gemini-3.1-pro-preview", true},
		{"gemini-3.5-flash", true},
		{"GEMINI-3-PRO", true},
		{"gemini-2.5-pro", false},
		{"gemini-2.0-flash", false},
		{"", false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			if got := isGemini3(tc.id); got != tc.want {
				t.Errorf("isGemini3(%q) = %v, want %v", tc.id, got, tc.want)
			}
		})
	}
}

func TestIsGemma(t *testing.T) {
	cases := []struct {
		id   string
		want bool
	}{
		{"gemma-3-4b-it", true},
		{"gemma-3n-e4b-it", true},
		{"Gemma-3-1B-It", true},
		{"gemini-2.5-pro", false},
		{"", false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			if got := isGemma(tc.id); got != tc.want {
				t.Errorf("isGemma(%q) = %v, want %v", tc.id, got, tc.want)
			}
		})
	}
}
