package openai

import "testing"

// TestGptImageFamilyAnchoredDetection verifies anchored gpt-image version
// membership, including future versions and malformed IDs.
func TestGptImageFamilyAnchoredDetection(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"gpt-image-1", true},
		{"gpt-image-1.5", true},
		{"gpt-image-1-mini", true},
		{"gpt-image-2", true},
		{"gpt-image-3", true},
		{"GPT-IMAGE-2", true},
		{"dall-e-3", false},
		{"dall-e-2", false},
		{"gpt-image-", false},
		{"gpt-image-x", false},
		{"xgpt-image-1", false},
		{"", false},
	}
	for _, c := range cases {
		if got := isGptImageFamily(c.in); got != c.want {
			t.Errorf("isGptImageFamily(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestHasDefaultResponseFormatCaseInsensitive verifies the default-format
// lookup normalizes case.
func TestHasDefaultResponseFormatCaseInsensitive(t *testing.T) {
	if !hasDefaultResponseFormat("GPT-IMAGE-1") {
		t.Error("GPT-IMAGE-1 should have a default response format")
	}
	if hasDefaultResponseFormat("dall-e-3") {
		t.Error("dall-e-3 should not have a default response format")
	}
}

// TestGptImageQualityFutureVersion verifies future gpt-image versions keep
// the gpt-image quality enum.
func TestGptImageQualityFutureVersion(t *testing.T) {
	if err := validateImageQuality("gpt-image-3", "high"); err != nil {
		t.Errorf("gpt-image-3 high: %v", err)
	}
	if err := validateImageQuality("gpt-image-3", "bogus"); err == nil {
		t.Error("gpt-image-3 bogus quality should error")
	}
	if err := validateImageQuality("dall-e-3", "standard"); err != nil {
		t.Errorf("dall-e-3 standard: %v", err)
	}
}
