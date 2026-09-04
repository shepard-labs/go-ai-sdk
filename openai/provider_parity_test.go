package openai

import (
	"os"
	"strings"
	"testing"
)

// TestProviderRejectsBlankBaseURL verifies CreateOpenAI rejects empty or
// whitespace-only BaseURL with a config error naming baseURL, while
// still defaulting when unset and trimming trailing slashes.
func TestProviderRejectsBlankBaseURL(t *testing.T) {
	oldEnv, hadEnv := os.LookupEnv("OPENAI_BASE_URL")
	os.Unsetenv("OPENAI_BASE_URL")
	defer func() {
		if hadEnv {
			os.Setenv("OPENAI_BASE_URL", oldEnv)
		}
	}()

	cases := []struct {
		name        string
		baseURL     string
		wantErr     bool
		wantBaseURL string
	}{
		{"whitespace-rejected", "   ", true, ""},
		{"tab-newline-rejected", "\t\n ", true, ""},
		{"slashes-only-defaults", "///", false, defaultBaseURL},
		{"unset-defaults", "", false, defaultBaseURL},
		{"trailing-slash-trimmed", "https://example.test/v1/", false, "https://example.test/v1"},
		{"clean-kept", "https://example.test/v1", false, "https://example.test/v1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := CreateOpenAI(ProviderSettings{APIKey: "k", BaseURL: c.baseURL})
			err := p.Err()
			if c.wantErr {
				if err == nil {
					t.Fatal("expected config error")
				}
				if !strings.Contains(err.Error(), "baseURL") {
					t.Errorf("error should name baseURL, got: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Err: %v", err)
			}
			if got := p.(*openaiProvider).baseURL; got != c.wantBaseURL {
				t.Errorf("baseURL=%q, want %q", got, c.wantBaseURL)
			}
		})
	}
}

// TestProviderRejectsWhitespaceBaseURLEnv verifies a whitespace-only
// OPENAI_BASE_URL env value is also rejected.
func TestProviderRejectsWhitespaceBaseURLEnv(t *testing.T) {
	oldEnv, hadEnv := os.LookupEnv("OPENAI_BASE_URL")
	os.Setenv("OPENAI_BASE_URL", "   ")
	defer func() {
		if hadEnv {
			os.Setenv("OPENAI_BASE_URL", oldEnv)
		} else {
			os.Unsetenv("OPENAI_BASE_URL")
		}
	}()
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	if err := p.Err(); err == nil || !strings.Contains(err.Error(), "baseURL") {
		t.Errorf("expected baseURL config error, got: %v", err)
	}
}
