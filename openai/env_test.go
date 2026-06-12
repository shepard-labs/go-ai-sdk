package openai

import (
	"net/http"
	"os"
	"strings"
	"testing"
)

// Verifies that Err() returns ErrMissingAPIKey when no API key is supplied
// and OPENAI_API_KEY is not set.
func TestProviderErrWhenNoAPIKey(t *testing.T) {
	// Make sure the env var is empty for the duration of this test.
	old, had := os.LookupEnv("OPENAI_API_KEY")
	os.Unsetenv("OPENAI_API_KEY")
	defer func() {
		if had {
			os.Setenv("OPENAI_API_KEY", old)
		}
	}()

	p := CreateOpenAI(ProviderSettings{})
	if p.Err() == nil {
		t.Fatal("expected Err() to be non-nil")
	}
	if p.Err() != ErrMissingAPIKey {
		t.Errorf("Err() = %v, want ErrMissingAPIKey", p.Err())
	}
}

// Verifies that an explicit API key disables the missing-key error.
func TestProviderErrClearsWithExplicitKey(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "explicit"})
	if p.Err() != nil {
		t.Errorf("Err() = %v, want nil", p.Err())
	}
}

// Verifies that the OPENAI_API_KEY env fallback works.
func TestProviderErrClearsWithEnvKey(t *testing.T) {
	old, had := os.LookupEnv("OPENAI_API_KEY")
	os.Setenv("OPENAI_API_KEY", "from-env")
	defer func() {
		if had {
			os.Setenv("OPENAI_API_KEY", old)
		} else {
			os.Unsetenv("OPENAI_API_KEY")
		}
	}()
	p := CreateOpenAI(ProviderSettings{})
	if p.Err() != nil {
		t.Errorf("Err() = %v, want nil", p.Err())
	}
}

// Verifies that baseURL trailing slashes are trimmed.
func TestProviderBaseURLTrailingSlashTrimmed(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k", BaseURL: "https://example.test/v1/"})
	if strings.HasSuffix(p.(*openaiProvider).baseURL, "/") {
		t.Errorf("baseURL should not have trailing slash: %q", p.(*openaiProvider).baseURL)
	}
}

// Verifies that the default baseURL is used when no env or setting is provided.
func TestProviderDefaultBaseURL(t *testing.T) {
	old, had := os.LookupEnv("OPENAI_BASE_URL")
	os.Unsetenv("OPENAI_BASE_URL")
	defer func() {
		if had {
			os.Setenv("OPENAI_BASE_URL", old)
		}
	}()
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	if got := p.(*openaiProvider).baseURL; got != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", got, defaultBaseURL)
	}
}

// Verifies that the OPENAI_BASE_URL env override works.
func TestProviderBaseURLEnvOverride(t *testing.T) {
	old, had := os.LookupEnv("OPENAI_BASE_URL")
	os.Setenv("OPENAI_BASE_URL", "https://env.example/v9/")
	defer func() {
		if had {
			os.Setenv("OPENAI_BASE_URL", old)
		} else {
			os.Unsetenv("OPENAI_BASE_URL")
		}
	}()
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	if got := p.(*openaiProvider).baseURL; got != "https://env.example/v9" {
		t.Errorf("baseURL = %q, want %q", got, "https://env.example/v9")
	}
}

// Verifies that the explicit BaseURL setting wins over the env.
func TestProviderBaseURLExplicitWinsOverEnv(t *testing.T) {
	old, had := os.LookupEnv("OPENAI_BASE_URL")
	os.Setenv("OPENAI_BASE_URL", "https://env.example/v1/")
	defer func() {
		if had {
			os.Setenv("OPENAI_BASE_URL", old)
		} else {
			os.Unsetenv("OPENAI_BASE_URL")
		}
	}()
	p := CreateOpenAI(ProviderSettings{APIKey: "k", BaseURL: "https://explicit.example/v2/"})
	if got := p.(*openaiProvider).baseURL; got != "https://explicit.example/v2" {
		t.Errorf("baseURL = %q, want %q", got, "https://explicit.example/v2")
	}
}

// Verifies that the User-Agent header is set to the versioned constant.
func TestProviderUserAgentHeader(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	got := p.compatHeaders().Get("User-Agent")
	if !strings.HasPrefix(got, "ai-sdk-go/openai/") {
		t.Errorf("User-Agent = %q, want prefix ai-sdk-go/openai/", got)
	}
	if got != userAgent {
		t.Errorf("User-Agent = %q, want %q", got, userAgent)
	}
}

// Verifies that Organization and Project propagate to the on-the-wire
// headers when set.
func TestProviderHeadersOrgAndProject(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k", Organization: "o1", Project: "p1"})
	hdrs := p.(*openaiProvider).compatHeaders()
	if hdrs.Get("OpenAI-Organization") != "o1" {
		t.Errorf("org: %q", hdrs.Get("OpenAI-Organization"))
	}
	if hdrs.Get("OpenAI-Project") != "p1" {
		t.Errorf("project: %q", hdrs.Get("OpenAI-Project"))
	}
}

// Verifies that spread Headers override the auth header.
func TestProviderHeadersSpreadOverridesAuth(t *testing.T) {
	custom := http.Header{"Authorization": []string{"Bearer override"}}
	p := CreateOpenAI(ProviderSettings{APIKey: "real-key", Headers: custom})
	hdrs := p.(*openaiProvider).compatHeaders()
	if got := hdrs.Get("Authorization"); got != "Bearer override" {
		t.Errorf("Authorization = %q, want Bearer override", got)
	}
}

// Verifies the default provider Name is "openai" when none is set.
func TestProviderNameDefaultsToOpenAI(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	if got := p.Name(); got != "openai" {
		t.Errorf("Name() = %q, want openai", got)
	}
}

// Verifies that an explicit Name is honored.
func TestProviderNameExplicit(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k", Name: "my-openai"})
	if got := p.Name(); got != "my-openai" {
		t.Errorf("Name() = %q, want my-openai", got)
	}
}

// Verifies that the callable form (Call) routes to ResponsesModel,
// matching upstream openai("modelID") behavior.
func TestProviderCallReturnsResponsesModel(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	m := p.Call("gpt-5")
	if m == nil {
		t.Fatal("Call returned nil")
	}
	if m.ModelID() != "gpt-5" {
		t.Errorf("ModelID = %q, want gpt-5", m.ModelID())
	}
}

// Verifies that Model and LanguageModel aliases are equivalent to Call.
func TestProviderModelLanguageModelAliases(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	if p.Model("x").ModelID() != "x" {
		t.Errorf("Model alias broken")
	}
	if p.LanguageModel("y").ModelID() != "y" {
		t.Errorf("LanguageModel alias broken")
	}
}

// Verifies OpenAI() (the package-level constructor) returns a working
// provider that defers env var handling.
func TestOpenAIDefaultReturnsProvider(t *testing.T) {
	p := OpenAI()
	if p == nil {
		t.Fatal("OpenAI() returned nil")
	}
	// Err() may be ErrMissingAPIKey if env var is unset; that's the
	// expected behavior. Just verify it doesn't panic.
	_ = p.Err()
}

// Verifies that the Tools() factory bag returns the openai tools.
func TestProviderToolsFactoryBag(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	if p.Tools() == nil {
		t.Fatal("Tools() returned nil")
	}
}

// Verifies that Files() and Skills() return non-nil clients.
func TestProviderFilesAndSkills(t *testing.T) {
	p := CreateOpenAI(ProviderSettings{APIKey: "k"})
	if p.Files() == nil {
		t.Fatal("Files() returned nil")
	}
	if p.Skills() == nil {
		t.Fatal("Skills() returned nil")
	}
	if p.ExperimentalRealtime() == nil {
		t.Fatal("ExperimentalRealtime() returned nil")
	}
}
