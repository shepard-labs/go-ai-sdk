package google

import (
	"errors"
	"net/http"
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestCreateGoogle_Defaults verifies that CreateGoogle uses the default base
// URL, name, and User-Agent when no settings are provided (but an API key env
// var is present to avoid an error).
func TestCreateGoogle_Defaults(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	p := CreateGoogle(ProviderSettings{})
	if err := p.Err(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Name() != defaultProviderName {
		t.Errorf("Name() = %q, want %q", p.Name(), defaultProviderName)
	}
	gp := p.(*googleProvider)
	if gp.baseURL != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", gp.baseURL, defaultBaseURL)
	}
}

// TestCreateGoogle_EnvVarFallback verifies that the API key is read from the
// environment variable when not supplied in settings.
func TestCreateGoogle_EnvVarFallback(t *testing.T) {
	os.Unsetenv("GOOGLE_GENERATIVE_AI_API_KEY")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-api-key")

	p := CreateGoogle(ProviderSettings{})
	if err := p.Err(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	gp := p.(*googleProvider)
	got := gp.headers.Get("x-goog-api-key")
	if got != "env-api-key" {
		t.Errorf("x-goog-api-key header = %q, want %q", got, "env-api-key")
	}
}

// TestCreateGoogle_MissingAPIKey verifies that Err() returns ErrMissingAPIKey
// when neither ProviderSettings.APIKey nor the env var is set.
func TestCreateGoogle_MissingAPIKey(t *testing.T) {
	os.Unsetenv("GOOGLE_GENERATIVE_AI_API_KEY")

	p := CreateGoogle(ProviderSettings{})
	if !errors.Is(p.Err(), ErrMissingAPIKey) {
		t.Errorf("Err() = %v, want ErrMissingAPIKey", p.Err())
	}
}

// TestCreateGoogle_ExplicitAPIKey verifies that an explicitly supplied API key
// takes precedence over the env var.
func TestCreateGoogle_ExplicitAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-key")

	p := CreateGoogle(ProviderSettings{APIKey: "explicit-key"})
	if err := p.Err(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	gp := p.(*googleProvider)
	got := gp.headers.Get("x-goog-api-key")
	if got != "explicit-key" {
		t.Errorf("x-goog-api-key = %q, want %q", got, "explicit-key")
	}
}

// TestCreateGoogle_BaseURLOverride verifies that a custom BaseURL is used.
func TestCreateGoogle_BaseURLOverride(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	customURL := "https://my-proxy.example.com/v1"
	p := CreateGoogle(ProviderSettings{BaseURL: customURL})
	gp := p.(*googleProvider)
	if gp.baseURL != customURL {
		t.Errorf("baseURL = %q, want %q", gp.baseURL, customURL)
	}
}

// TestCreateGoogle_BaseURLTrailingSlashStripped verifies that trailing slashes
// are stripped from BaseURL.
func TestCreateGoogle_BaseURLTrailingSlashStripped(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	p := CreateGoogle(ProviderSettings{BaseURL: "https://example.com/v1/"})
	gp := p.(*googleProvider)
	if strings.HasSuffix(gp.baseURL, "/") {
		t.Errorf("baseURL has trailing slash: %q", gp.baseURL)
	}
}

// TestCreateGoogle_UserAgent verifies that the User-Agent header includes the
// package version.
func TestCreateGoogle_UserAgent(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	p := CreateGoogle(ProviderSettings{})
	gp := p.(*googleProvider)
	ua := gp.headers.Get("User-Agent")
	if !strings.Contains(ua, "ai-sdk-go/google/") {
		t.Errorf("User-Agent %q does not contain 'ai-sdk-go/google/'", ua)
	}
	if !strings.Contains(ua, Version) {
		t.Errorf("User-Agent %q does not contain version %q", ua, Version)
	}
}

// TestCreateGoogle_CustomHeaders verifies that custom headers from settings are
// preserved in provider headers.
func TestCreateGoogle_CustomHeaders(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	custom := http.Header{}
	custom.Set("X-Custom-Header", "custom-value")
	p := CreateGoogle(ProviderSettings{Headers: custom})
	gp := p.(*googleProvider)
	if gp.headers.Get("X-Custom-Header") != "custom-value" {
		t.Error("custom header not preserved")
	}
}

// TestCreateGoogle_CustomName verifies that ProviderSettings.Name overrides the
// default provider name.
func TestCreateGoogle_CustomName(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	p := CreateGoogle(ProviderSettings{Name: "my-google"})
	if p.Name() != "my-google" {
		t.Errorf("Name() = %q, want %q", p.Name(), "my-google")
	}
}

// TestCreateGoogle_SupportURLs verifies that SupportURLs returns a non-empty
// map with the expected regexp patterns.
func TestCreateGoogle_SupportURLs(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	p := CreateGoogle(ProviderSettings{})
	lm := p.LanguageModel("gemini-2.0-flash")
	urls := lm.SupportURLs()
	if len(urls) == 0 {
		t.Fatal("SupportURLs() returned empty map")
	}
	patterns, ok := urls["*"]
	if !ok {
		t.Fatal("SupportURLs() has no '*' key")
	}
	if len(patterns) == 0 {
		t.Fatal("SupportURLs()['*'] is empty")
	}
}

// TestCreateGoogle_SupportURLs_YouTube verifies that YouTube URLs match the
// returned patterns.
func TestCreateGoogle_SupportURLs_YouTube(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	p := CreateGoogle(ProviderSettings{})
	lm := p.LanguageModel("gemini-2.0-flash")
	urls := lm.SupportURLs()
	patterns := urls["*"]

	testURLs := []struct {
		url  string
		want bool
	}{
		{"https://www.youtube.com/watch?v=dQw4w9WgXcQ", true},
		{"https://youtube.com/watch?v=abc123", true},
		{"https://youtu.be/dQw4w9WgXcQ", true},
		{"https://example.com/video.mp4", false},
	}

	for _, tc := range testURLs {
		matched := matchesAny(patterns, tc.url)
		if matched != tc.want {
			t.Errorf("URL %q: matched=%v, want %v", tc.url, matched, tc.want)
		}
	}
}

func matchesAny(patterns []*regexp.Regexp, s string) bool {
	for _, p := range patterns {
		if p.MatchString(s) {
			return true
		}
	}
	return false
}

// TestCreateGoogle_ModelFactories verifies that each model factory returns a
// non-nil value implementing the correct interface.
func TestCreateGoogle_ModelFactories(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})

	if lm := p.LanguageModel("gemini-2.0-flash"); lm == nil {
		t.Error("LanguageModel returned nil")
	}
	if lm := p.Model("gemini-2.0-flash"); lm == nil {
		t.Error("Model returned nil")
	}
	if lm := p.Chat("gemini-2.0-flash"); lm == nil {
		t.Error("Chat returned nil")
	}
	if lm := p.ChatModel("gemini-2.0-flash"); lm == nil {
		t.Error("ChatModel returned nil")
	}
	if lm := p.GenerativeAI("gemini-2.0-flash"); lm == nil {
		t.Error("GenerativeAI returned nil")
	}
	if em := p.EmbeddingModel("gemini-embedding-001"); em == nil {
		t.Error("EmbeddingModel returned nil")
	}
	if em := p.Embedding("gemini-embedding-001"); em == nil {
		t.Error("Embedding returned nil")
	}
	if em := p.TextEmbeddingModel("gemini-embedding-001"); em == nil {
		t.Error("TextEmbeddingModel returned nil")
	}
	if em := p.TextEmbedding("gemini-embedding-001"); em == nil {
		t.Error("TextEmbedding returned nil")
	}
	if im := p.ImageModel("imagen-4.0-generate-001"); im == nil {
		t.Error("ImageModel returned nil")
	}
	if im := p.Image("imagen-4.0-generate-001"); im == nil {
		t.Error("Image returned nil")
	}
	if vm := p.VideoModel("veo-3.0-generate-001"); vm == nil {
		t.Error("VideoModel returned nil")
	}
	if vm := p.Video("veo-3.0-generate-001"); vm == nil {
		t.Error("Video returned nil")
	}
	if sm := p.SpeechModel("gemini-2.5-flash-preview-tts"); sm == nil {
		t.Error("SpeechModel returned nil")
	}
	if sm := p.Speech("gemini-2.5-flash-preview-tts"); sm == nil {
		t.Error("Speech returned nil")
	}
	if f := p.Files(); f == nil {
		t.Error("Files returned nil")
	}
	if tf := p.Tools(); tf.GoogleSearch == nil {
		t.Error("Tools().GoogleSearch is nil")
	}
}

// TestCreateGoogle_EmbeddingModel_MaxAndParallel verifies the default
// MaxEmbeddingsPerCall and SupportsParallelCalls, and that overrides via
// ProviderSettings take effect.
func TestCreateGoogle_EmbeddingModel_MaxAndParallel(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")

	p := CreateGoogle(ProviderSettings{})
	em := p.EmbeddingModel("gemini-embedding-001")
	if em.MaxEmbeddingsPerCall() != defaultMaxEmbeddingsPerCall {
		t.Errorf("MaxEmbeddingsPerCall() = %d, want %d", em.MaxEmbeddingsPerCall(), defaultMaxEmbeddingsPerCall)
	}
	if !em.SupportsParallelCalls() {
		t.Error("SupportsParallelCalls() = false, want true")
	}

	// Override max
	falseVal := false
	p2 := CreateGoogle(ProviderSettings{
		APIKey:                         "k",
		MaxEmbeddingsPerCall:           100,
		SupportsParallelEmbeddingCalls: &falseVal,
	})
	em2 := p2.EmbeddingModel("gemini-embedding-001")
	if em2.MaxEmbeddingsPerCall() != 100 {
		t.Errorf("MaxEmbeddingsPerCall() = %d, want 100", em2.MaxEmbeddingsPerCall())
	}
	if em2.SupportsParallelCalls() {
		t.Error("SupportsParallelCalls() = true, want false")
	}
}

// TestCreateGoogle_ImageModel_MaxImagesPerCall verifies that MaxImagesPerCall
// returns 10 for Gemini image models and 4 for Imagen models.
func TestCreateGoogle_ImageModel_MaxImagesPerCall(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})

	tests := []struct {
		modelID string
		want    int
	}{
		{"gemini-2.5-flash-image", 10},
		{"nano-banana-pro-preview", 10},
		{"imagen-4.0-generate-001", 4},
		{"imagen-4.0-ultra-generate-001", 4},
	}
	for _, tc := range tests {
		im := p.ImageModel(tc.modelID)
		if got := im.MaxImagesPerCall(); got != tc.want {
			t.Errorf("MaxImagesPerCall(%q) = %d, want %d", tc.modelID, got, tc.want)
		}
	}
}

// TestCreateGoogle_VideoModel_MaxVideosPerCall verifies the default.
func TestCreateGoogle_VideoModel_MaxVideosPerCall(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})
	vm := p.VideoModel("veo-3.0-generate-001")
	if vm.MaxVideosPerCall() != defaultMaxVideosPerCall {
		t.Errorf("MaxVideosPerCall() = %d, want %d", vm.MaxVideosPerCall(), defaultMaxVideosPerCall)
	}
}

// TestCreateGoogle_VertexDetection verifies isVertex is set correctly.
func TestCreateGoogle_VertexDetection(t *testing.T) {
	tests := []struct {
		baseURL            string
		useVertexAIHeaders bool
		wantIsVertex       bool
	}{
		{defaultBaseURL, false, false},
		{"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l", false, true},
		{"https://my-proxy.example.com", true, true},
		{"https://my-proxy.example.com", false, false},
	}
	for _, tc := range tests {
		p := CreateGoogle(ProviderSettings{
			APIKey:             "k",
			BaseURL:            tc.baseURL,
			UseVertexAIHeaders: tc.useVertexAIHeaders,
		})
		gp := p.(*googleProvider)
		if gp.isVertex != tc.wantIsVertex {
			t.Errorf("baseURL=%q useVertex=%v: isVertex=%v, want %v",
				tc.baseURL, tc.useVertexAIHeaders, gp.isVertex, tc.wantIsVertex)
		}
	}
}

// TestModelID verifies that ModelID returns the model ID string.
func TestModelID(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})

	lm := p.LanguageModel("gemini-2.0-flash")
	if lm.ModelID() != "gemini-2.0-flash" {
		t.Errorf("ModelID() = %q, want %q", lm.ModelID(), "gemini-2.0-flash")
	}
}

// TestProviderName verifies the Provider() string on each model type.
func TestProviderName(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})

	tests := []struct {
		model interface{ Provider() string }
		want  string
	}{
		{p.LanguageModel("m"), defaultProviderName + ".chat"},
		{p.EmbeddingModel("m"), defaultProviderName + ".embedding"},
		{p.ImageModel("m"), defaultProviderName + ".image"},
		{p.VideoModel("m"), defaultProviderName + ".video"},
		{p.SpeechModel("m"), defaultProviderName + ".speech"},
	}
	for _, tc := range tests {
		if got := tc.model.Provider(); got != tc.want {
			t.Errorf("Provider() = %q, want %q", got, tc.want)
		}
	}
}

// TestStubs_ReturnUnsupportedError verifies that stub methods (image /
// video / speech / files) return UnsupportedFunctionalityError (not nil,
// not a different error type). DoStream is implemented in M5.
func TestStubs_ReturnUnsupportedError(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})

	var uf UnsupportedFunctionalityError

	// Other model families are still stubs in M5.
	if _, err := p.VideoModel("veo-3.0-generate-001").(*googleVideoModel).DoGenerate(nil, VideoGenerateOptions{}); !errors.As(err, &uf) {
		t.Errorf("video DoGenerate: got %T (%v), want UnsupportedFunctionalityError", err, err)
	}
	if _, err := p.SpeechModel("gemini-2.5-flash-preview-tts").(*googleSpeechModel).DoGenerate(nil, SpeechGenerateOptions{}); !errors.As(err, &uf) {
		t.Errorf("speech DoGenerate: got %T (%v), want UnsupportedFunctionalityError", err, err)
	}
	if _, err := p.Files().(*googleFiles).Upload(nil, nil, FilesUploadOptions{}); !errors.As(err, &uf) {
		t.Errorf("files Upload: got %T (%v), want UnsupportedFunctionalityError", err, err)
	}
}

// TestRetryOptions_Defaults verifies the default retry configuration.
func TestRetryOptions_Defaults(t *testing.T) {
	opts := defaultRetryOptions(nil)
	if opts.MaxRetries != 2 {
		t.Errorf("MaxRetries = %d, want 2", opts.MaxRetries)
	}
	if opts.Jitter != true {
		t.Error("Jitter = false, want true")
	}
}

// TestSupportURLs_FilesPath verifies that a provider files URL matches.
func TestSupportURLs_FilesPath(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})
	lm := p.LanguageModel("gemini-2.0-flash")
	urls := lm.SupportURLs()

	filesURL := defaultBaseURL + "/files/some-file-id"
	patterns := urls["*"]
	if !matchesAny(patterns, filesURL) {
		t.Errorf("files URL %q did not match any pattern", filesURL)
	}
}

// TestToolFactories verifies that the tool factories return Tools with the
// correct Type, ID, and Name.
func TestToolFactories(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "test-key")
	p := CreateGoogle(ProviderSettings{})
	tf := p.Tools()

	tests := []struct {
		name string
		tool Tool
	}{
		{"GoogleSearch", tf.GoogleSearch()},
		{"EnterpriseWebSearch", tf.EnterpriseWebSearch()},
		{"GoogleMaps", tf.GoogleMaps()},
		{"UrlContext", tf.UrlContext()},
		{"FileSearch", tf.FileSearch(FileSearchArgs{})},
		{"CodeExecution", tf.CodeExecution()},
		{"VertexRagStore", tf.VertexRagStore(VertexRagStoreArgs{})},
	}

	for _, tc := range tests {
		if tc.tool.Type != "provider" {
			t.Errorf("%s: Type=%q, want 'provider'", tc.name, tc.tool.Type)
		}
		if tc.tool.ID == "" {
			t.Errorf("%s: ID is empty", tc.name)
		}
		if tc.tool.Name == "" {
			t.Errorf("%s: Name is empty", tc.name)
		}
	}
}

// TestGetModelPath verifies getModelPath wraps bare IDs and passes through
// slash-containing paths.
func TestGetModelPath(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"gemini-2.0-flash", "models/gemini-2.0-flash"},
		{"models/gemini-2.0-flash", "models/gemini-2.0-flash"},
		{"projects/p/locations/l/models/m", "projects/p/locations/l/models/m"},
	}
	for _, tc := range tests {
		if got := getModelPath(tc.input); got != tc.want {
			t.Errorf("getModelPath(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// TestFinishReasonMapping covers the mapping table from the spec.
func TestFinishReasonMapping(t *testing.T) {
	tests := []struct {
		raw          string
		hasToolCalls bool
		wantUnified  string
	}{
		{"STOP", false, "stop"},
		{"STOP", true, "tool-calls"},
		{"MAX_TOKENS", false, "length"},
		{"IMAGE_SAFETY", false, "content-filter"},
		{"RECITATION", false, "content-filter"},
		{"SAFETY", false, "content-filter"},
		{"BLOCKLIST", false, "content-filter"},
		{"PROHIBITED_CONTENT", false, "content-filter"},
		{"SPII", false, "content-filter"},
		{"MALFORMED_FUNCTION_CALL", false, "error"},
		{"FINISH_REASON_UNSPECIFIED", false, "other"},
		{"OTHER", false, "other"},
		{"UNKNOWN_FUTURE_VALUE", false, "other"},
	}
	for _, tc := range tests {
		fr := mapGoogleFinishReason(tc.raw, tc.hasToolCalls)
		if fr.Unified != tc.wantUnified {
			t.Errorf("mapGoogleFinishReason(%q, %v).Unified = %q, want %q",
				tc.raw, tc.hasToolCalls, fr.Unified, tc.wantUnified)
		}
		if fr.Raw != tc.raw {
			t.Errorf("mapGoogleFinishReason(%q, %v).Raw = %q, want %q",
				tc.raw, tc.hasToolCalls, fr.Raw, tc.raw)
		}
	}
}

// TestIsVertexLike covers the cross-namespace detection logic.
func TestIsVertexLike(t *testing.T) {
	tests := []struct {
		baseURL  string
		useVAIH  bool
		opts     ProviderOptions
		wantLike bool
	}{
		{defaultBaseURL, false, nil, false},
		{defaultBaseURL, true, nil, true},
		{"https://us-central1-aiplatform.googleapis.com", false, nil, true},
		{defaultBaseURL, false, ProviderOptions{"googleVertex": {"key": "val"}}, true},
		{defaultBaseURL, false, ProviderOptions{"vertex": {"key": "val"}}, true},
		{defaultBaseURL, false, ProviderOptions{"google": {"key": "val"}}, false},
	}
	for _, tc := range tests {
		got := isVertexLike(tc.baseURL, tc.useVAIH, tc.opts)
		if got != tc.wantLike {
			t.Errorf("isVertexLike(%q, %v, %v) = %v, want %v",
				tc.baseURL, tc.useVAIH, tc.opts, got, tc.wantLike)
		}
	}
}
