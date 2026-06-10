package google

import (
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

const (
	defaultBaseURL                    = "https://generativelanguage.googleapis.com/v1beta"
	defaultProviderName               = "google.generative-ai"
	defaultMaxResponseBodyBytes int64 = 32 << 20
	defaultMaxErrorBodyBytes    int64 = 1 << 20
	defaultMaxEmbeddingsPerCall       = 2048
	defaultMaxImagesPerCall           = 10
	defaultMaxVideosPerCall           = 4
)

// Google is the default Provider instance. It reads the API key from the
// GOOGLE_GENERATIVE_AI_API_KEY environment variable.
var Google Provider = CreateGoogle(ProviderSettings{})

// ProviderSettings configures a Google Generative AI provider.
type ProviderSettings struct {
	// BaseURL overrides the default API endpoint
	// (https://generativelanguage.googleapis.com/v1beta).
	BaseURL string
	// APIKey is the Google API key. When empty, the GOOGLE_GENERATIVE_AI_API_KEY
	// environment variable is used.
	APIKey string
	// Headers are merged into every outgoing request.
	Headers http.Header
	// QueryParams are merged into every request URL.
	QueryParams map[string]string
	// Fetch overrides the HTTP client.
	Fetch Fetcher
	// GenerateID generates per-request IDs (set as x-request-id header).
	GenerateID IDGenerator
	// Name overrides the provider name (default: "google.generative-ai").
	Name string
	// Logger receives operational logs.
	Logger Logger
	// Retry configures retry behavior.
	Retry *RetryOptions
	// MaxResponseBodyBytes caps the response body size (default: 32 MiB).
	MaxResponseBodyBytes int64
	// MaxErrorResponseBytes caps the error response body size (default: 1 MiB).
	MaxErrorResponseBytes int64
	// MaxEmbeddingsPerCall limits the number of embedding values per batch call
	// (default: 2048).
	MaxEmbeddingsPerCall int
	// SupportsParallelEmbeddingCalls controls whether the embedding model
	// advertises parallel-call support (default: true).
	SupportsParallelEmbeddingCalls *bool

	// UseVertexAIHeaders, when true, attaches Vertex AI LLM request-type
	// headers and adjusts wire behavior to match Vertex endpoints.
	UseVertexAIHeaders bool
	// DisableHeaderAuth, when true, suppresses the x-goog-api-key header.
	// By default (false) the header is sent. This field name avoids the
	// zero-value ambiguity of a "UseHeaderAuth bool" field whose default
	// would need to be true.
	DisableHeaderAuth bool

	// ErrorStructure provides optional hooks for parsing non-standard error
	// bodies (e.g. Vertex-specific error shapes).
	ErrorStructure ProviderErrorStructure
}

type googleProvider struct {
	baseURL                        string
	name                           string
	headers                        http.Header
	queryParams                    map[string]string
	fetch                          Fetcher
	generateID                     IDGenerator
	logger                         Logger
	retry                          RetryOptions
	maxResponseBodyBytes           int64
	maxErrorResponseBytes          int64
	maxEmbeddingsPerCall           int
	supportsParallelEmbeddingCalls bool
	isVertex                       bool // true when baseURL targets aiplatform.googleapis.com
	useVertexAIHeaders             bool
	errorStructure                 ProviderErrorStructure
	apiKey                         string
	err                            error
}

// CreateGoogle creates a Google Generative AI provider.
//
// If APIKey is empty and GOOGLE_GENERATIVE_AI_API_KEY is not set, the
// returned Provider's Err() returns ErrMissingAPIKey and every model call
// fails without issuing HTTP requests.
func CreateGoogle(settings ProviderSettings) Provider {
	baseURL := settings.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	apiKey := settings.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY")
	}

	name := settings.Name
	if name == "" {
		name = defaultProviderName
	}

	p := &googleProvider{
		baseURL:                        baseURL,
		name:                           name,
		headers:                        cloneHeader(settings.Headers),
		queryParams:                    cloneStringMap(settings.QueryParams),
		fetch:                          settings.Fetch,
		generateID:                     settings.GenerateID,
		logger:                         settings.Logger,
		retry:                          defaultRetryOptions(settings.Retry),
		maxResponseBodyBytes:           maxPositiveOrDefault(settings.MaxResponseBodyBytes, defaultMaxResponseBodyBytes),
		maxErrorResponseBytes:          maxPositiveOrDefault(settings.MaxErrorResponseBytes, defaultMaxErrorBodyBytes),
		maxEmbeddingsPerCall:           settings.MaxEmbeddingsPerCall,
		supportsParallelEmbeddingCalls: true,
		isVertex:                       isVertexProvider(baseURL, settings.UseVertexAIHeaders),
		useVertexAIHeaders:             settings.UseVertexAIHeaders,
		errorStructure:                 settings.ErrorStructure,
		apiKey:                         apiKey,
	}

	if p.logger == nil {
		p.logger = noopLogger{}
	}
	if p.fetch == nil {
		p.fetch = defaultHTTPClient()
	}
	if p.maxEmbeddingsPerCall <= 0 {
		p.maxEmbeddingsPerCall = defaultMaxEmbeddingsPerCall
	}
	if settings.SupportsParallelEmbeddingCalls != nil {
		p.supportsParallelEmbeddingCalls = *settings.SupportsParallelEmbeddingCalls
	}

	// Build auth + default headers.
	// UseHeaderAuth defaults to true (sending x-goog-api-key). The caller must
	// explicitly set DisableHeaderAuth to opt out; because UseHeaderAuth is bool
	// its zero value would mean "disabled", so we treat the zero value as enabled.
	if apiKey != "" && !settings.DisableHeaderAuth {
		p.headers.Set("x-goog-api-key", apiKey)
	}
	p.headers.Set("User-Agent", "ai-sdk-go/google/"+Version)

	if apiKey == "" {
		p.err = ErrMissingAPIKey
	}

	return p
}

// defaultRetryOptions resolves user-supplied RetryOptions into a fully populated
// RetryOptions struct. Matches openaicompatible's behavior.
func defaultRetryOptions(opts *RetryOptions) RetryOptions {
	retry := RetryOptions{
		MaxRetries: 2,
		BaseDelay:  200 * time.Millisecond,
		MaxDelay:   2 * time.Second,
		Jitter:     true,
	}
	if opts == nil {
		return retry
	}
	retry.MaxRetries = opts.MaxRetries
	if opts.BaseDelay > 0 {
		retry.BaseDelay = opts.BaseDelay
	}
	if opts.MaxDelay > 0 {
		retry.MaxDelay = opts.MaxDelay
	}
	retry.Jitter = opts.Jitter
	if opts.MaxRetries < 0 {
		retry.MaxRetries = 0
	}
	if retry.MaxDelay < retry.BaseDelay {
		retry.MaxDelay = retry.BaseDelay
	}
	return retry
}

// ---- Provider interface methods ----

func (p *googleProvider) Name() string { return p.name }
func (p *googleProvider) Err() error   { return p.err }

func (p *googleProvider) Model(modelID string) LanguageModel {
	return p.languageModel(modelID)
}
func (p *googleProvider) LanguageModel(modelID string) LanguageModel {
	return p.languageModel(modelID)
}
func (p *googleProvider) ChatModel(modelID string) LanguageModel {
	return p.languageModel(modelID)
}
func (p *googleProvider) Chat(modelID string) LanguageModel {
	return p.languageModel(modelID)
}

// GenerativeAI is a deprecated alias for LanguageModel, kept for upstream parity.
func (p *googleProvider) GenerativeAI(modelID string) LanguageModel {
	return p.languageModel(modelID)
}

func (p *googleProvider) EmbeddingModel(modelID string) EmbeddingModel {
	return p.embeddingModel(modelID)
}
func (p *googleProvider) Embedding(modelID string) EmbeddingModel {
	return p.embeddingModel(modelID)
}

// TextEmbeddingModel is a deprecated alias for EmbeddingModel.
func (p *googleProvider) TextEmbeddingModel(modelID string) EmbeddingModel {
	return p.embeddingModel(modelID)
}

// TextEmbedding is a deprecated alias for EmbeddingModel.
func (p *googleProvider) TextEmbedding(modelID string) EmbeddingModel {
	return p.embeddingModel(modelID)
}

func (p *googleProvider) ImageModel(modelID string, settings ...ImageModelSettings) ImageModel {
	return p.imageModel(modelID, settings...)
}
func (p *googleProvider) Image(modelID string, settings ...ImageModelSettings) ImageModel {
	return p.imageModel(modelID, settings...)
}

func (p *googleProvider) VideoModel(modelID string) VideoModel {
	return p.videoModel(modelID)
}
func (p *googleProvider) Video(modelID string) VideoModel {
	return p.videoModel(modelID)
}

func (p *googleProvider) SpeechModel(modelID string) SpeechModel {
	return p.speechModel(modelID)
}
func (p *googleProvider) Speech(modelID string) SpeechModel {
	return p.speechModel(modelID)
}

func (p *googleProvider) Files() Files {
	return p.filesClient()
}

func (p *googleProvider) Tools() ToolFactories {
	return p.toolFactories()
}

// SupportURLs returns the URL patterns that Google language models support as
// file inputs. This is a provider-level method; individual model structs delegate
// to it.
func (p *googleProvider) SupportURLs() map[string][]*regexp.Regexp {
	// Escape the base URL for use in a regexp literal.
	escaped := regexp.QuoteMeta(p.baseURL)
	return map[string][]*regexp.Regexp{
		"*": {
			regexp.MustCompile(`^` + escaped + `/files/.*$`),
			regexp.MustCompile(`^https://(www\.)?youtube\.com/watch\?v=[\w-]+(?:&[\w=&.-]*)?$`),
			regexp.MustCompile(`^https://youtu\.be/[\w-]+(?:\?[\w=&.-]*)?$`),
		},
	}
}

// ---- factory helpers (delegate to model_stubs.go) ----

func (p *googleProvider) languageModel(modelID string) LanguageModel {
	return &googleLanguageModel{provider: p, modelID: modelID}
}

func (p *googleProvider) embeddingModel(modelID string) EmbeddingModel {
	return &googleEmbeddingModel{provider: p, modelID: modelID}
}

func (p *googleProvider) imageModel(modelID string, settings ...ImageModelSettings) ImageModel {
	var s ImageModelSettings
	if len(settings) > 0 {
		s = settings[0]
	}
	return &googleImageModel{provider: p, modelID: modelID, settings: s}
}

func (p *googleProvider) videoModel(modelID string) VideoModel {
	return &googleVideoModel{provider: p, modelID: modelID}
}

func (p *googleProvider) speechModel(modelID string) SpeechModel {
	return &googleSpeechModel{provider: p, modelID: modelID}
}

func (p *googleProvider) filesClient() Files {
	return &googleFiles{provider: p}
}

func (p *googleProvider) toolFactories() ToolFactories {
	return buildToolFactories()
}

// ---- noop logger ----

type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
