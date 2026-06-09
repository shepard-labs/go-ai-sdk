package openaicompatible

import (
	"errors"
	"net/http"
	"regexp"
	"strings"
	"time"
)

const (
	defaultMaxResponseBodyBytes int64 = 32 << 20
	defaultMaxErrorBodyBytes    int64 = 1 << 20
	defaultMaxEmbeddingsPerCall       = 2048
	defaultMaxImagesPerCall           = 10
)

// Fetcher executes HTTP requests.
type Fetcher interface {
	Do(req *http.Request) (*http.Response, error)
}

// IDGenerator generates request IDs.
type IDGenerator func() string

// RetryOptions configures retry behavior.
type RetryOptions struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
	Jitter     bool
}

// Logger receives operational request logs.
type Logger interface {
	Debug(msg string, keysAndValues ...any)
	Info(msg string, keysAndValues ...any)
	Warn(msg string, keysAndValues ...any)
	Error(msg string, keysAndValues ...any)
}

// ProviderSettings configures an OpenAI-compatible provider.
type ProviderSettings struct {
	BaseURL                        string
	Name                           string
	APIKey                         string
	Headers                        http.Header
	QueryParams                    map[string]string
	Fetch                          Fetcher
	GenerateID                     IDGenerator
	Logger                         Logger
	Retry                          *RetryOptions
	MaxResponseBodyBytes           int64
	MaxErrorResponseBytes          int64
	MaxEmbeddingsPerCall           int
	SupportsParallelEmbeddingCalls *bool
	IncludeUsage                   bool
	SupportsStructuredOutputs      bool
	TransformRequestBody           func(map[string]any) map[string]any
	MetadataExtractor              MetadataExtractor
	SupportedURLs                  func() map[string][]*regexp.Regexp
	ConvertUsage                   func(OpenAICompatibleTokenUsage) Usage
	ErrorStructure                 ProviderErrorStructure
}

type openAICompatibleProvider struct {
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
	includeUsage                   bool
	supportsStructuredOutputs      bool
	transformRequestBody           func(map[string]any) map[string]any
	metadataExtractor              MetadataExtractor
	supportedURLs                  func() map[string][]*regexp.Regexp
	convertUsage                   func(OpenAICompatibleTokenUsage) Usage
	errorStructure                 ProviderErrorStructure
	err                            error
}

// CreateOpenAICompatible creates an OpenAI-compatible provider.
func CreateOpenAICompatible(settings ProviderSettings) Provider {
	p := &openAICompatibleProvider{
		baseURL:                        strings.TrimRight(settings.BaseURL, "/"),
		name:                           settings.Name,
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
		includeUsage:                   settings.IncludeUsage,
		supportsStructuredOutputs:      settings.SupportsStructuredOutputs,
		transformRequestBody:           settings.TransformRequestBody,
		metadataExtractor:              settings.MetadataExtractor,
		supportedURLs:                  settings.SupportedURLs,
		convertUsage:                   settings.ConvertUsage,
		errorStructure:                 settings.ErrorStructure,
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
	if settings.APIKey != "" {
		p.headers.Set("Authorization", "Bearer "+settings.APIKey)
		for k, values := range settings.Headers {
			p.headers.Del(k)
			for _, v := range values {
				p.headers.Add(k, v)
			}
		}
	}
	p.headers.Set("User-Agent", "ai-sdk-go/openai-compatible/"+Version)
	p.err = providerSettingsError(settings.BaseURL, settings.Name)
	return p
}

func providerSettingsError(baseURL, name string) error {
	var errs []error
	if baseURL == "" {
		errs = append(errs, ErrMissingBaseURL)
	}
	if strings.TrimSpace(name) == "" {
		errs = append(errs, ErrMissingName)
	}
	return errors.Join(errs...)
}

func (p *openAICompatibleProvider) Model(modelID string) LanguageModel { return p.ChatModel(modelID) }
func (p *openAICompatibleProvider) LanguageModel(modelID string) LanguageModel {
	return p.ChatModel(modelID)
}
func (p *openAICompatibleProvider) ChatModel(modelID string) LanguageModel {
	return &openAICompatibleChatLanguageModel{provider: p, modelID: modelID}
}
func (p *openAICompatibleProvider) Chat(modelID string) LanguageModel { return p.ChatModel(modelID) }
func (p *openAICompatibleProvider) CompletionModel(modelID string) LanguageModel {
	return &openAICompatibleCompletionLanguageModel{provider: p, modelID: modelID}
}
func (p *openAICompatibleProvider) Completion(modelID string) LanguageModel {
	return p.CompletionModel(modelID)
}
func (p *openAICompatibleProvider) EmbeddingModel(modelID string) EmbeddingModel {
	return &openAICompatibleEmbeddingModel{provider: p, modelID: modelID}
}
func (p *openAICompatibleProvider) Embedding(modelID string) EmbeddingModel {
	return p.EmbeddingModel(modelID)
}

// TextEmbeddingModel is deprecated. Use EmbeddingModel.
func (p *openAICompatibleProvider) TextEmbeddingModel(modelID string) EmbeddingModel {
	return p.EmbeddingModel(modelID)
}
func (p *openAICompatibleProvider) ImageModel(modelID string) ImageModel {
	return &openAICompatibleImageModel{provider: p, modelID: modelID}
}
func (p *openAICompatibleProvider) Image(modelID string) ImageModel { return p.ImageModel(modelID) }
func (p *openAICompatibleProvider) Name() string                    { return p.name }
func (p *openAICompatibleProvider) Err() error                      { return p.err }

type openAICompatibleChatLanguageModel struct {
	provider *openAICompatibleProvider
	modelID  string
}

func (m *openAICompatibleChatLanguageModel) ModelID() string  { return m.modelID }
func (m *openAICompatibleChatLanguageModel) Provider() string { return m.provider.name + ".chat" }
func (m *openAICompatibleChatLanguageModel) SupportURLs() map[string][]*regexp.Regexp {
	if m.provider.supportedURLs == nil {
		return map[string][]*regexp.Regexp{}
	}
	return cloneRegexpMap(m.provider.supportedURLs())
}

type openAICompatibleCompletionLanguageModel struct {
	provider *openAICompatibleProvider
	modelID  string
}

func (m *openAICompatibleCompletionLanguageModel) ModelID() string { return m.modelID }
func (m *openAICompatibleCompletionLanguageModel) Provider() string {
	return m.provider.name + ".completion"
}
func (m *openAICompatibleCompletionLanguageModel) SupportURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{}
}

type openAICompatibleEmbeddingModel struct {
	provider *openAICompatibleProvider
	modelID  string
}

func (m *openAICompatibleEmbeddingModel) ModelID() string { return m.modelID }
func (m *openAICompatibleEmbeddingModel) Provider() string {
	return m.provider.name + ".embedding"
}
func (m *openAICompatibleEmbeddingModel) MaxEmbeddingsPerCall() int {
	return m.provider.maxEmbeddingsPerCall
}
func (m *openAICompatibleEmbeddingModel) SupportsParallelCalls() bool {
	return m.provider.supportsParallelEmbeddingCalls
}

type openAICompatibleImageModel struct {
	provider *openAICompatibleProvider
	modelID  string
}

func (m *openAICompatibleImageModel) ModelID() string       { return m.modelID }
func (m *openAICompatibleImageModel) Provider() string      { return m.provider.name + ".image" }
func (m *openAICompatibleImageModel) MaxImagesPerCall() int { return defaultMaxImagesPerCall }

func defaultRetryOptions(opts *RetryOptions) RetryOptions {
	retry := RetryOptions{MaxRetries: 2, BaseDelay: 200 * time.Millisecond, MaxDelay: 2 * time.Second, Jitter: true}
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

func defaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 100
	transport.MaxIdleConnsPerHost = 20
	transport.IdleConnTimeout = 90 * time.Second
	transport.TLSHandshakeTimeout = 10 * time.Second
	transport.ExpectContinueTimeout = time.Second
	transport.ResponseHeaderTimeout = 2 * time.Minute
	return &http.Client{Transport: transport}
}

type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}

func maxPositiveOrDefault(value, fallback int64) int64 {
	if value > 0 {
		return value
	}
	return fallback
}
