package cohere

import (
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

const (
	defaultBaseURL                    = "https://api.cohere.com/v2"
	defaultMaxResponseBodyBytes int64 = 32 << 20
	defaultMaxErrorBodyBytes    int64 = 1 << 20
)

type Fetcher interface {
	Do(req *http.Request) (*http.Response, error)
}
type IDGenerator func() string
type RetryOptions struct {
	MaxRetries          int
	BaseDelay, MaxDelay time.Duration
	Jitter              bool
}
type Logger interface {
	Debug(string, ...any)
	Info(string, ...any)
	Warn(string, ...any)
	Error(string, ...any)
}

type ProviderSettings struct {
	BaseURL               string
	APIKey                string
	Headers               http.Header
	Fetch                 Fetcher
	GenerateID            IDGenerator
	Logger                Logger
	Retry                 *RetryOptions
	MaxResponseBodyBytes  int64
	MaxErrorResponseBytes int64
}

var Cohere Provider = CreateCohere(ProviderSettings{})

type cohereProvider struct {
	baseURL, apiKey                             string
	headers                                     http.Header
	fetch                                       Fetcher
	generateID                                  IDGenerator
	logger                                      Logger
	retry                                       RetryOptions
	maxResponseBodyBytes, maxErrorResponseBytes int64
	err                                         error
}

func CreateCohere(settings ProviderSettings) Provider {
	base := strings.TrimRight(settings.BaseURL, "/")
	if base == "" {
		base = defaultBaseURL
	}
	key := settings.APIKey
	if key == "" {
		key = os.Getenv("COHERE_API_KEY")
	}
	p := &cohereProvider{baseURL: base, apiKey: key, headers: cloneHeader(settings.Headers), fetch: settings.Fetch, generateID: settings.GenerateID, logger: settings.Logger, retry: defaultRetryOptions(settings.Retry), maxResponseBodyBytes: maxPositiveOrDefault(settings.MaxResponseBodyBytes, defaultMaxResponseBodyBytes), maxErrorResponseBytes: maxPositiveOrDefault(settings.MaxErrorResponseBytes, defaultMaxErrorBodyBytes)}
	if p.logger == nil {
		p.logger = noopLogger{}
	}
	if p.fetch == nil {
		p.fetch = defaultHTTPClient()
	}
	if key == "" {
		p.err = ErrMissingAPIKey
	} else {
		p.headers.Set("Authorization", "Bearer "+key)
	}
	for k, values := range settings.Headers {
		p.headers.Del(k)
		for _, v := range values {
			p.headers.Add(k, v)
		}
	}
	p.headers.Set("User-Agent", "ai-sdk-go/cohere/"+Version)
	return p
}

func (p *cohereProvider) Model(modelID string) LanguageModel { return p.LanguageModel(modelID) }
func (p *cohereProvider) LanguageModel(modelID string) LanguageModel {
	return &cohereChatLanguageModel{provider: p, modelID: modelID}
}
func (p *cohereProvider) EmbeddingModel(modelID string) EmbeddingModel {
	return &cohereEmbeddingModel{provider: p, modelID: modelID}
}
func (p *cohereProvider) Embedding(modelID string) EmbeddingModel { return p.EmbeddingModel(modelID) }
func (p *cohereProvider) TextEmbeddingModel(modelID string) EmbeddingModel {
	return p.EmbeddingModel(modelID)
}
func (p *cohereProvider) TextEmbedding(modelID string) EmbeddingModel {
	return p.EmbeddingModel(modelID)
}
func (p *cohereProvider) RerankingModel(modelID string) RerankingModel {
	return &cohereRerankingModel{provider: p, modelID: modelID}
}
func (p *cohereProvider) Reranking(modelID string) RerankingModel { return p.RerankingModel(modelID) }
func (p *cohereProvider) Name() string                            { return "cohere" }
func (p *cohereProvider) Err() error                              { return p.err }

type cohereChatLanguageModel struct {
	provider *cohereProvider
	modelID  string
}

func (m *cohereChatLanguageModel) ModelID() string  { return m.modelID }
func (m *cohereChatLanguageModel) Provider() string { return "cohere.chat" }
func (m *cohereChatLanguageModel) SupportURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{"image/*": {regexp.MustCompile(`^https?://.*$`)}}
}

type cohereEmbeddingModel struct {
	provider *cohereProvider
	modelID  string
}

func (m *cohereEmbeddingModel) ModelID() string             { return m.modelID }
func (m *cohereEmbeddingModel) Provider() string            { return "cohere.textEmbedding" }
func (m *cohereEmbeddingModel) MaxEmbeddingsPerCall() int   { return 96 }
func (m *cohereEmbeddingModel) SupportsParallelCalls() bool { return true }

type cohereRerankingModel struct {
	provider *cohereProvider
	modelID  string
}

func (m *cohereRerankingModel) ModelID() string  { return m.modelID }
func (m *cohereRerankingModel) Provider() string { return "cohere.reranking" }

func defaultRetryOptions(opts *RetryOptions) RetryOptions {
	r := RetryOptions{MaxRetries: 2, BaseDelay: 200 * time.Millisecond, MaxDelay: 2 * time.Second, Jitter: true}
	if opts == nil {
		return r
	}
	r.MaxRetries = opts.MaxRetries
	if opts.BaseDelay > 0 {
		r.BaseDelay = opts.BaseDelay
	}
	if opts.MaxDelay > 0 {
		r.MaxDelay = opts.MaxDelay
	}
	r.Jitter = opts.Jitter
	if opts.MaxRetries < 0 {
		r.MaxRetries = 0
	}
	if r.MaxDelay < r.BaseDelay {
		r.MaxDelay = r.BaseDelay
	}
	return r
}
func defaultHTTPClient() *http.Client {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns = 100
	tr.MaxIdleConnsPerHost = 20
	tr.IdleConnTimeout = 90 * time.Second
	tr.TLSHandshakeTimeout = 10 * time.Second
	tr.ExpectContinueTimeout = time.Second
	tr.ResponseHeaderTimeout = 2 * time.Minute
	return &http.Client{Transport: tr}
}

type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
func maxPositiveOrDefault(v, fallback int64) int64 {
	if v > 0 {
		return v
	}
	return fallback
}
