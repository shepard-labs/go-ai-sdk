package openrouter

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

var OpenRouter Provider = CreateOpenRouter(ProviderSettings{Compatibility: CompatibilityStrict})

type openRouterProvider struct {
	baseURL               string
	apiKey                string
	headers               http.Header
	fetch                 Fetcher
	generateID            IDGenerator
	logger                Logger
	retry                 RetryOptions
	compatibility         Compatibility
	extraBody             map[string]any
	apiKeys               map[string]string
	appName               string
	appURL                string
	maxResponseBodyBytes  int64
	maxErrorResponseBytes int64
	err                   error
}

func CreateOpenRouter(settings ProviderSettings) Provider {
	base := settings.BaseURL
	if base == "" {
		base = settings.BaseUrl
	}
	if base == "" {
		base = defaultBaseURL
	}
	compat := settings.Compatibility
	if compat == "" {
		compat = CompatibilityCompatible
	}
	p := &openRouterProvider{
		baseURL:               strings.TrimRight(base, "/"),
		apiKey:                settings.APIKey,
		headers:               cloneHeader(settings.Headers),
		fetch:                 settings.Fetch,
		generateID:            settings.GenerateID,
		logger:                settings.Logger,
		retry:                 defaultRetryOptions(settings.Retry),
		compatibility:         compat,
		extraBody:             cloneMap(settings.ExtraBody),
		apiKeys:               cloneStringMap(settings.APIKeys),
		appName:               settings.AppName,
		appURL:                settings.AppURL,
		maxResponseBodyBytes:  maxPositiveOrDefault(settings.MaxResponseBodyBytes, defaultMaxResponseBodySize),
		maxErrorResponseBytes: maxPositiveOrDefault(settings.MaxErrorResponseBytes, defaultMaxErrorBodySize),
	}
	if p.fetch == nil {
		p.fetch = defaultHTTPClient()
	}
	if p.generateID == nil {
		p.generateID = randomID
	}
	if p.logger == nil {
		p.logger = noopLogger{}
	}
	return p
}

func (p *openRouterProvider) Model(modelID string, opts ...ChatOptions) LanguageModel {
	return p.LanguageModel(modelID, opts...)
}

func (p *openRouterProvider) LanguageModel(modelID string, opts ...ChatOptions) LanguageModel {
	if modelID == "openai/gpt-3.5-turbo-instruct" {
		return p.Completion(modelID)
	}
	return p.Chat(modelID, opts...)
}

func (p *openRouterProvider) Chat(modelID string, opts ...ChatOptions) LanguageModel {
	var opt ChatOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return &chatModel{provider: p, modelID: modelID, options: opt}
}

func (p *openRouterProvider) Completion(modelID string, opts ...CompletionOptions) LanguageModel {
	var opt CompletionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return &completionModel{provider: p, modelID: modelID, options: opt}
}

func (p *openRouterProvider) TextEmbeddingModel(modelID string, opts ...EmbeddingOptions) EmbeddingModel {
	return p.Embedding(modelID, opts...)
}

func (p *openRouterProvider) Embedding(modelID string, opts ...EmbeddingOptions) EmbeddingModel {
	var opt EmbeddingOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return &embeddingModel{provider: p, modelID: modelID, options: opt}
}

func (p *openRouterProvider) ImageModel(modelID string, opts ...ImageOptions) ImageModel {
	return p.Image(modelID, opts...)
}

func (p *openRouterProvider) Image(modelID string, opts ...ImageOptions) ImageModel {
	var opt ImageOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return &imageModel{provider: p, modelID: modelID, options: opt}
}

func (p *openRouterProvider) VideoModel(modelID string, opts ...VideoOptions) VideoModel {
	var opt VideoOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	return &videoModel{provider: p, modelID: modelID, options: opt}
}

func (p *openRouterProvider) Tools() Tools { return Tools{} }
func (p *openRouterProvider) Name() string { return "openrouter" }
func (p *openRouterProvider) Err() error   { return p.err }

func (p *openRouterProvider) requestHeaders(call http.Header) http.Header {
	h := make(http.Header)
	apiKey := p.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_API_KEY")
	}
	if apiKey != "" {
		h.Set("Authorization", "Bearer "+apiKey)
	}
	h.Set("User-Agent", "ai-sdk-go/openrouter/"+Version)
	if p.appName != "" {
		h.Set("X-OpenRouter-Title", p.appName)
	}
	if p.appURL != "" {
		h.Set("HTTP-Referer", p.appURL)
	}
	if len(p.apiKeys) > 0 {
		if b, err := json.Marshal(p.apiKeys); err == nil {
			h.Set("X-Provider-API-Keys", string(b))
		}
	}
	mergeHeader(h, p.headers)
	mergeHeader(h, call)
	return h
}

func (p *openRouterProvider) nextID() string { return p.generateID() }

type baseModel struct {
	provider *openRouterProvider
	modelID  string
}

func (m baseModel) ModelID() string { return m.modelID }

var chatSupportURLs = map[string][]*regexp.Regexp{
	"image/*":       {regexp.MustCompile(`^data:image/`), regexp.MustCompile(`^https?://.*\.(?i:jpg|jpeg|png|gif|webp)(?:\?.*)?$`)},
	"application/*": {regexp.MustCompile(`^data:application/`), regexp.MustCompile(`^https?://`)},
}

var completionSupportURLs = map[string][]*regexp.Regexp{
	"image/*":       {regexp.MustCompile(`^data:image/`), regexp.MustCompile(`^https?://.*\.(?i:jpg|jpeg|png|gif|webp)(?:\?.*)?$`)},
	"text/*":        {regexp.MustCompile(`^data:text/`), regexp.MustCompile(`^https?://`)},
	"application/*": {regexp.MustCompile(`^data:application/`), regexp.MustCompile(`^https?://`)},
}

func cloneRegexpMap(in map[string][]*regexp.Regexp) map[string][]*regexp.Regexp {
	out := make(map[string][]*regexp.Regexp, len(in))
	for k, v := range in {
		out[k] = append([]*regexp.Regexp(nil), v...)
	}
	return out
}

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
	if retry.MaxRetries < 0 {
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

func maxPositiveOrDefault(value, fallback int64) int64 {
	if value > 0 {
		return value
	}
	return fallback
}

func randomID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return hex.EncodeToString([]byte(time.Now().Format(time.RFC3339Nano)))
	}
	return hex.EncodeToString(b[:])
}

type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
