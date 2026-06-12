package anthropic

import (
	"errors"
	"math/rand"
	"net/http"
	"os"
	"time"
)

const defaultBaseURL = "https://api.anthropic.com/v1"
const anthropicVersion = "2023-06-01"
const defaultMaxResponseBodyBytes int64 = 32 << 20
const defaultMaxErrorBodyBytes int64 = 1 << 20

var ErrMultipleAuthMethods = errors.New("anthropic: APIKey and AuthToken cannot both be set")

type Fetcher interface {
	Do(req *http.Request) (*http.Response, error)
}

type IDGenerator func() string

type RetryOptions struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
	Jitter     bool
}

type ProviderSettings struct {
	BaseURL               string
	APIKey                string
	AuthToken             string
	Headers               http.Header
	Fetch                 Fetcher
	GenerateID            IDGenerator
	Name                  string
	Logger                Logger
	Retry                 *RetryOptions
	MaxResponseBodyBytes  int64
	MaxErrorResponseBytes int64
}

var Anthropic = CreateAnthropic(ProviderSettings{})

type anthropicProvider struct {
	baseURL               string
	headers               http.Header
	fetch                 Fetcher
	generateID            IDGenerator
	name                  string
	logger                Logger
	retry                 RetryOptions
	maxResponseBodyBytes  int64
	maxErrorResponseBytes int64
	err                   error
}

func CreateAnthropic(opts ProviderSettings) Provider {
	p := &anthropicProvider{
		baseURL:               valueOrDefault(opts.BaseURL, defaultBaseURL),
		headers:               make(http.Header),
		fetch:                 opts.Fetch,
		generateID:            opts.GenerateID,
		name:                  valueOrDefault(opts.Name, "anthropic"),
		logger:                opts.Logger,
		retry:                 defaultRetryOptions(opts.Retry),
		maxResponseBodyBytes:  maxPositiveOrDefault(opts.MaxResponseBodyBytes, defaultMaxResponseBodyBytes),
		maxErrorResponseBytes: maxPositiveOrDefault(opts.MaxErrorResponseBytes, defaultMaxErrorBodyBytes),
	}
	if p.logger == nil {
		p.logger = noopLogger{}
	}
	if p.fetch == nil {
		p.fetch = defaultHTTPClient()
	}

	for k, values := range opts.Headers {
		for _, v := range values {
			p.headers.Add(k, v)
		}
	}

	apiKey := opts.APIKey
	authToken := opts.AuthToken
	if apiKey == "" && authToken == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		authToken = os.Getenv("ANTHROPIC_AUTH_TOKEN")
	}
	if apiKey != "" && authToken != "" {
		p.err = ErrMultipleAuthMethods
	}
	if apiKey != "" {
		p.headers.Set("x-api-key", apiKey)
	}
	if authToken != "" {
		p.headers.Set("Authorization", "Bearer "+authToken)
	}
	p.headers.Set("anthropic-version", anthropicVersion)
	p.headers.Set("User-Agent", "ai-sdk-go/anthropic/"+Version)

	return p
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

func (p *anthropicProvider) retryDelay(attempt int, resp *http.Response) time.Duration {
	if resp != nil {
		if delay, ok := retryAfterDelay(resp.Header.Get("Retry-After")); ok {
			return delay
		}
	}
	delay := p.retry.BaseDelay
	for range attempt {
		delay *= 2
		if delay >= p.retry.MaxDelay {
			delay = p.retry.MaxDelay
			break
		}
	}
	if p.retry.Jitter && delay > 0 {
		return time.Duration(rand.Int63n(int64(delay)))
	}
	return delay
}

func retryAfterDelay(value string) (time.Duration, bool) {
	if value == "" {
		return 0, false
	}
	if seconds, err := time.ParseDuration(value + "s"); err == nil {
		return seconds, true
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	delay := time.Until(when)
	if delay < 0 {
		delay = 0
	}
	return delay, true
}

func (p *anthropicProvider) Model(modelID string, opts ...ModelOptions) LanguageModel {
	return p.LanguageModel(modelID, opts...)
}

func (p *anthropicProvider) LanguageModel(modelID string, opts ...ModelOptions) LanguageModel {
	mo := ModelOptions{StructuredOutputMode: StructuredOutputModeAuto, Effort: "high"}
	toolStreaming := true
	mo.ToolStreaming = &toolStreaming
	if len(opts) > 0 {
		mo = opts[0]
		if mo.StructuredOutputMode == "" {
			mo.StructuredOutputMode = StructuredOutputModeAuto
		}
		if mo.Effort == "" {
			mo.Effort = "high"
		}
		if mo.ToolStreaming == nil {
			mo.ToolStreaming = &toolStreaming
		}
	}
	return &anthropicLanguageModel{provider: p, modelID: modelID, options: mo}
}

func (p *anthropicProvider) Chat(modelID string, opts ...ModelOptions) LanguageModel {
	return p.LanguageModel(modelID, opts...)
}

func (p *anthropicProvider) Messages(modelID string, opts ...ModelOptions) LanguageModel {
	return p.LanguageModel(modelID, opts...)
}

func (p *anthropicProvider) Tools() Tools { return Tools{ToolNameMapping: ToolNameMapping{}} }
func (p *anthropicProvider) Name() string { return p.name }
func (p *anthropicProvider) Err() error   { return p.err }

func valueOrDefault(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}

func maxPositiveOrDefault(value, fallback int64) int64 {
	if value > 0 {
		return value
	}
	return fallback
}

func (p *anthropicProvider) headersForOptions(opts ModelOptions) http.Header {
	headers := make(http.Header)
	for k, values := range p.headers {
		for _, v := range values {
			headers.Add(k, v)
		}
	}
	for _, beta := range betaHeadersForOptions(opts) {
		headers.Add("anthropic-beta", beta)
	}
	return headers
}

func betaHeadersForOptions(opts ModelOptions) []string {
	seen := map[string]struct{}{}
	var betas []string
	add := func(beta string) {
		if beta == "" {
			return
		}
		if _, ok := seen[beta]; ok {
			return
		}
		seen[beta] = struct{}{}
		betas = append(betas, beta)
	}
	if opts.StructuredOutputMode != "" && opts.StructuredOutputMode != StructuredOutputModeAuto {
		add("structured-outputs-2025-11-13")
	}
	if opts.Container != nil && len(opts.Container.Skills) > 0 {
		add("skills-2025-10-02")
		add("files-api-2025-04-14")
	}
	if len(opts.MCPServers) > 0 {
		add("mcp-client-2025-11-20")
	}
	if opts.ContextManagement != nil && len(opts.ContextManagement.Edits) > 0 {
		add("context-management-2025-06-27")
		for _, edit := range opts.ContextManagement.Edits {
			if _, ok := edit.(CompactEdit); ok {
				add("compact-2026-01-12")
			}
		}
	}
	if opts.TaskBudget != nil {
		add("task-budgets-2026-03-13")
	}
	if opts.Speed == "fast" {
		add("fast-mode-2026-02-01")
	}
	for _, tool := range opts.RequestTools {
		switch tool.ID {
		case "anthropic.code_execution_20250522":
			add("code-execution-2025-05-22")
		case "anthropic.code_execution_20250825", "anthropic.code_execution_20260120":
			add("code-execution-2025-08-25")
		case "anthropic.computer_20251124":
			add("computer-use-2025-11-24")
		case "anthropic.text_editor_20250728":
			add("text-editor-2025-07-28")
		case "anthropic.web_fetch_20250910":
			add("web-fetch-2025-09-10")
		case "anthropic.web_search_20250305":
			add("web-search-2025-03-05")
		case "anthropic.web_fetch_20260209", "anthropic.web_search_20260209":
			add("code-execution-web-tools-2026-02-09")
		case "anthropic.tool_search_regex_20251119", "anthropic.tool_search_bm25_20251119":
			add("tool-search-2025-11-19")
		case "anthropic.advisor_20260301":
			add("advisor-tool-2026-03-01")
		}
		if len(tool.InputExamples) > 0 || len(tool.AllowedCallers) > 0 {
			add("advanced-tool-use-2025-11-20")
		}
	}
	for _, beta := range opts.AnthropicBeta {
		add(beta)
	}
	return betas
}
