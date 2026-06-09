package openrouter

import (
	"net/http"
	"time"
)

const (
	Version                    = "2.9.0"
	defaultBaseURL             = "https://openrouter.ai/api/v1"
	defaultMaxResponseBodySize = int64(16 << 20)
	defaultMaxErrorBodySize    = int64(1 << 20)
	defaultMaxEmbeddings       = 2048
	defaultMaxImages           = 1
	defaultMaxVideos           = 1
)

type Compatibility string

const (
	CompatibilityStrict     Compatibility = "strict"
	CompatibilityCompatible Compatibility = "compatible"
)

type ProviderSettings struct {
	BaseURL               string
	BaseUrl               string // deprecated alias
	APIKey                string
	Headers               http.Header
	Fetch                 Fetcher
	GenerateID            IDGenerator
	Logger                Logger
	Retry                 *RetryOptions
	Compatibility         Compatibility
	ExtraBody             map[string]any
	APIKeys               map[string]string
	AppName               string
	AppURL                string
	MaxResponseBodyBytes  int64
	MaxErrorResponseBytes int64
}

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

type Logger interface {
	Debug(msg string, keysAndValues ...any)
	Info(msg string, keysAndValues ...any)
	Warn(msg string, keysAndValues ...any)
	Error(msg string, keysAndValues ...any)
}

type CacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type DebugOptions struct {
	EchoUpstreamBody *bool `json:"echo_upstream_body,omitempty"`
}

type StructuredOutputsOptions struct {
	Strict *bool
}

type LogprobsOptions struct {
	Enabled *bool
	Top     *int
}

type ReasoningOptions struct {
	Enabled   *bool           `json:"enabled,omitempty"`
	Exclude   *bool           `json:"exclude,omitempty"`
	MaxTokens *int            `json:"max_tokens,omitempty"`
	Effort    ReasoningEffort `json:"effort,omitempty"`
}

type ReasoningEffort string

const (
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortNone    ReasoningEffort = "none"
)

type UsageOptions struct {
	Include bool `json:"include"`
}

type ChatOptions struct {
	Models            []string
	LogitBias         map[int]float64
	Logprobs          *LogprobsOptions
	ParallelToolCalls *bool
	User              string
	Plugins           []Plugin
	WebSearchOptions  *WebSearchOptions
	CacheControl      *CacheControl
	Debug             *DebugOptions
	StructuredOutputs *StructuredOutputsOptions
	Provider          *ProviderRouting
	IncludeReasoning  *bool
	Reasoning         *ReasoningOptions
	Usage             *UsageOptions
	Temperature       *float64
	TopP              *float64
	TopK              *int
	FrequencyPenalty  *float64
	PresencePenalty   *float64
	MaxTokens         *int
	ExtraBody         map[string]any
}

type CompletionOptions struct {
	Models           []string
	LogitBias        map[int]float64
	Logprobs         *LogprobsOptions
	Suffix           string
	User             string
	IncludeReasoning *bool
	Reasoning        *ReasoningOptions
	Temperature      *float64
	TopP             *float64
	TopK             *int
	FrequencyPenalty *float64
	PresencePenalty  *float64
	MaxTokens        *int
	ExtraBody        map[string]any
}

type EmbeddingOptions struct {
	User      string
	Provider  *EmbeddingProviderRouting
	ExtraBody map[string]any
}

type ImageOptions struct {
	User      string
	Provider  *ImageProviderRouting
	ExtraBody map[string]any
}

type VideoOptions struct {
	GenerateAudio *bool
	PollInterval  time.Duration
	MaxPollTime   time.Duration
	ExtraBody     map[string]any
}

type ProviderOptions map[string]map[string]any

func (o ProviderOptions) OpenRouter() map[string]any {
	if o == nil {
		return nil
	}
	return o["openrouter"]
}
