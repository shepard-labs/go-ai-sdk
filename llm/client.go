// Package llm defines provider-neutral LLM requests, tool calls, and agent loops.
package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Client is the single interface for LLM completion calls.
type Client interface {
	Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error)
	// Stream initiates a streaming completion. The returned channel emits
	// StreamPart values in order and is closed when the stream ends; exactly one
	// StreamFinish (on success) or StreamError (on failure) is emitted before
	// close. Implementations that do not support streaming return
	// ErrStreamNotImplemented from Stream itself (not via a StreamError part).
	// spec §1.1
	Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error)
}

// ErrStreamNotImplemented indicates the provider adapter does not implement
// streaming. Returned directly from Stream (not wrapped in a StreamError part)
// so callers can distinguish "unsupported" from "stream failed". spec §1.1
var ErrStreamNotImplemented = errors.New("llm: streaming not implemented for this provider")

// StreamPart is a tagged union of streaming events. The channel closes when
// the stream ends; exactly one StreamFinish or StreamError precedes close.
// spec §1.1
type StreamPart interface{ isStreamPart() }

// StreamTextStart marks the beginning of a text content block.
type StreamTextStart struct{}

func (StreamTextStart) isStreamPart() {}

// StreamTextDelta carries an incremental text fragment.
type StreamTextDelta struct{ Text string }

func (StreamTextDelta) isStreamPart() {}

// StreamTextEnd marks the end of a text content block.
type StreamTextEnd struct{}

func (StreamTextEnd) isStreamPart() {}

// StreamReasoningStart marks the beginning of a reasoning/thinking block.
type StreamReasoningStart struct{}

func (StreamReasoningStart) isStreamPart() {}

// StreamReasoningDelta carries an incremental reasoning fragment.
type StreamReasoningDelta struct{ Text string }

func (StreamReasoningDelta) isStreamPart() {}

// StreamReasoningEnd marks the end of a reasoning block.
type StreamReasoningEnd struct{}

func (StreamReasoningEnd) isStreamPart() {}

// StreamToolCallStart marks the beginning of a tool-call input block and
// carries the tool-use id and tool name. spec §1.1
type StreamToolCallStart struct {
	ID   string
	Name string
}

func (StreamToolCallStart) isStreamPart() {}

// StreamToolInputDelta carries a partial-JSON fragment of a tool-call input.
// spec §1.1
type StreamToolInputDelta struct {
	ID   string
	JSON string
}

func (StreamToolInputDelta) isStreamPart() {}

// StreamToolInputEnd marks the end of a tool-call input block and carries the
// final parsed input. spec §1.1
type StreamToolInputEnd struct {
	ID    string
	Input json.RawMessage
}

func (StreamToolInputEnd) isStreamPart() {}

// StreamFinish is the terminal success part carrying the finish reason and
// provider-reported token usage. Exactly one StreamFinish is emitted before
// channel close on a successful stream. spec §1.1
type StreamFinish struct {
	FinishReason     FinishReason
	Usage            Usage
	ProviderMetadata ProviderMetadata
}

func (StreamFinish) isStreamPart() {}

// StreamError is the terminal error part. Exactly one StreamError is emitted
// before channel close when the stream fails. spec §1.1
type StreamError struct{ Err error }

func (StreamError) isStreamPart() {}

// StreamRaw is a passthrough part for provider frames the adapter does not
// otherwise map. Emitted only when the adapter chooses to forward raw chunks.
// spec §1.1
type StreamRaw struct{ Bytes []byte }

func (StreamRaw) isStreamPart() {}

// StreamMetadata carries request and response metadata emitted during a stream.
type StreamMetadata struct {
	Request  RequestMetadata
	Response ResponseMetadata
}

func (StreamMetadata) isStreamPart() {}

// StreamWarning carries a non-fatal advisory emitted during a stream.
type StreamWarning struct {
	Warning Warning
}

func (StreamWarning) isStreamPart() {}

// GenerateOptions contains the prompt, tools, and limits for one generation call.
type GenerateOptions struct {
	// ModelID optionally overrides the client default model for this request.
	// It is provider-local, e.g. "gpt-4o", "claude-haiku-4-5", or
	// "llama3.1:8b". Empty uses the client default. It is not a registry
	// selector like "openai:gpt-4o".
	ModelID string

	System    string
	Messages  []Message
	Tools     []Tool
	MaxTokens int

	Temperature *float64
	TopP        *float64
	TopK        *int
	Stop        []string
	Seed        *int

	ToolChoice     ToolChoice
	ResponseFormat *ResponseFormat
	Reasoning      *ReasoningOptions

	Headers         map[string]string
	Metadata        map[string]string
	ProviderOptions ProviderOptions

	UnsupportedFeaturePolicy UnsupportedFeaturePolicy
}

// ReasoningOptions controls provider-neutral reasoning/thinking behavior for a
// single generation call. A nil Reasoning field means "use provider defaults";
// ReasoningNone explicitly disables reasoning when the provider supports it.
type ReasoningOptions struct {
	Effort       ReasoningEffort
	BudgetTokens *int
}

// ReasoningEffort is a provider-neutral reasoning effort level.
type ReasoningEffort string

const (
	ReasoningNone    ReasoningEffort = "none"
	ReasoningMinimal ReasoningEffort = "minimal"
	ReasoningLow     ReasoningEffort = "low"
	ReasoningMedium  ReasoningEffort = "medium"
	ReasoningHigh    ReasoningEffort = "high"
	ReasoningXHigh   ReasoningEffort = "xhigh"
)

// ToolChoice describes the tool-selection constraint sent to the provider.
// A zero-value ToolChoice (empty Type) means "provider default".
type ToolChoice struct {
	Type     ToolChoiceType
	ToolName string // only used when Type == ToolChoiceTool
}

// ToolChoiceType selects how tool use is constrained.
type ToolChoiceType string

const (
	ToolChoiceAuto     ToolChoiceType = "auto"
	ToolChoiceNone     ToolChoiceType = "none"
	ToolChoiceRequired ToolChoiceType = "required"
	ToolChoiceTool     ToolChoiceType = "tool"
)

// ResponseFormat describes the desired output format for structured output.
type ResponseFormat struct {
	Type       ResponseFormatType
	Name       string
	JSONSchema json.RawMessage
	Strict     *bool
}

// ResponseFormatType selects the response format mode.
type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// ProviderOptions is a namespaced escape hatch for provider-specific options.
// The outer key is the provider name (e.g. "anthropic", "openai").
type ProviderOptions map[string]map[string]any

// UnsupportedFeaturePolicy controls whether unsupported fields return an error
// or emit a warning and continue. The zero value behaves as "error".
type UnsupportedFeaturePolicy string

const (
	UnsupportedFeaturePolicyError UnsupportedFeaturePolicy = "error"
	UnsupportedFeaturePolicyWarn  UnsupportedFeaturePolicy = "warn"
)

// UnsupportedFeatureError is the typed error returned when a caller supplies a
// field that the target provider does not support and policy is "error".
type UnsupportedFeatureError struct {
	Provider string
	Feature  string
	Message  string
}

func (e *UnsupportedFeatureError) Error() string {
	return fmt.Sprintf("llm: provider %q does not support %q: %s", e.Provider, e.Feature, e.Message)
}

// Warning is a non-fatal advisory returned alongside a GenerateResult or
// emitted as a StreamWarning part.
type Warning struct {
	Code     string
	Message  string
	Provider string
}

// Message is a role-tagged item in the LLM conversation history.
type Message struct {
	Role    string
	Content []Content
}

// Content is one typed part of a message.
type Content interface{ isContent() }

// TextContent is plain text in a user or assistant message.
type TextContent struct {
	Text string
}

// ToolUseContent is an assistant request to invoke a tool.
type ToolUseContent struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResultContent is the user-side response to a tool invocation.
type ToolResultContent struct {
	ToolUseID string
	Text      string
	IsError   bool
}

// ReasoningContent carries assistant reasoning/thinking text. spec §1.2
type ReasoningContent struct{ Text string }

// ImageContent carries an image in a user or assistant message, sourced from
// either a URL or inline bytes. spec §1.2
type ImageContent struct {
	Source ImageSource
	MIME   string
}

// ImageSource is a tagged union of image source locations. spec §1.2
type ImageSource interface{ isImageSource() }

// ImageURLSource references an image by URL. spec §1.2
type ImageURLSource struct{ URL string }

func (ImageURLSource) isImageSource() {}

// ImageInlineSource carries raw image bytes. spec §1.2
type ImageInlineSource struct{ Data []byte }

func (ImageInlineSource) isImageSource() {}

func (TextContent) isContent()       {}
func (ToolUseContent) isContent()    {}
func (ToolResultContent) isContent() {}
func (ReasoningContent) isContent()  {}
func (ImageContent) isContent()      {}

// Tool describes an LLM-callable tool and its JSON input schema.
type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage

	Strict          *bool
	OutputSchema    json.RawMessage
	Type            ToolType
	ProviderOptions ProviderOptions
}

// ToolType selects whether a tool is a standard function or a provider-native tool.
type ToolType string

const (
	ToolTypeFunction ToolType = "function"
	ToolTypeProvider ToolType = "provider"
)

// GenerateResult is the normalized response returned by a Client.
type GenerateResult struct {
	Content          []Content
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	Request          RequestMetadata
	Response         ResponseMetadata
	ProviderMetadata ProviderMetadata
}

// FinishReason carries both the unified finish-reason and the raw provider value.
type FinishReason struct {
	Unified FinishReasonType
	Raw     string
}

// FinishReasonType is the normalized finish-reason vocabulary.
type FinishReasonType string

const (
	// FinishReasonStop means the model completed normally.
	FinishReasonStop FinishReasonType = "stop"
	// FinishReasonToolCalls means the model emitted one or more tool calls.
	FinishReasonToolCalls FinishReasonType = "tool-calls"
	// FinishReasonLength means generation stopped at a token limit.
	FinishReasonLength FinishReasonType = "length"
	// FinishReasonContentFilter means the provider's content filter stopped generation.
	FinishReasonContentFilter FinishReasonType = "content-filter"
	// FinishReasonError means the provider reported an error finish state.
	FinishReasonError FinishReasonType = "error"
	// FinishReasonOther means the provider reported an unrecognized finish reason.
	FinishReasonOther FinishReasonType = "other"
)

// Usage records token counts reported by the provider.
type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int

	ReasoningTokens          int
	CachedInputTokens        int
	CacheCreationInputTokens int
	ToolUseTokens            int
	ImageTokens              int

	ProviderMetadata ProviderMetadata
}

// RequestMetadata carries metadata about the outbound request.
type RequestMetadata struct {
	Headers map[string]string
}

// ResponseMetadata carries metadata about the provider HTTP response.
type ResponseMetadata struct {
	ID         string
	ModelID    string
	StatusCode int
	Headers    map[string]string
	Timestamp  time.Time
}

// ProviderMetadata is a namespaced map of provider-specific metadata.
// The outer key is the provider name (e.g. "anthropic", "openai").
type ProviderMetadata map[string]map[string]any

// Capabilities describes which features a provider adapter supports.
type Capabilities struct {
	Provider string
	ModelID  string // empty means "all models for this provider"

	Streaming          bool
	ToolCalling        bool
	ToolChoiceAuto     bool
	ToolChoiceNone     bool
	ToolChoiceRequired bool
	ToolChoiceTool     bool
	StructuredOutput   bool
	JSONMode           bool
	Images             bool
	Reasoning          bool
	ParallelToolCalls  bool
	PromptCaching      bool
}

// CapabilitiesProvider is an optional interface implemented by adapters
// that can report their supported feature set.
type CapabilitiesProvider interface {
	Capabilities() Capabilities
}

// ModelCapabilities is an optional interface implemented by adapters that can
// report best-known capabilities for a specific provider-local model ID.
type ModelCapabilities interface {
	CapabilitiesForModel(modelID string) Capabilities
}

// Identity identifies a client's effective provider-local model endpoint.
type Identity struct {
	Provider string
	ModelID  string
}

// IdentifiedClient is implemented by clients that can report their default
// provider/model identity.
type IdentifiedClient interface {
	Client
	Identity() Identity
}

// RequestIdentityProvider is implemented by clients and wrappers that can
// report the effective provider/model identity for a request.
type RequestIdentityProvider interface {
	RequestIdentity(opts GenerateOptions) (Identity, error)
}

// Middleware wraps a Client to add cross-cutting behavior.
// Middlewares compose left-to-right: the first middleware in a chain
// is the outermost wrapper.
type Middleware func(Client) Client

// Chain applies middlewares in order, outermost first.
func Chain(client Client, middlewares ...Middleware) Client {
	for i := len(middlewares) - 1; i >= 0; i-- {
		client = middlewares[i](client)
	}
	return client
}

// FailoverConfig controls when and how WithFailover switches clients.
type FailoverConfig struct {
	// ShouldFailover classifies an error as eligible for failover. If nil,
	// no error triggers failover and the original error is returned.
	ShouldFailover func(ctx context.Context, err error) bool
	// GetNext returns the client to try for the given attempt (1-based), or
	// nil to stop and return the last error.
	GetNext func(attempt int) Client
	// MaxAttempts caps the number of failover hops. Zero means unlimited
	// (bounded only by GetNext returning nil).
	MaxAttempts int
	// RetryDelay is an optional fixed delay between failover attempts.
	// Zero means no delay.
	RetryDelay time.Duration
	// RewriteOptions can map provider-local request options before each attempt.
	// The attempt is zero for the primary client and one-based for failover hops.
	RewriteOptions func(attempt int, opts GenerateOptions) GenerateOptions
}

// WithFailover wraps a client with failover-on-provider-failure behavior.
//
// When Generate or Stream returns an error classified by ShouldFailover, the
// next client from GetNext is tried. Failover never retries the same client;
// each hop uses a distinct client supplied by GetNext.
//
// Stream semantics: failover applies only to errors returned synchronously
// from Stream before any part is emitted. In-stream errors (StreamError
// parts) are forwarded unchanged and never trigger failover. Callers that
// need in-stream failover must implement it at the application layer.
func WithFailover(client Client, cfg FailoverConfig) Client {
	return failoverClient{client: client, cfg: cfg}
}

type failoverClient struct {
	client Client
	cfg    FailoverConfig
}

func (c failoverClient) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	client := c.client
	attempt := 0
	var lastErr error
	for client != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		attemptOpts := opts
		if c.cfg.RewriteOptions != nil {
			attemptOpts = c.cfg.RewriteOptions(attempt, opts)
		}
		result, err := client.Generate(ctx, attemptOpts)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if c.cfg.ShouldFailover == nil || !c.cfg.ShouldFailover(ctx, err) {
			return nil, err
		}
		attempt++
		if c.cfg.MaxAttempts > 0 && attempt > c.cfg.MaxAttempts {
			return nil, lastErr
		}
		if c.cfg.GetNext == nil {
			return nil, lastErr
		}
		if c.cfg.RetryDelay > 0 {
			if err := sleepCtx(ctx, c.cfg.RetryDelay); err != nil {
				return nil, err
			}
		}
		client = c.cfg.GetNext(attempt)
	}
	return nil, lastErr
}

// Stream forwards to the wrapped client. See WithFailover for stream
// failover semantics.
func (c failoverClient) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	client := c.client
	attempt := 0
	var lastErr error
	for client != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		attemptOpts := opts
		if c.cfg.RewriteOptions != nil {
			attemptOpts = c.cfg.RewriteOptions(attempt, opts)
		}
		ch, err := client.Stream(ctx, attemptOpts)
		if err == nil {
			return ch, nil
		}
		lastErr = err
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if c.cfg.ShouldFailover == nil || !c.cfg.ShouldFailover(ctx, err) {
			return nil, err
		}
		attempt++
		if c.cfg.MaxAttempts > 0 && attempt > c.cfg.MaxAttempts {
			return nil, lastErr
		}
		if c.cfg.GetNext == nil {
			return nil, lastErr
		}
		if c.cfg.RetryDelay > 0 {
			if err := sleepCtx(ctx, c.cfg.RetryDelay); err != nil {
				return nil, err
			}
		}
		client = c.cfg.GetNext(attempt)
	}
	return nil, lastErr
}

func (c failoverClient) RequestIdentity(opts GenerateOptions) (Identity, error) {
	attemptOpts := opts
	if c.cfg.RewriteOptions != nil {
		attemptOpts = c.cfg.RewriteOptions(0, opts)
	}
	return clientIdentity(c.client, attemptOpts)
}

// sleepCtx sleeps for d or until ctx is cancelled, whichever comes first.
// It returns ctx.Err() if the context is cancelled.
func sleepCtx(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// CacheBackend stores GenerateResult values by deterministic request key.
type CacheBackend interface {
	Get(ctx context.Context, key string) (*GenerateResult, bool)
	Set(ctx context.Context, key string, result *GenerateResult)
}

// WithCache wraps a client with read-through response caching.
func WithCache(client Client, backend CacheBackend) Client {
	if backend == nil {
		panic("llm: nil cache backend")
	}
	return cacheClient{client: client, backend: backend}
}

type cacheClient struct {
	client  Client
	backend CacheBackend
}

func (c cacheClient) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	identity, err := clientIdentity(c.client, opts)
	if err != nil {
		return nil, err
	}
	key, err := cacheKey(identity, opts)
	if err != nil {
		return nil, err
	}
	if result, ok := c.backend.Get(ctx, key); ok {
		return result, nil
	}
	result, err := c.client.Generate(ctx, opts)
	if err != nil {
		return nil, err
	}
	if result != nil {
		c.backend.Set(ctx, key, result)
	}
	return result, nil
}

// Stream is not cached. The cache wrapper only caches Generate results.
// Callers that need stream caching must implement it at the application layer.
func (c cacheClient) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	return c.client.Stream(ctx, opts)
}

func (c cacheClient) Identity() Identity {
	if p, ok := c.client.(IdentifiedClient); ok {
		return p.Identity()
	}
	return Identity{}
}

func (c cacheClient) RequestIdentity(opts GenerateOptions) (Identity, error) {
	return clientIdentity(c.client, opts)
}

func clientIdentity(client Client, opts GenerateOptions) (Identity, error) {
	if err := ValidateModelID(opts.ModelID); err != nil {
		return Identity{}, err
	}
	if p, ok := client.(RequestIdentityProvider); ok {
		return p.RequestIdentity(opts)
	}
	if p, ok := client.(IdentifiedClient); ok {
		return p.Identity(), nil
	}
	return Identity{}, nil
}

// cacheKey produces a deterministic SHA-256 hash of GenerateOptions.
// Maps in GenerateOptions (Headers, Metadata, ProviderOptions) are sorted
// before serialization to ensure a stable key regardless of insertion order.
func cacheKey(identity Identity, opts GenerateOptions) (string, error) {
	// Use a canonical intermediate representation to handle maps deterministically.
	type canonicalProviderOptions struct {
		Provider string            `json:"provider"`
		Keys     []string          `json:"keys"`
		Values   map[string]string `json:"values"`
	}
	type canonicalOpts struct {
		Provider        string                     `json:"provider,omitempty"`
		ModelID         string                     `json:"model_id,omitempty"`
		System          string                     `json:"system,omitempty"`
		Messages        any                        `json:"messages,omitempty"`
		Tools           any                        `json:"tools,omitempty"`
		MaxTokens       int                        `json:"max_tokens,omitempty"`
		Temperature     *float64                   `json:"temperature,omitempty"`
		TopP            *float64                   `json:"top_p,omitempty"`
		TopK            *int                       `json:"top_k,omitempty"`
		Stop            []string                   `json:"stop,omitempty"`
		Seed            *int                       `json:"seed,omitempty"`
		ToolChoice      any                        `json:"tool_choice,omitempty"`
		ResponseFormat  any                        `json:"response_format,omitempty"`
		Reasoning       *ReasoningOptions          `json:"reasoning,omitempty"`
		ProviderOptions []canonicalProviderOptions `json:"provider_options,omitempty"`
		// Sorted header and metadata pairs
		HeaderKeys   []string `json:"header_keys,omitempty"`
		MetadataKeys []string `json:"metadata_keys,omitempty"`
	}

	co := canonicalOpts{
		Provider:       identity.Provider,
		ModelID:        identity.ModelID,
		System:         opts.System,
		Messages:       opts.Messages,
		Tools:          opts.Tools,
		MaxTokens:      opts.MaxTokens,
		Temperature:    opts.Temperature,
		TopP:           opts.TopP,
		TopK:           opts.TopK,
		Stop:           opts.Stop,
		Seed:           opts.Seed,
		ToolChoice:     opts.ToolChoice,
		ResponseFormat: opts.ResponseFormat,
		Reasoning:      opts.Reasoning,
	}

	// Sort Headers keys for stable output.
	if len(opts.Headers) > 0 {
		keys := make([]string, 0, len(opts.Headers))
		for k := range opts.Headers {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			co.HeaderKeys = append(co.HeaderKeys, k+"="+opts.Headers[k])
		}
	}
	// Sort Metadata keys.
	if len(opts.Metadata) > 0 {
		keys := make([]string, 0, len(opts.Metadata))
		for k := range opts.Metadata {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			co.MetadataKeys = append(co.MetadataKeys, k+"="+opts.Metadata[k])
		}
	}
	if len(opts.ProviderOptions) > 0 {
		providers := make([]string, 0, len(opts.ProviderOptions))
		for provider := range opts.ProviderOptions {
			providers = append(providers, provider)
		}
		sort.Strings(providers)
		for _, provider := range providers {
			values := opts.ProviderOptions[provider]
			entry := canonicalProviderOptions{Provider: provider, Values: map[string]string{}}
			for key, value := range values {
				entry.Keys = append(entry.Keys, key)
				encoded, err := json.Marshal(value)
				if err != nil {
					return "", err
				}
				entry.Values[key] = string(encoded)
			}
			sort.Strings(entry.Keys)
			co.ProviderOptions = append(co.ProviderOptions, entry)
		}
	}

	data, err := json.Marshal(co)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// ValidateModelID validates provider-neutral model ID syntax. It does not check
// provider-specific model existence.
func ValidateModelID(modelID string) error {
	if modelID == "" {
		return nil
	}
	if strings.TrimSpace(modelID) != modelID {
		return fmt.Errorf("model_id: ModelID must not contain leading or trailing whitespace")
	}
	return nil
}
