// Package llm defines provider-neutral LLM requests, tool calls, and agent loops.
package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
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
	FinishReason FinishReason
	Usage        Usage
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

// GenerateOptions contains the prompt, tools, and limits for one generation call.
type GenerateOptions struct {
	System    string
	Messages  []Message
	Tools     []Tool
	MaxTokens int
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
}

// GenerateResult is the normalized response returned by a Client.
type GenerateResult struct {
	Content      []Content
	FinishReason FinishReason
	Usage        Usage
}

// FinishReason explains why a generation call stopped.
type FinishReason string

const (
	// FinishReasonStop means the model completed normally.
	FinishReasonStop FinishReason = "stop"
	// FinishReasonToolCalls means the model emitted one or more tool calls.
	FinishReasonToolCalls FinishReason = "tool-calls"
	// FinishReasonLength means generation stopped at a token limit.
	FinishReasonLength FinishReason = "length"
	// FinishReasonError means the provider reported an error finish state.
	FinishReasonError FinishReason = "error"
)

// Usage records token counts reported by the provider.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// FailoverConfig controls when and how WithFailover switches clients.
type FailoverConfig struct {
	ShouldFailover func(ctx context.Context, err error) bool
	GetNext        func(attempt int) Client
	MaxAttempts    int
}

// WithFailover wraps a client with retry-on-provider-failure behavior.
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
		result, err := client.Generate(ctx, opts)
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
		client = c.cfg.GetNext(attempt)
	}
	return nil, lastErr
}

// Stream forwards to the wrapped client. Failover applies only to errors
// returned synchronously from Stream itself (before any part is emitted);
// in-stream StreamError parts are forwarded unchanged. spec §1.1
func (c failoverClient) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	client := c.client
	attempt := 0
	var lastErr error
	for client != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ch, err := client.Stream(ctx, opts)
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
		client = c.cfg.GetNext(attempt)
	}
	return nil, lastErr
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
	key, err := cacheKey(opts)
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

// Stream is a no-op pass-through for streams. Cacheable streams are out of
// scope for the spike; the cache wrapper forwards unchanged. spec §1.1
func (c cacheClient) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	return c.client.Stream(ctx, opts)
}

func cacheKey(opts GenerateOptions) (string, error) {
	data, err := json.Marshal(opts)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
