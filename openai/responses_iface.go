package openai

import (
	"context"
	"net/http"
)

// ResponsesModel is the openai Responses API model interface.
type ResponsesModel interface {
	ModelID() string
	Provider() string
	DoGenerate(ctx context.Context, opts ResponsesGenerateOptions) (*ResponsesGenerateResult, error)
	DoStream(ctx context.Context, opts ResponsesStreamOptions) (*ResponsesStreamResult, error)
}

// ResponsesGenerateOptions carries the parameters for a non-streaming
// Responses API call.
type ResponsesGenerateOptions struct {
	// Top-level.
	Instructions string

	// Conversation state.
	Conversation       *string
	PreviousResponseID *string
	Store              *bool

	// Sampling.
	MaxOutputTokens *int
	Temperature     *float64
	TopP            *float64
	Seed            *int
	StopSequences   []string

	// Reasoning.
	Reasoning      *ReasoningConfig
	ForceReasoning *bool

	// Standard prompt payload.
	Messages         []Message
	Tools            []Tool
	ToolChoice       *ToolChoice
	ResponseFormat   *ResponseFormat
	StructuredOutput *StructuredOutput

	// I/O.
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// ResponsesStreamOptions extends ResponsesGenerateOptions with
// IncludeRawChunks.
type ResponsesStreamOptions struct {
	ResponsesGenerateOptions
	IncludeRawChunks bool
}

// ResponsesGenerateResult is the result of a non-streaming Responses call.
type ResponsesGenerateResult struct {
	Content          []Content
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// ResponsesStreamResult is the result of a Responses streaming call.
type ResponsesStreamResult struct {
	Stream   <-chan StreamPart
	Parts    <-chan StreamPart
	Request  RequestMetadata
	Response *StreamResponse
}

// ReasoningConfig configures reasoning effort and summary behavior.
type ReasoningConfig struct {
	Effort  *string // "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
	Summary *string // "auto" | "detailed"
}
