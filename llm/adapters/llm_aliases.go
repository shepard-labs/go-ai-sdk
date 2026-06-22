package adapters

import "github.com/shepard-labs/go-ai-sdk/llm"

type Client = llm.Client
type Capabilities = llm.Capabilities
type Content = llm.Content
type GenerateOptions = llm.GenerateOptions
type GenerateResult = llm.GenerateResult
type Identity = llm.Identity
type FinishReason = llm.FinishReason
type FinishReasonType = llm.FinishReasonType
type ImageContent = llm.ImageContent
type ImageInlineSource = llm.ImageInlineSource
type ImageURLSource = llm.ImageURLSource
type Message = llm.Message
type ProviderMetadata = llm.ProviderMetadata
type ReasoningContent = llm.ReasoningContent
type RequestMetadata = llm.RequestMetadata
type ResponseMetadata = llm.ResponseMetadata
type StreamError = llm.StreamError
type StreamFinish = llm.StreamFinish
type StreamMetadata = llm.StreamMetadata
type StreamPart = llm.StreamPart
type StreamRaw = llm.StreamRaw
type StreamReasoningDelta = llm.StreamReasoningDelta
type StreamReasoningEnd = llm.StreamReasoningEnd
type StreamReasoningStart = llm.StreamReasoningStart
type StreamTextDelta = llm.StreamTextDelta
type StreamTextEnd = llm.StreamTextEnd
type StreamTextStart = llm.StreamTextStart
type StreamWarning = llm.StreamWarning
type StreamToolCallStart = llm.StreamToolCallStart
type StreamToolInputDelta = llm.StreamToolInputDelta
type StreamToolInputEnd = llm.StreamToolInputEnd
type TextContent = llm.TextContent
type Tool = llm.Tool
type ToolChoice = llm.ToolChoice
type ResponseFormat = llm.ResponseFormat
type ToolResultContent = llm.ToolResultContent
type ToolUseContent = llm.ToolUseContent
type Usage = llm.Usage
type Warning = llm.Warning

// FinishReasonType constants re-exported for use within this package.
const (
	FinishReasonError         = llm.FinishReasonError
	FinishReasonLength        = llm.FinishReasonLength
	FinishReasonStop          = llm.FinishReasonStop
	FinishReasonToolCalls     = llm.FinishReasonToolCalls
	FinishReasonContentFilter = llm.FinishReasonContentFilter
	FinishReasonOther         = llm.FinishReasonOther
)

var ErrStreamNotImplemented = llm.ErrStreamNotImplemented
