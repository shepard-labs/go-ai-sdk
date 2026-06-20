package adapters

import "github.com/shepard-labs/go-ai-sdk/llm"

type Client = llm.Client
type Content = llm.Content
type GenerateOptions = llm.GenerateOptions
type GenerateResult = llm.GenerateResult
type FinishReason = llm.FinishReason
type ImageContent = llm.ImageContent
type ImageInlineSource = llm.ImageInlineSource
type ImageURLSource = llm.ImageURLSource
type Message = llm.Message
type ReasoningContent = llm.ReasoningContent
type StreamError = llm.StreamError
type StreamFinish = llm.StreamFinish
type StreamPart = llm.StreamPart
type StreamRaw = llm.StreamRaw
type StreamReasoningDelta = llm.StreamReasoningDelta
type StreamReasoningEnd = llm.StreamReasoningEnd
type StreamReasoningStart = llm.StreamReasoningStart
type StreamTextDelta = llm.StreamTextDelta
type StreamTextEnd = llm.StreamTextEnd
type StreamTextStart = llm.StreamTextStart
type StreamToolCallStart = llm.StreamToolCallStart
type StreamToolInputDelta = llm.StreamToolInputDelta
type StreamToolInputEnd = llm.StreamToolInputEnd
type TextContent = llm.TextContent
type Tool = llm.Tool
type ToolResultContent = llm.ToolResultContent
type ToolUseContent = llm.ToolUseContent
type Usage = llm.Usage

const (
	FinishReasonError     = llm.FinishReasonError
	FinishReasonLength    = llm.FinishReasonLength
	FinishReasonStop      = llm.FinishReasonStop
	FinishReasonToolCalls = llm.FinishReasonToolCalls
)

var ErrStreamNotImplemented = llm.ErrStreamNotImplemented
