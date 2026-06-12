package openai

import (
	"encoding/json"
	"net/http"
	"regexp"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// Message is a prompt message union marker.
type Message = openaicompatible.Message

// UserContent is a user content union marker.
type UserContent = openaicompatible.UserContent

// AssistantContent is an assistant content union marker.
type AssistantContent = openaicompatible.AssistantContent

// ToolContent is a tool content union marker.
type ToolContent = openaicompatible.ToolContent

// Content is a generated content union marker.
type Content = openaicompatible.Content

// StreamPart is a stream event union marker.
type StreamPart = openaicompatible.StreamPart

// Delta is a stream delta union marker.
type Delta = openaicompatible.Delta

// SystemMessage is a system-role prompt message.
type SystemMessage = openaicompatible.SystemMessage

// UserMessage is a user-role prompt message carrying one or more content parts.
type UserMessage = openaicompatible.UserMessage

// AssistantMessage is an assistant-role prompt message.
type AssistantMessage = openaicompatible.AssistantMessage

// ToolMessage carries tool-call result content for the tool role.
type ToolMessage = openaicompatible.ToolMessage

// TextContent is plain text in user, assistant, or response messages.
type TextContent = openaicompatible.TextContent

// FileContent is a file part in a user message.
type FileContent = openaicompatible.FileContent

// ReasoningContent carries model reasoning text and optional encrypted state.
type ReasoningContent struct {
	Text             string
	ProviderOptions  ProviderMetadata
	ItemID           string
	EncryptedContent string
	Summary          []string
}

func (ReasoningContent) IsAssistantContent() {}
func (ReasoningContent) IsContent()          {}

// openaiCompatReasoningContent is the embedded base for chat models that
// only need text.
type openaiCompatReasoningContent = openaicompatible.ReasoningContent

// ToolCallContent is an assistant tool invocation.
type ToolCallContent struct {
	ToolCallContentEmbed
	// ProviderExecuted indicates the call was executed by OpenAI (e.g.
	// web_search, code_interpreter, file_search, image_generation, MCP).
	ProviderExecuted bool
	// Dynamic indicates the tool is a dynamic remote tool (MCP).
	Dynamic bool
}

// ToolCallContentEmbed is the field set shared with the openaicompatible
// ToolCallContent type. It is used to build the openai.ToolCallContent
// struct without an import cycle.
type ToolCallContentEmbed struct {
	ToolCallID       string
	ToolName         string
	Input            json.RawMessage
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderMetadata
}

// IsAssistantContent lets the openai ToolCallContent satisfy the
// openaicompatible assistant-content union.
func (ToolCallContent) IsAssistantContent() {}

// IsContent lets the openai ToolCallContent satisfy the openaicompatible
// content union.
func (ToolCallContent) IsContent() {}

// ToolResultContent is a tool role result. It wraps the openaicompatible
// type to expose a ToolName field for the Responses API output.
type ToolResultContent struct {
	openaicompatible.ToolResultContent
	ToolName         string
	ProviderMetadata ProviderMetadata
}

func (ToolResultContent) IsContent()    {}
func (ToolResultContent) IsToolContent() {}

// ToolResultOutput is the typed payload of a tool result.
type ToolResultOutput = openaicompatible.ToolResultOutput

// CustomContent is an openai-package custom content part with arbitrary data.
type CustomContent struct {
	Kind string
	Data any
}

func (CustomContent) IsUserContent()      {}
func (CustomContent) IsAssistantContent() {}
func (CustomContent) IsContent()          {}

// CompactionContent is a Responses API compaction item payload.
type CompactionContent struct {
	ItemID           string
	EncryptedContent string
}

func (CompactionContent) IsContent() {}

// ToolApprovalResponse is the SDK's representation of a tool-approval
// response (e.g. an MCP approval). It maps to /v1/responses input items of
// type mcp_approval_response.
type ToolApprovalResponse struct {
	ApprovalID string
	Approve    bool
	Reason     string
}

func (ToolApprovalResponse) IsAssistantContent() {}

// Tool is the openai tool definition. Function tools map directly to OpenAI's
// function tool format; provider tools carry an ID and Args payload that the
// Responses model serializes at request-build time.
type Tool struct {
	Type        string
	ID          string
	Name        string
	Description string
	InputSchema any
	Strict      *bool
	Args        any
	// ProviderOptions is the unparsed provider-options map for advanced
	// fields like `defer_loading` or `namespace`.
	ProviderOptions map[string]any
}

// ToolChoice describes the model tool choice. Type is one of "auto", "none",
// "required", "tool". When Type == "tool", ToolName selects the function.
type ToolChoice struct {
	Type     string
	ToolName string
}

// ResponseFormat is the AI-SDK-style response format. When Type is "json"
// the openai package emits a json_schema (with structured outputs) or
// json_object payload on the request.
type ResponseFormat = openaicompatible.ResponseFormat

// StructuredOutput is the AI-SDK-style structured output descriptor.
type StructuredOutput = openaicompatible.StructuredOutput

// RequestMetadata is the captured request body and form fields.
type RequestMetadata = openaicompatible.RequestMetadata

// ResponseMetadata is the response id, model id, timestamp, headers, and body.
type ResponseMetadata = openaicompatible.ResponseMetadata

// StreamResponse is metadata about the response a stream was generated for.
type StreamResponse = openaicompatible.StreamResponse

// ImageResponseMetadata is the image-generation response metadata.
type ImageResponseMetadata = openaicompatible.ImageResponseMetadata

// ImageFile describes an input image for image edit requests.
type ImageFile = openaicompatible.ImageFile

// ProviderMetadata is the AI-SDK provider metadata map.
type ProviderMetadata = openaicompatible.ProviderMetadata

// ProviderOptions is the AI-SDK provider options map.
type ProviderOptions = openaicompatible.ProviderOptions

// Warning describes a non-fatal issue encountered while building or parsing a
// request or response.
type Warning = openaicompatible.Warning

// FinishReason is the unified/raw finish reason pair.
type FinishReason = openaicompatible.FinishReason

// TokenCounts is the input-side token count breakdown.
type TokenCounts = openaicompatible.TokenCounts

// OutputTokenCounts is the output-side token count breakdown.
type OutputTokenCounts = openaicompatible.OutputTokenCounts

// Usage is the per-call token usage.
type Usage = openaicompatible.Usage

// EmbeddingUsage is the embedding token usage.
type EmbeddingUsage = openaicompatible.EmbeddingUsage

// StreamStart marks the start of a stream and carries any pre-flight warnings.
type StreamStart = openaicompatible.StreamStart

// StreamResponseMetadata carries the response id, model id, and timestamp
// from the first chunk.
type StreamResponseMetadata = openaicompatible.StreamResponseMetadata

// StreamTextStart marks the start of a text part.
type StreamTextStart struct {
	ID              string
	ProviderMetadata ProviderMetadata
}

func (StreamTextStart) IsStreamPart() {}

// StreamTextDelta is a text delta.
type StreamTextDelta struct {
	ID              string
	Text            string
	ProviderMetadata ProviderMetadata
}

func (StreamTextDelta) IsStreamPart() {}
func (StreamTextDelta) IsDelta()      {}

// StreamTextEnd marks the end of a text part.
type StreamTextEnd struct {
	ID              string
	ProviderMetadata ProviderMetadata
}

func (StreamTextEnd) IsStreamPart() {}

// StreamReasoningStart marks the start of a reasoning part.
type StreamReasoningStart struct {
	ID              string
	ProviderMetadata ProviderMetadata
}

func (StreamReasoningStart) IsStreamPart() {}

// StreamReasoningDelta is a reasoning delta.
type StreamReasoningDelta struct {
	ID              string
	Text            string
	ProviderMetadata ProviderMetadata
}

func (StreamReasoningDelta) IsStreamPart() {}
func (StreamReasoningDelta) IsDelta()      {}

// StreamReasoningEnd marks the end of a reasoning part.
type StreamReasoningEnd struct {
	ID              string
	ProviderMetadata ProviderMetadata
}

func (StreamReasoningEnd) IsStreamPart() {}

// StreamToolInputStart marks the start of a tool-call input part.
type StreamToolInputStart struct {
	ID              string
	ToolName        string
	ProviderMetadata ProviderMetadata
}

func (StreamToolInputStart) IsStreamPart() {}

// StreamToolInputDelta is a tool-call input delta.
type StreamToolInputDelta struct {
	ID              string
	Delta           string
	ProviderMetadata ProviderMetadata
}

func (StreamToolInputDelta) IsStreamPart() {}
func (StreamToolInputDelta) IsDelta()      {}

// StreamToolInputEnd marks the end of a tool-call input part.
type StreamToolInputEnd struct {
	ID              string
	ProviderMetadata ProviderMetadata
}

func (StreamToolInputEnd) IsStreamPart() {}

// StreamToolCall is a fully-formed tool call (input has closed).
type StreamToolCall struct {
	ToolCallContent
}

func (StreamToolCall) IsStreamPart() {}

// StreamFinish marks the end of the stream and carries finish reason, usage,
// and provider metadata.
type StreamFinish = openaicompatible.StreamFinish

// StreamError reports a stream-level error.
type StreamError = openaicompatible.StreamError

// StreamRaw is the raw SSE chunk and its decoded map; only emitted when
// IncludeRawChunks is set.
type StreamRaw = openaicompatible.StreamRaw

// StreamCustomPart is a Responses-specific custom stream part (e.g. compaction).
type StreamCustomPart struct {
	Kind string
	Data any
}

func (StreamCustomPart) IsStreamPart() {}

// StreamToolApprovalRequest is emitted alongside an MCP tool call when the
// model requests approval from the user.
type StreamToolApprovalRequest struct {
	ApprovalID string
	ToolCallID string
}

func (StreamToolApprovalRequest) IsStreamPart() {}

// StreamCompactionEnd marks the end of a compaction output item.
type StreamCompactionEnd struct {
	ItemID           string
	EncryptedContent string
}

func (StreamCompactionEnd) IsStreamPart() {}

// SourceContent is a source/annotation part attached to a generated text
// (web citation, file citation, etc.).
type SourceContent struct {
	SourceType string
	ID         string
	URL        string
	Title      string
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderMetadata
}

func (SourceContent) IsContent()    {}
func (SourceContent) IsStreamPart() {}

// GenerateOptions carries the parameters for a non-streaming chat,
// completion, or responses generate call.
type GenerateOptions = openaicompatible.GenerateOptions

// StreamOptions extends [GenerateOptions] with streaming-specific parameters.
type StreamOptions = openaicompatible.StreamOptions

// EmbedOptions carries parameters for an embedding call.
type EmbedOptions = openaicompatible.EmbedOptions

// ImageGenerateOptions carries parameters for an image generation or edit call.
type ImageGenerateOptions = openaicompatible.ImageGenerateOptions

// Speech / Transcription / Files / Skills / Realtime payload types

// SpeechGenerateOptions carries parameters for a speech synthesis call.
type SpeechGenerateOptions struct {
	Text           string
	Voice          string
	OutputFormat   string
	Speed          *float64
	Instructions   *string
	Headers        http.Header
	ProviderOptions ProviderOptions
}

// SpeechGenerateResult is the result of a speech synthesis call.
type SpeechGenerateResult struct {
	Audio    []byte
	Warnings []Warning
	Request  RequestMetadata
	Response SpeechResponseMetadata
}

// SpeechResponseMetadata is the speech response metadata. The audio body is
// mirrored into Body for parity with [ResponseMetadata].
type SpeechResponseMetadata struct {
	Timestamp time.Time
	ModelID   string
	Headers   http.Header
	Body      []byte
}

// TranscriptionOptions carries parameters for a transcription call.
type TranscriptionOptions struct {
	Audio                 []byte
	MediaType             string
	Filename              string
	Language              *string
	Prompt                *string
	Temperature           *float64
	TimestampGranularities []string
	Include               []string
	Headers               http.Header
	ProviderOptions       ProviderOptions
}

// TranscriptionResult is the result of a transcription call.
type TranscriptionResult struct {
	Text             string
	Segments         []TranscriptionSegment
	Language         string
	Duration         *float64
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// TranscriptionSegment is a per-segment or per-word transcript slice.
type TranscriptionSegment struct {
	Text        string
	StartSecond float64
	EndSecond   float64
}

// ProviderReference maps provider names to opaque server-side identifiers.
type ProviderReference map[string]string

// FilesUploadOptions carries parameters for a file upload.
type FilesUploadOptions struct {
	Data            []byte
	MediaType       string
	Filename        string
	ProviderOptions ProviderOptions
}

// FilesUploadResult is the result of a file upload.
type FilesUploadResult struct {
	ProviderReference ProviderReference
	MediaType         string
	Filename          string
	ProviderMetadata  ProviderMetadata
	Warnings          []Warning
}

// SkillsFile is a single zip file to upload as part of a skills request.
type SkillsFile struct {
	Path      string
	Data      []byte
	MediaType string
}

// SkillsUploadOptions carries parameters for a skill upload.
type SkillsUploadOptions struct {
	Files        []SkillsFile
	DisplayTitle string
	Headers      http.Header
}

// SkillsUploadResult is the result of a skill upload.
type SkillsUploadResult struct {
	ProviderReference ProviderReference
	Name              string
	Description       string
	LatestVersion     string
	ProviderMetadata  ProviderMetadata
	Warnings          []Warning
}

// ClientSecretOptions carries parameters for creating a Realtime client secret.
type ClientSecretOptions struct {
	ExpiresAfterSeconds *int
	SessionConfig       *SessionConfig
}

// ClientSecretResult is the result of creating a Realtime client secret.
type ClientSecretResult struct {
	Token     string
	URL       string
	ExpiresAt *int64
}

// WebSocketConfigInput is the input to GetWebSocketConfig.
type WebSocketConfigInput struct {
	Token string
	URL   string
}

// WebSocketConfig is the connection configuration a caller uses to open a
// WebSocket to the OpenAI Realtime API.
type WebSocketConfig struct {
	URL       string
	Protocols []string
}

// SessionConfig is the typed Realtime session configuration.
type SessionConfig struct {
	Instructions             string
	Voice                    string
	OutputModalities         []string
	InputAudioFormat         *AudioFormat
	InputAudioTranscription  *InputAudioTranscription
	OutputAudioFormat        *AudioFormat
	OutputAudioTranscription *OutputAudioTranscription
	TurnDetection            *TurnDetection
	Tools                    []RealtimeToolDefinition
	ProviderOptions          ProviderOptions
}

// AudioFormat is the audio format for Realtime input/output.
type AudioFormat struct {
	Type string
	Rate *int
}

// InputAudioTranscription configures input audio transcription.
type InputAudioTranscription struct {
	Model    *string
	Language *string
	Prompt   *string
}

// OutputAudioTranscription configures output audio transcription.
type OutputAudioTranscription struct {
	Model *string
}

// TurnDetection configures server-side turn detection.
type TurnDetection struct {
	Type              string
	Threshold         *float64
	SilenceDurationMs *int
	PrefixPaddingMs   *int
}

// RealtimeToolDefinition is a function tool definition for Realtime sessions.
type RealtimeToolDefinition struct {
	Type        string
	Name        string
	Description string
	Parameters  any
}

// RegexpPattern is a regular expression URL match (used by the underlying
// openaicompatible package).
type RegexpPattern = regexp.Regexp

// jsonRaw is a convenience for raw JSON values used internally.
type jsonRaw = json.RawMessage
