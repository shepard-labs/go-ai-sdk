package openaicompatible

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"time"
)

// Version is the package version used in User-Agent headers.
const Version = "0.1.0"

// Provider creates OpenAI-compatible model families.
type Provider interface {
	Model(modelID string) LanguageModel
	LanguageModel(modelID string) LanguageModel
	ChatModel(modelID string) LanguageModel
	Chat(modelID string) LanguageModel
	CompletionModel(modelID string) LanguageModel
	Completion(modelID string) LanguageModel
	EmbeddingModel(modelID string) EmbeddingModel
	Embedding(modelID string) EmbeddingModel
	TextEmbeddingModel(modelID string) EmbeddingModel
	ImageModel(modelID string) ImageModel
	Image(modelID string) ImageModel
	Name() string
	Err() error
}

// LanguageModel is the package-local OpenAI-compatible language model interface.
type LanguageModel interface {
	ModelID() string
	Provider() string
	SupportURLs() map[string][]*regexp.Regexp
	DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error)
	DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error)
}

// EmbeddingModel is the package-local OpenAI-compatible embedding model interface.
type EmbeddingModel interface {
	ModelID() string
	Provider() string
	MaxEmbeddingsPerCall() int
	SupportsParallelCalls() bool
	DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error)
}

// ImageModel is the package-local OpenAI-compatible image model interface.
type ImageModel interface {
	ModelID() string
	Provider() string
	MaxImagesPerCall() int
	DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error)
}

// ChatLanguageModel is the chat model family interface.
type ChatLanguageModel = LanguageModel

// OpenAICompatibleChatLanguageModel is the concrete chat model type.
type OpenAICompatibleChatLanguageModel = openAICompatibleChatLanguageModel

// CompletionLanguageModel is the completion model family interface.
type CompletionLanguageModel = LanguageModel

// OpenAICompatibleCompletionLanguageModel is the concrete completion model type.
type OpenAICompatibleCompletionLanguageModel = openAICompatibleCompletionLanguageModel

// OpenAICompatibleEmbeddingModel is the concrete embedding model type.
type OpenAICompatibleEmbeddingModel = openAICompatibleEmbeddingModel

// OpenAICompatibleImageModel is the concrete image model type.
type OpenAICompatibleImageModel = openAICompatibleImageModel

// Message is a prompt message union marker.
type Message interface{ IsMessage() }

// UserContent is a user content union marker.
type UserContent interface{ IsUserContent() }

// AssistantContent is an assistant content union marker.
type AssistantContent interface{ IsAssistantContent() }

// ToolContent is a tool content union marker.
type ToolContent interface{ IsToolContent() }

// Content is a generated content union marker.
type Content interface{ IsContent() }

// StreamPart is a stream event union marker.
type StreamPart interface{ IsStreamPart() }

// Delta is a stream delta union marker.
type Delta interface{ IsDelta() }

// SystemMessage is a system-role prompt message.
type SystemMessage struct {
	Content         string
	ProviderOptions ProviderMetadata
}

func (SystemMessage) IsMessage() {}

// UserMessage is a user-role prompt message carrying one or more content parts.
type UserMessage struct {
	Content         []UserContent
	ProviderOptions ProviderMetadata
}

func (UserMessage) IsMessage() {}

// AssistantMessage is an assistant-role prompt message.
type AssistantMessage struct {
	Content         []AssistantContent
	ProviderOptions ProviderMetadata
}

func (AssistantMessage) IsMessage() {}

// ToolMessage carries tool-call result content for the tool role.
type ToolMessage struct{ Content []ToolContent }

func (ToolMessage) IsMessage() {}

type TextContent struct {
	Text            string
	ProviderOptions ProviderMetadata
}

func (TextContent) IsUserContent()      {}
func (TextContent) IsAssistantContent() {}
func (TextContent) IsContent()          {}

type FileContent struct {
	Data            any
	MediaType       string
	Filename        string
	ProviderOptions ProviderMetadata
}

func (FileContent) IsUserContent() {}

type ReasoningContent struct {
	Text            string
	ProviderOptions ProviderMetadata
}

func (ReasoningContent) IsAssistantContent() {}
func (ReasoningContent) IsContent()          {}

type ToolCallContent struct {
	ToolCallID       string
	ToolName         string
	Input            json.RawMessage
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderMetadata
}

func (ToolCallContent) IsAssistantContent() {}
func (ToolCallContent) IsContent()          {}

type ToolResultContent struct {
	ToolCallID      string
	Output          ToolResultOutput
	ProviderOptions ProviderMetadata
}

func (ToolResultContent) IsToolContent() {}

type ToolResultOutput struct {
	Type   string
	Value  any
	Reason string
}

// GenerateOptions carries all parameters for a non-streaming chat or completion
// generate call.
type GenerateOptions struct {
	Messages         []Message
	Tools            []Tool
	ToolChoice       *ToolChoice
	MaxOutputTokens  *int
	Temperature      *float64
	TopK             *int
	TopP             *float64
	StopSequences    []string
	ResponseFormat   *ResponseFormat
	StructuredOutput *StructuredOutput
	FrequencyPenalty *float64
	PresencePenalty  *float64
	Seed             *int
	Headers          http.Header
	ProviderOptions  ProviderOptions
}

// StreamOptions extends [GenerateOptions] with streaming-specific parameters.
type StreamOptions struct {
	GenerateOptions
	IncludeRawChunks bool
}

// EmbedOptions carries parameters for an embedding call.
type EmbedOptions struct {
	Values          []string
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// ImageGenerateOptions carries parameters for an image generation or edit call.
type ImageGenerateOptions struct {
	Prompt          string
	N               int
	Size            string
	AspectRatio     string
	Seed            *int
	Files           []ImageFile
	Mask            *ImageFile
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// ImageFile describes an input image for image edit requests.
// Type must be one of "bytes", "base64", or "url".
type ImageFile struct {
	Type      string
	Data      []byte
	Base64    string
	URL       string
	MediaType string
}

type Tool struct {
	Type        string
	ID          string
	Name        string
	Description string
	InputSchema any
	Strict      *bool
	Args        any
}

type ToolChoice struct {
	Type     string
	ToolName string
}

type ResponseFormat struct {
	Type        string
	Schema      any
	Name        string
	Description string
}

type StructuredOutput struct {
	Name        string
	Description string
	Schema      any
}

// GenerateResult is the result of a non-streaming [LanguageModel.DoGenerate] call.
type GenerateResult struct {
	Content          []Content
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// StreamResult is the result of a [LanguageModel.DoStream] call.
// Read events from Stream (or its alias Parts) channel.
type StreamResult struct {
	Stream   <-chan StreamPart
	Parts    <-chan StreamPart
	Request  RequestMetadata
	Response *StreamResponse
}

// EmbedResult is the result of an [EmbeddingModel.DoEmbed] call.
type EmbedResult struct {
	Embeddings       [][]float64
	Usage            *EmbeddingUsage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// ImageGenerateResult is the result of an [ImageModel.DoGenerate] call.
type ImageGenerateResult struct {
	Images   []string
	Warnings []Warning
	Request  RequestMetadata
	Response ImageResponseMetadata
}

type RequestMetadata struct {
	Body       []byte
	FormFields map[string][]string
}

type ResponseMetadata struct {
	ID        string
	ModelID   string
	Timestamp *time.Time
	Headers   http.Header
	Body      []byte
}

type StreamResponse struct {
	ID               string
	ModelID          string
	Timestamp        *time.Time
	Headers          http.Header
	ProviderMetadata ProviderMetadata
}

type ImageResponseMetadata struct {
	Timestamp time.Time
	ModelID   string
	Headers   http.Header
}

type ProviderMetadata map[string]any
type MessageMetadata map[string]any
type ProviderOptions map[string]map[string]any

type Warning struct {
	Type    string
	Feature string
	Details string
	Message string
}

type FinishReason struct {
	Unified string
	Raw     string
}

type TokenCounts struct {
	Total      *int
	NoCache    *int
	CacheRead  *int
	CacheWrite *int
}

type OutputTokenCounts struct {
	Total     *int
	Text      *int
	Reasoning *int
}

type Usage struct {
	InputTokens  TokenCounts
	OutputTokens OutputTokenCounts
	Raw          json.RawMessage
}

type EmbeddingUsage struct{ Tokens int }

type StreamStart struct{ Warnings []Warning }
type StreamResponseMetadata struct {
	ID        string
	ModelID   string
	Timestamp *time.Time
}
type StreamTextStart struct{ ID string }
type StreamTextDelta struct {
	ID   string
	Text string
}
type StreamTextEnd struct{ ID string }
type StreamReasoningStart struct{ ID string }
type StreamReasoningDelta struct {
	ID   string
	Text string
}
type StreamReasoningEnd struct{ ID string }
type StreamToolInputStart struct {
	ID       string
	ToolName string
}
type StreamToolInputDelta struct {
	ID    string
	Delta string
}
type StreamToolInputEnd struct{ ID string }
type StreamToolCall struct{ ToolCallContent }
type StreamFinish struct {
	FinishReason     FinishReason
	Usage            Usage
	ProviderMetadata ProviderMetadata
}
type StreamError struct{ Err error }
type StreamRaw struct {
	Raw     []byte
	Decoded map[string]any
}

func (StreamStart) IsStreamPart()            {}
func (StreamResponseMetadata) IsStreamPart() {}
func (StreamTextStart) IsStreamPart()        {}
func (StreamTextDelta) IsStreamPart()        {}
func (StreamTextDelta) IsDelta()             {}
func (StreamTextEnd) IsStreamPart()          {}
func (StreamReasoningStart) IsStreamPart()   {}
func (StreamReasoningDelta) IsStreamPart()   {}
func (StreamReasoningDelta) IsDelta()        {}
func (StreamReasoningEnd) IsStreamPart()     {}
func (StreamToolInputStart) IsStreamPart()   {}
func (StreamToolInputDelta) IsStreamPart()   {}
func (StreamToolInputDelta) IsDelta()        {}
func (StreamToolInputEnd) IsStreamPart()     {}
func (StreamToolCall) IsStreamPart()         {}
func (StreamFinish) IsStreamPart()           {}
func (StreamError) IsStreamPart()            {}
func (StreamRaw) IsStreamPart()              {}

// ChatOptions carries recognized typed provider options for chat model calls.
type ChatOptions struct {
	User             string
	ReasoningEffort  string
	TextVerbosity    string
	StrictJSONSchema *bool
}

// CompletionOptions carries recognized typed provider options for completion
// model calls.
type CompletionOptions struct {
	Echo      *bool
	LogitBias map[string]float64
	Suffix    string
	User      string
}

// EmbeddingOptions carries recognized typed provider options for embedding
// model calls.
type EmbeddingOptions struct {
	Dimensions *int
	User       string
}

type MetadataExtractor interface {
	ExtractMetadata(raw []byte, decoded map[string]any) (ProviderMetadata, error)
	CreateStreamExtractor() StreamMetadataExtractor
}

type StreamMetadataExtractor interface {
	ProcessChunk(raw []byte, decoded map[string]any)
	BuildMetadata() ProviderMetadata
}

type OpenAICompatibleTokenUsage struct {
	PromptTokens            *int
	CompletionTokens        *int
	TotalTokens             *int
	PromptTokensDetails     *struct{ CachedTokens *int }
	CompletionTokensDetails *struct {
		ReasoningTokens          *int
		AcceptedPredictionTokens *int
		RejectedPredictionTokens *int
	}
	Raw json.RawMessage
}

type OpenAICompatibleCompletionUsage struct {
	PromptTokens     *int
	CompletionTokens *int
	TotalTokens      *int
	Raw              json.RawMessage
}
