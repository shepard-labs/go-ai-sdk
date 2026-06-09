package openrouter

import (
	"context"
	"net/http"
	"regexp"
	"time"
)

type Provider interface {
	Model(modelID string, opts ...ChatOptions) LanguageModel
	LanguageModel(modelID string, opts ...ChatOptions) LanguageModel
	Chat(modelID string, opts ...ChatOptions) LanguageModel
	Completion(modelID string, opts ...CompletionOptions) LanguageModel
	TextEmbeddingModel(modelID string, opts ...EmbeddingOptions) EmbeddingModel
	Embedding(modelID string, opts ...EmbeddingOptions) EmbeddingModel
	ImageModel(modelID string, opts ...ImageOptions) ImageModel
	Image(modelID string, opts ...ImageOptions) ImageModel
	VideoModel(modelID string, opts ...VideoOptions) VideoModel
	Tools() Tools
	Name() string
	Err() error
}

type LanguageModel interface {
	ModelID() string
	Provider() string
	SupportURLs() map[string][]*regexp.Regexp
	DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error)
	DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error)
}

type EmbeddingModel interface {
	ModelID() string
	Provider() string
	MaxEmbeddingsPerCall() int
	SupportsParallelCalls() bool
	DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error)
}

type ImageModel interface {
	ModelID() string
	Provider() string
	MaxImagesPerCall() int
	DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error)
}

type VideoModel interface {
	ModelID() string
	Provider() string
	MaxVideosPerCall() int
	DoGenerate(ctx context.Context, opts VideoGenerateOptions) (*VideoGenerateResult, error)
}

type Message interface{ IsMessage() }
type UserContent interface{ IsUserContent() }
type AssistantContent interface{ IsAssistantContent() }
type ToolContent interface{ IsToolContent() }
type Content interface{ IsContent() }
type StreamPart interface{ IsStreamPart() }

type ProviderMetadata map[string]any

type SystemMessage struct {
	Content         string
	ProviderOptions ProviderOptions
}

func (SystemMessage) IsMessage() {}

type UserMessage struct {
	Content         []UserContent
	ProviderOptions ProviderOptions
}

func (UserMessage) IsMessage() {}

type AssistantMessage struct {
	Content         []AssistantContent
	ProviderOptions ProviderOptions
}

func (AssistantMessage) IsMessage() {}

type ToolMessage struct {
	Content         []ToolContent
	ProviderOptions ProviderOptions
}

func (ToolMessage) IsMessage() {}

type TextContent struct {
	Text            string
	ProviderOptions ProviderOptions
}

func (TextContent) IsUserContent()      {}
func (TextContent) IsAssistantContent() {}
func (TextContent) IsToolContent()      {}
func (TextContent) IsContent()          {}

type ReasoningContent struct {
	Text             string
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderOptions
}

func (ReasoningContent) IsAssistantContent() {}
func (ReasoningContent) IsContent()          {}

type FileContent struct {
	Data            any
	MediaType       string
	Filename        string
	ProviderOptions ProviderOptions
}

func (FileContent) IsUserContent() {}
func (FileContent) IsToolContent() {}
func (FileContent) IsContent()     {}

type ToolCallContent struct {
	ToolCallID       string
	ToolName         string
	Input            any
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderOptions
}

func (ToolCallContent) IsAssistantContent() {}
func (ToolCallContent) IsContent()          {}

type ToolResultContent struct {
	ToolCallID      string
	ToolName        string
	Output          any
	IsError         bool
	ProviderOptions ProviderOptions
}

func (ToolResultContent) IsToolContent() {}

type SourceContent struct {
	SourceType       string
	ID               string
	URL              string
	Title            string
	ProviderMetadata ProviderMetadata
}

func (SourceContent) IsContent() {}

type GenerateOptions struct {
	Messages         []Message
	Tools            []Tool
	ToolChoice       ToolChoice
	ResponseFormat   *ResponseFormat
	Temperature      *float64
	TopP             *float64
	TopK             *int
	FrequencyPenalty *float64
	PresencePenalty  *float64
	MaxTokens        *int
	Seed             *int
	Stop             []string
	Headers          http.Header
	ProviderOptions  ProviderOptions
}

type StreamOptions struct {
	GenerateOptions
	IncludeRawChunks bool
}

type GenerateResult struct {
	Content          []Content
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

type StreamResult struct {
	Stream   <-chan StreamPart
	Parts    <-chan StreamPart
	Request  RequestMetadata
	Response *StreamResponse
}

type StreamResponse struct {
	Headers http.Header
}

type StreamRaw struct{ Chunk any }
type StreamResponseMetadata struct{ ID, ModelID string }
type StreamReasoningStart struct{ ID string }
type StreamReasoningDelta struct{ ID, Delta string }
type StreamReasoningEnd struct {
	ID               string
	ProviderMetadata ProviderMetadata
}
type StreamTextStart struct{ ID string }
type StreamTextDelta struct{ ID, Delta string }
type StreamTextEnd struct{ ID string }
type StreamToolInputStart struct{ ID, ToolName string }
type StreamToolInputDelta struct{ ID, Delta string }
type StreamToolInputEnd struct{ ID string }
type StreamToolCall struct{ ToolCallContent }
type StreamFile struct{ MediaType, Data string }
type StreamSource struct{ SourceContent }
type StreamFinish struct {
	FinishReason     FinishReason
	Usage            Usage
	ProviderMetadata ProviderMetadata
}
type StreamError struct{ Err error }

func (StreamRaw) IsStreamPart()              {}
func (StreamResponseMetadata) IsStreamPart() {}
func (StreamReasoningStart) IsStreamPart()   {}
func (StreamReasoningDelta) IsStreamPart()   {}
func (StreamReasoningEnd) IsStreamPart()     {}
func (StreamTextStart) IsStreamPart()        {}
func (StreamTextDelta) IsStreamPart()        {}
func (StreamTextEnd) IsStreamPart()          {}
func (StreamToolInputStart) IsStreamPart()   {}
func (StreamToolInputDelta) IsStreamPart()   {}
func (StreamToolInputEnd) IsStreamPart()     {}
func (StreamToolCall) IsStreamPart()         {}
func (StreamFile) IsStreamPart()             {}
func (StreamSource) IsStreamPart()           {}
func (StreamFinish) IsStreamPart()           {}
func (StreamError) IsStreamPart()            {}

type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content-filter"
	FinishReasonToolCalls     FinishReason = "tool-calls"
	FinishReasonError         FinishReason = "error"
	FinishReasonOther         FinishReason = "other"
)

type Usage struct {
	InputTokens         int
	InputTokensDetails  InputTokensDetails
	OutputTokens        int
	OutputTokensDetails OutputTokensDetails
	TotalTokens         int
	Raw                 any
}

type InputTokensDetails struct {
	CachedTokens     int
	CacheWriteTokens int
}

type OutputTokensDetails struct{ ReasoningTokens int }

type Warning struct {
	Type    string
	Message string
}

type RequestMetadata struct{ Body []byte }

type ResponseMetadata struct {
	ID        string
	ModelID   string
	Headers   http.Header
	RawBody   []byte
	Timestamp time.Time
}

type ImageResponseMetadata struct {
	ID        string
	ModelID   string
	Headers   http.Header
	Timestamp time.Time
}

type ResponseFormat struct {
	Type        string
	Schema      any
	Name        string
	Description string
	Strict      *bool
}

type ToolChoice struct {
	Type     string
	ToolName string
}

type EmbedOptions struct {
	Values          []string
	Headers         http.Header
	ProviderOptions ProviderOptions
}

type EmbedResult struct {
	Embeddings       [][]float64
	Usage            EmbeddingUsage
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

type EmbeddingUsage struct{ Tokens int }

type ImageGenerateOptions struct {
	Prompt          string
	N               int
	Size            string
	AspectRatio     string
	Seed            *int
	InputFiles      []FileContent
	Mask            *FileContent
	Headers         http.Header
	ProviderOptions ProviderOptions
}

type ImageGenerateResult struct {
	Images           []string
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Usage            *ImageUsage
	Request          RequestMetadata
	Response         ImageResponseMetadata
}

type ImageUsage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

type VideoGenerateOptions struct {
	Prompt          string
	N               int
	AspectRatio     string
	Resolution      string
	Duration        string
	Seed            *int
	Image           *VideoFile
	Headers         http.Header
	ProviderOptions ProviderOptions
}

type VideoFile struct {
	Type      string
	URL       string
	Data      []byte
	Base64    string
	MediaType string
}

type VideoData struct {
	Type      string
	URL       string
	MediaType string
}

type VideoGenerateResult struct {
	Videos           []VideoData
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ImageResponseMetadata
}
