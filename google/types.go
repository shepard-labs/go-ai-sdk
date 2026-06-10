package google

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// Version is the package version used in the User-Agent header.
const Version = "0.1.0"

// ProviderName is the outer key used in ProviderOptions maps.
const ProviderName = "google"

// SkipThoughtSignatureValidator is the documented Google sentinel value for
// replayed functionCall parts whose original thoughtSignature is no longer
// available to the client. Injected only on Gemini 3+ models.
const SkipThoughtSignatureValidator = "skip_thought_signature_validator"

// Fetcher executes HTTP requests. Type alias of openaicompatible.Fetcher so
// callers see a single vocabulary across providers.
type Fetcher = openaicompatible.Fetcher

// IDGenerator generates request IDs. Type alias of openaicompatible.IDGenerator.
type IDGenerator = openaicompatible.IDGenerator

// RetryOptions configures retry behavior. Type alias of openaicompatible.RetryOptions.
type RetryOptions = openaicompatible.RetryOptions

// Logger receives operational logs. Type alias of openaicompatible.Logger.
type Logger = openaicompatible.Logger

// ProviderOptions is the map of provider-specific options keyed by provider name.
// Use "google" as the outer key.
type ProviderOptions = map[string]map[string]any

// ProviderMetadata carries untyped provider-specific metadata returned alongside
// results and stream events.
type ProviderMetadata = map[string]any

// RequestMetadata carries the serialized request body and any form fields.
type RequestMetadata struct {
	Body       []byte
	FormFields map[string][]string
}

// ResponseMetadata carries the parsed HTTP response metadata returned alongside
// results.
type ResponseMetadata struct {
	ID        string
	ModelID   string
	Timestamp *time.Time
	Headers   http.Header
	Body      []byte
}

// StreamResponse carries the initial HTTP response metadata for a streaming
// call.
type StreamResponse struct {
	ID               string
	ModelID          string
	Timestamp        *time.Time
	Headers          http.Header
	ProviderMetadata ProviderMetadata
}

// Provider creates Google Generative AI model families.
type Provider interface {
	// Model returns a LanguageModel for the given model ID. Canonical alias.
	Model(modelID string) LanguageModel
	// LanguageModel returns a LanguageModel for the given model ID.
	LanguageModel(modelID string) LanguageModel
	// ChatModel returns a LanguageModel for the given model ID.
	ChatModel(modelID string) LanguageModel
	// Chat returns a LanguageModel for the given model ID.
	Chat(modelID string) LanguageModel
	// GenerativeAI is a deprecated alias for LanguageModel, kept for parity
	// with the upstream createGoogleGenerativeAI factory.
	GenerativeAI(modelID string) LanguageModel

	// EmbeddingModel returns an EmbeddingModel for the given model ID.
	EmbeddingModel(modelID string) EmbeddingModel
	// Embedding returns an EmbeddingModel for the given model ID.
	Embedding(modelID string) EmbeddingModel
	// TextEmbeddingModel is a deprecated alias for EmbeddingModel.
	TextEmbeddingModel(modelID string) EmbeddingModel
	// TextEmbedding is a deprecated alias for EmbeddingModel.
	TextEmbedding(modelID string) EmbeddingModel

	// ImageModel returns an ImageModel for the given model ID.
	ImageModel(modelID string, settings ...ImageModelSettings) ImageModel
	// Image returns an ImageModel for the given model ID.
	Image(modelID string, settings ...ImageModelSettings) ImageModel

	// VideoModel returns a VideoModel for the given model ID.
	VideoModel(modelID string) VideoModel
	// Video returns a VideoModel for the given model ID.
	Video(modelID string) VideoModel

	// SpeechModel returns a SpeechModel for the given model ID.
	SpeechModel(modelID string) SpeechModel
	// Speech returns a SpeechModel for the given model ID.
	Speech(modelID string) SpeechModel

	// Files returns the Files API client.
	Files() Files

	// Tools returns the provider-tool factory set.
	Tools() ToolFactories

	// Name returns the provider name.
	Name() string
	// Err returns a non-nil error if the provider was constructed with invalid
	// settings (e.g. missing API key).
	Err() error
}

// LanguageModel is the Google language model interface.
type LanguageModel interface {
	// ModelID returns the model's ID string.
	ModelID() string
	// Provider returns the provider name suffix (e.g. "google.generative-ai.chat").
	Provider() string
	// SupportURLs returns the URL patterns that this model supports as file inputs.
	SupportURLs() map[string][]*regexp.Regexp
	// DoGenerate performs a non-streaming generation call.
	DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error)
	// DoStream performs a streaming generation call.
	DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error)
}

// ChatLanguageModel is an alias for LanguageModel (matches upstream type alias).
type ChatLanguageModel = LanguageModel

// EmbeddingModel is the Google embedding model interface.
type EmbeddingModel interface {
	// ModelID returns the model's ID string.
	ModelID() string
	// Provider returns the provider name suffix.
	Provider() string
	// MaxEmbeddingsPerCall returns the maximum number of values per batch call.
	MaxEmbeddingsPerCall() int
	// SupportsParallelCalls reports whether the model supports concurrent batches.
	SupportsParallelCalls() bool
	// DoEmbed performs an embedding call.
	DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error)
}

// ImageModel is the Google image model interface.
type ImageModel interface {
	// ModelID returns the model's ID string.
	ModelID() string
	// Provider returns the provider name suffix.
	Provider() string
	// MaxImagesPerCall returns the maximum number of images per call.
	MaxImagesPerCall() int
	// DoGenerate performs an image generation call.
	DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error)
}

// VideoModel is the Google video model interface.
type VideoModel interface {
	// ModelID returns the model's ID string.
	ModelID() string
	// Provider returns the provider name suffix.
	Provider() string
	// MaxVideosPerCall returns the maximum number of videos per call.
	MaxVideosPerCall() int
	// DoGenerate performs a video generation call.
	DoGenerate(ctx context.Context, opts VideoGenerateOptions) (*VideoGenerateResult, error)
}

// SpeechModel is the Google speech synthesis model interface.
type SpeechModel interface {
	// ModelID returns the model's ID string.
	ModelID() string
	// Provider returns the provider name suffix.
	Provider() string
	// DoGenerate performs a speech synthesis call.
	DoGenerate(ctx context.Context, opts SpeechGenerateOptions) (*SpeechGenerateResult, error)
}

// Files is the Google Files API client interface.
type Files interface {
	// Upload performs a resumable file upload and polls until the file is ready.
	Upload(ctx context.Context, data []byte, opts FilesUploadOptions) (*FilesUploadResult, error)
}

// ToolFactories holds the provider-tool factory functions.
type ToolFactories struct {
	// GoogleSearch returns a googleSearch provider tool.
	GoogleSearch func(args ...GoogleSearchArgs) Tool
	// EnterpriseWebSearch returns an enterpriseWebSearch provider tool (Vertex only).
	EnterpriseWebSearch func() Tool
	// GoogleMaps returns a googleMaps provider tool.
	GoogleMaps func() Tool
	// UrlContext returns a urlContext provider tool.
	UrlContext func() Tool
	// FileSearch returns a fileSearch provider tool.
	FileSearch func(args FileSearchArgs) Tool
	// CodeExecution returns a codeExecution provider tool.
	CodeExecution func() Tool
	// VertexRagStore returns a vertexRagStore provider tool (Vertex only).
	VertexRagStore func(args VertexRagStoreArgs) Tool
}

// ---- Sealed-union marker interfaces ----

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

// ---- Message types ----

// SystemMessage is a system-role prompt message.
type SystemMessage struct {
	Content         string
	ProviderOptions ProviderOptions
}

func (SystemMessage) IsMessage() {}

// UserMessage is a user-role prompt message carrying one or more content parts.
type UserMessage struct {
	Content         []UserContent
	ProviderOptions ProviderOptions
}

func (UserMessage) IsMessage() {}

// AssistantMessage is an assistant-role prompt message.
type AssistantMessage struct {
	Content         []AssistantContent
	ProviderOptions ProviderOptions
}

func (AssistantMessage) IsMessage() {}

// ToolMessage carries tool-call result content for the tool role.
type ToolMessage struct {
	Content         []ToolContent
	ProviderOptions ProviderOptions
}

func (ToolMessage) IsMessage() {}

// ---- User content types ----

// TextContent is a plain text content part. Implements UserContent and
// AssistantContent.
type TextContent struct {
	Text            string
	ProviderOptions ProviderOptions
}

func (TextContent) IsUserContent()      {}
func (TextContent) IsAssistantContent() {}
func (TextContent) IsContent()          {}

// ImageContent is an image content part with inline data, URL, or reference.
type ImageContent struct {
	Source          ImageSource
	ProviderOptions ProviderOptions
}

func (ImageContent) IsUserContent() {}
func (ImageContent) IsContent()     {}

// AudioContent is an audio content part.
type AudioContent struct {
	Source          AudioSource
	ProviderOptions ProviderOptions
}

func (AudioContent) IsUserContent() {}

// DocumentContent is a document content part (PDF, text, etc.).
type DocumentContent struct {
	Source          DocumentSource
	ProviderOptions ProviderOptions
}

func (DocumentContent) IsUserContent() {}

// VideoContent is a video content part (URL or YouTube link).
type VideoContent struct {
	Source          VideoSource
	ProviderOptions ProviderOptions
}

func (VideoContent) IsUserContent() {}

// FileContent is a raw file content part with data bytes, base64, or URL.
type FileContent struct {
	Data            any // []byte | string | *url.URL
	MediaType       string
	Filename        string
	ProviderOptions ProviderOptions
}

func (FileContent) IsUserContent() {}

// ---- Assistant content types ----

// ReasoningContent carries a model thinking/reasoning block.
type ReasoningContent struct {
	Text            string
	Signature       string
	ProviderOptions ProviderOptions
}

func (ReasoningContent) IsAssistantContent() {}
func (ReasoningContent) IsContent()          {}

// ToolCallContent carries a function-call or server-tool-call from the model.
type ToolCallContent struct {
	ToolCallID       string
	ToolName         string
	Input            json.RawMessage
	ProviderExecuted bool
	Dynamic          bool
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderOptions
}

func (ToolCallContent) IsAssistantContent() {}
func (ToolCallContent) IsContent()          {}

// ExecutableCodeContent carries a code block produced by the code-execution tool.
type ExecutableCodeContent struct {
	Language        string
	Code            string
	ProviderOptions ProviderOptions
}

func (ExecutableCodeContent) IsAssistantContent() {}
func (ExecutableCodeContent) IsContent()          {}

// CodeExecutionResultContent carries the output of an executed code block.
type CodeExecutionResultContent struct {
	Outcome         string
	Output          string
	ProviderOptions ProviderOptions
}

func (CodeExecutionResultContent) IsAssistantContent() {}
func (CodeExecutionResultContent) IsContent()          {}

// ---- Tool content types ----

// ToolResultContent carries the output of a tool call.
type ToolResultContent struct {
	ToolCallID      string
	Output          ToolResultOutput
	ProviderOptions ProviderOptions
}

func (ToolResultContent) IsToolContent() {}

// ToolResultOutput describes the output value of a tool call.
type ToolResultOutput struct {
	Type   string // "text" | "json" | "content" | "execution-denied"
	Value  any    // string for "text", any for "json", []ContentPart for "content"
	Reason string
}

// ---- Source / media types ----

// ImageSource describes an image input as data, URL, or provider reference.
type ImageSource struct {
	Type      string // "data" | "url" | "reference"
	MediaType string
	Data      string // base64-encoded bytes (when Type == "data")
	URL       string // HTTP or provider-reference URL
}

// AudioSource describes an audio input.
type AudioSource = ImageSource

// DocumentSource describes a document input.
type DocumentSource = ImageSource

// VideoSource describes a video input (URL or YouTube link).
type VideoSource = ImageSource

// Source is an extracted grounding citation from groundingMetadata.
type Source struct {
	Type      string // "url" | "document"
	URL       string
	Title     string
	MediaType string
	Filename  string
}

// ContentPart is a multimodal part used in embedding content overrides.
type ContentPart struct {
	Text       string
	InlineData *InlineDataPart
	FileData   *FileDataPart
}

// InlineDataPart is a base64 inline data part.
type InlineDataPart struct {
	MimeType string
	Data     string // base64
}

// FileDataPart is a file URI part.
type FileDataPart struct {
	MimeType string
	FileURI  string
}

// ReferenceImage is an image passed via ProviderOptions for Veo video generation.
type ReferenceImage struct {
	BytesBase64Encoded string
	GcsUri             string
}

// ---- Capability types ----

// ModelCapabilities describes the features supported by a given model ID.
type ModelCapabilities struct {
	SupportsImages                bool
	SupportsAudio                 bool
	SupportsVideo                 bool
	SupportsCodeExecution         bool
	SupportsUrlContext            bool
	SupportsFileSearch            bool
	SupportsGoogleSearch          bool
	SupportsGrounding             bool
	SupportsThinking              bool
	SupportsImageOutput           bool
	SupportsAudioOutput           bool
	SupportsCachedContent         bool
	SupportsSystemInstruction     bool
	SupportsStructuredOutput      bool
	MaxOutputTokens               int
	SupportsTools                 bool
	SupportsFunctionCallStreaming bool
}

// ---- Tool types ----

// Tool describes a function or provider tool passed to a model.
type Tool struct {
	Type             string // "function" | provider-tool id like "google.google_search"
	ID               string
	Name             string
	Description      string
	InputSchema      any
	ArgsSchema       any
	OutputSchema     any
	Strict           *bool
	ProviderExecuted bool
	Dynamic          bool
}

// ToolChoice describes the tool-choice constraint.
type ToolChoice struct {
	Type     string // "auto" | "none" | "required" | "tool"
	ToolName string
}

// ResponseFormat describes the desired response format for structured output.
type ResponseFormat struct {
	Type        string // "text" | "json"
	Schema      any
	Name        string
	Description string
}

// StructuredOutput describes a named structured output schema.
type StructuredOutput struct {
	Name        string
	Description string
	Schema      any
}

// ---- Generate options ----

// GenerateOptions carries all parameters for a non-streaming language model call.
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
	// Reasoning maps to thinkingConfig; values: "none"|"minimal"|"low"|"medium"|"high"|"xhigh".
	Reasoning string
}

// StreamOptions extends GenerateOptions with streaming-specific parameters.
type StreamOptions struct {
	GenerateOptions
	IncludeRawChunks bool
}

// ---- Result types ----

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
type StreamResult struct {
	Stream   <-chan StreamPart
	Parts    <-chan StreamPart
	Request  RequestMetadata
	Response *StreamResponse
}

// EmbedOptions carries parameters for an embedding call.
type EmbedOptions struct {
	Values          []string
	Headers         http.Header
	ProviderOptions ProviderOptions
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

// EmbeddingUsage reports token usage for an embedding call.
type EmbeddingUsage struct{ Tokens int }

// ---- Image types ----

// ImageModelSettings configures per-call settings for image models.
type ImageModelSettings struct {
	MaxImagesPerCall *int
}

// ImageGenerateOptions carries parameters for an image generation call.
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

// ImageFile describes an input image for image generation or editing.
type ImageFile struct {
	Type      string // "data" | "url" | "reference"
	Data      []byte
	Base64    string
	URL       string
	MediaType string
}

// ImageGenerateResult is the result of an [ImageModel.DoGenerate] call.
type ImageGenerateResult struct {
	Images           []string // base64-encoded
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// ---- Video types ----

// VideoGenerateOptions carries parameters for a video generation call.
type VideoGenerateOptions struct {
	Prompt          string
	N               int
	AspectRatio     string
	Resolution      string
	Duration        int
	Seed            *int
	Image           *ImageFile
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// VideoGenerateResult is the result of a [VideoModel.DoGenerate] call.
type VideoGenerateResult struct {
	Videos           []string // download URIs (with ?key= appended)
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// ---- Speech types ----

// SpeechGenerateOptions carries parameters for a speech synthesis call.
type SpeechGenerateOptions struct {
	Text            string
	Voice           string
	Instructions    string
	OutputFormat    string // "wav" | "pcm"
	Speed           float64
	Language        string
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// SpeechGenerateResult is the result of a [SpeechModel.DoGenerate] call.
type SpeechGenerateResult struct {
	Audio            []byte
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// ---- Files types ----

// FilesUploadOptions carries parameters for a Files API upload.
type FilesUploadOptions struct {
	Data            []byte
	MediaType       string
	Filename        string
	PollIntervalMs  *int
	PollTimeoutMs   *int
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// FilesUploadProviderOptions carries recognized provider-specific options for
// file uploads, parsed from ProviderOptions["google"].
type FilesUploadProviderOptions struct {
	DisplayName    string
	PollIntervalMs *int
	PollTimeoutMs  *int
}

// FilesUploadResult is the result of a [Files.Upload] call.
type FilesUploadResult struct {
	ProviderReference map[string]any // {"google": "<file uri>"}
	MediaType         string
	ProviderMetadata  ProviderMetadata
}

// ---- Shared value types ----

// FinishReason carries both the unified finish-reason string and the raw Google
// finish-reason value.
type FinishReason struct {
	Unified string // "stop" | "length" | "tool-calls" | "content-filter" | "error" | "other"
	Raw     string // raw Google value: STOP, MAX_TOKENS, SAFETY, ...
}

// TokenCounts carries input token count breakdowns.
type TokenCounts struct {
	Total      *int
	NoCache    *int
	CacheRead  *int
	CacheWrite *int
}

// OutputTokenCounts carries output token count breakdowns.
type OutputTokenCounts struct {
	Total     *int
	Text      *int
	Reasoning *int
}

// Usage carries token usage for a generation call.
type Usage struct {
	InputTokens  TokenCounts
	OutputTokens OutputTokenCounts
	Raw          json.RawMessage
}

// Warning carries a non-fatal advisory from the provider.
type Warning struct {
	Type    string
	Feature string
	Details string
	Message string
}

// ---- Stream part types ----

// StreamStart is the first part emitted in a stream; carries any startup warnings.
type StreamStart struct{ Warnings []Warning }

// StreamResponseMetadata carries the HTTP response metadata emitted early in a stream.
type StreamResponseMetadata struct {
	ID        string
	ModelID   string
	Timestamp *time.Time
}

// StreamTextStart marks the beginning of a text content block.
type StreamTextStart struct{ ID string }

// StreamTextDelta carries an incremental text fragment.
type StreamTextDelta struct{ ID, Text string }

// StreamTextEnd marks the end of a text content block.
type StreamTextEnd struct{ ID string }

// StreamReasoningStart marks the beginning of a reasoning/thinking block.
type StreamReasoningStart struct{ ID string }

// StreamReasoningDelta carries an incremental reasoning fragment.
type StreamReasoningDelta struct{ ID, Text string }

// StreamReasoningEnd marks the end of a reasoning block.
type StreamReasoningEnd struct{ ID string }

// StreamToolInputStart marks the beginning of a tool-call input block.
type StreamToolInputStart struct {
	ID               string
	ToolName         string
	ProviderExecuted bool
	Dynamic          bool
}

// StreamToolInputDelta carries an incremental partial JSON fragment for a tool input.
type StreamToolInputDelta struct {
	ID    string
	Delta string
}

// StreamToolInputEnd marks the end of a tool-call input block.
type StreamToolInputEnd struct{ ID string }

// StreamToolCall carries a completed tool call.
type StreamToolCall struct{ ToolCall ToolCallContent }

// StreamToolResult carries a completed tool result (server-side).
type StreamToolResult struct{ ToolResult ToolResultContent }

// StreamSource carries an extracted grounding source.
type StreamSource struct {
	Source           Source
	ProviderMetadata ProviderMetadata
}

// StreamFile carries inline image or file data from the model response.
type StreamFile struct {
	Data      string // base64
	MediaType string
}

// StreamReasoningFile carries reasoning-related file data.
type StreamReasoningFile struct {
	Data      string
	MediaType string
}

// StreamFinish is the terminal stream part carrying finish reason, usage, and metadata.
type StreamFinish struct {
	FinishReason     FinishReason
	Usage            Usage
	ProviderMetadata ProviderMetadata
}

// StreamError carries a stream-level error.
type StreamError struct{ Err error }

// StreamRaw carries the raw SSE payload when IncludeRawChunks is true.
type StreamRaw struct {
	Raw     []byte
	Decoded map[string]any
}

// IsStreamPart marker implementations.
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
func (StreamToolResult) IsStreamPart()       {}
func (StreamSource) IsStreamPart()           {}
func (StreamFile) IsStreamPart()             {}
func (StreamReasoningFile) IsStreamPart()    {}
func (StreamFinish) IsStreamPart()           {}
func (StreamError) IsStreamPart()            {}
func (StreamRaw) IsStreamPart()              {}

// ---- Provider-tool argument types ----

// GoogleSearchArgs carries optional arguments for the googleSearch provider tool.
type GoogleSearchArgs struct {
	SearchTypes     *GoogleSearchTypes
	TimeRangeFilter *TimeRangeFilter
}

// GoogleSearchTypes specifies which search types to enable.
type GoogleSearchTypes struct {
	WebSearch   map[string]any
	ImageSearch map[string]any
}

// TimeRangeFilter restricts search results to a time range (RFC 3339 strings).
type TimeRangeFilter struct {
	StartTime string
	EndTime   string
}

// FileSearchArgs carries arguments for the fileSearch provider tool.
type FileSearchArgs struct {
	FileSearchStoreNames []string
	TopK                 *int
	MetadataFilter       string
}

// VertexRagStoreArgs carries arguments for the vertexRagStore provider tool.
type VertexRagStoreArgs struct {
	RagCorpus string // "projects/{p}/locations/{l}/ragCorpora/{c}"
	TopK      *int
}
