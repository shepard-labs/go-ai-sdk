package openai

import (
	"context"
	"regexp"
)

// Provider is the openai provider interface.
type Provider interface {
	// Callable form: openai("gpt-5") is shorthand for Responses("gpt-5").
	Call(modelID string) ResponsesModel

	// Language model families.
	Model(modelID string) ResponsesModel
	LanguageModel(modelID string) ResponsesModel
	Chat(modelID string) LanguageModel
	Responses(modelID string) ResponsesModel
	Completion(modelID string) LanguageModel

	// Embedding.
	EmbeddingModel(modelID string) EmbeddingModel
	Embedding(modelID string) EmbeddingModel
	TextEmbeddingModel(modelID string) EmbeddingModel

	// Image.
	ImageModel(modelID string) ImageModel
	Image(modelID string) ImageModel

	// Audio.
	Speech(modelID string) SpeechModel
	SpeechModel(modelID string) SpeechModel
	Transcription(modelID string) TranscriptionModel
	TranscriptionModel(modelID string) TranscriptionModel

	// File & skill upload.
	Files() Files
	Skills() Skills

	// Realtime.
	ExperimentalRealtime() ExperimentalRealtimeFactory

	// Tool factory bag.
	Tools() Tools

	Name() string
	Err() error
}

// LanguageModel is the openai package's chat / completion language model
// interface. Implementations wrap an openaicompatible model and apply
// OpenAI-specific request building and per-model capability branching.
type LanguageModel interface {
	ModelID() string
	Provider() string
	SupportURLs() map[string][]*regexp.Regexp
	DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error)
	DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error)
}

// GenerateResult is the result of a non-streaming DoGenerate call.
type GenerateResult struct {
	Content          []Content
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// StreamResult is the result of a DoStream call. Parts is a channel of
// stream events.
type StreamResult struct {
	Stream   <-chan StreamPart
	Parts    <-chan StreamPart
	Request  RequestMetadata
	Response *StreamResponse
}

// EmbeddingModel is the openai package's embedding model interface.
type EmbeddingModel interface {
	ModelID() string
	Provider() string
	MaxEmbeddingsPerCall() int
	SupportsParallelCalls() bool
	DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error)
}

// EmbedResult is the result of a DoEmbed call.
type EmbedResult struct {
	Embeddings       [][]float64
	Usage            *EmbeddingUsage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

// ImageModel is the openai package's image generation / edit model interface.
type ImageModel interface {
	ModelID() string
	Provider() string
	MaxImagesPerCall() int
	DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error)
}

// ImageGenerateResult is the result of a DoGenerate call on an ImageModel.
type ImageGenerateResult struct {
	Images           []string
	Warnings         []Warning
	Usage            *ImageUsage
	Request          RequestMetadata
	Response         ImageResponseMetadata
	ProviderMetadata ProviderMetadata
}

// ImageUsage is the per-image usage breakdown.
type ImageUsage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

// SpeechModel is the openai package's speech synthesis model interface.
type SpeechModel interface {
	ModelID() string
	Provider() string
	DoGenerate(ctx context.Context, opts SpeechGenerateOptions) (*SpeechGenerateResult, error)
}

// TranscriptionModel is the openai package's transcription model interface.
type TranscriptionModel interface {
	ModelID() string
	Provider() string
	DoGenerate(ctx context.Context, opts TranscriptionOptions) (*TranscriptionResult, error)
}

// Files is the openai Files API surface.
type Files interface {
	UploadFile(ctx context.Context, opts FilesUploadOptions) (*FilesUploadResult, error)
}

// Skills is the openai Skills API surface.
type Skills interface {
	UploadSkill(ctx context.Context, opts SkillsUploadOptions) (*SkillsUploadResult, error)
}

// ExperimentalRealtimeFactory creates Realtime models and mints client secrets.
type ExperimentalRealtimeFactory interface {
	RealtimeModel(modelID string) RealtimeModel
	GetToken(opts ClientSecretOptions) (ClientSecretResult, error)
}

// RealtimeModel exposes helpers for driving the OpenAI Realtime API.
type RealtimeModel interface {
	ModelID() string
	Provider() string
	DoCreateClientSecret(ctx context.Context, opts ClientSecretOptions) (*ClientSecretResult, error)
	GetWebSocketConfig(opts WebSocketConfigInput) WebSocketConfig
	ParseServerEvent(raw []byte) RealtimeServerEvent
	SerializeClientEvent(event RealtimeClientEvent) ([]byte, error)
	BuildSessionConfig(cfg SessionConfig) map[string]any
}

// Tools is the openai provider tool factory bag.
type Tools interface {
	ApplyPatch() Tool
	CustomTool(description string, format *CustomToolFormat) Tool
	CodeInterpreter(container *CodeInterpreterContainer) Tool
	FileSearch(args FileSearchArgs) Tool
	ImageGeneration(args ImageGenerationArgs) Tool
	LocalShell() Tool
	Shell(args ShellArgs) Tool
	WebSearch(args WebSearchArgs) Tool
	WebSearchPreview(args WebSearchPreviewArgs) Tool
	MCP(args MCPArgs) Tool
	ToolSearch(args ToolSearchArgs) Tool
}

// Ensure the concrete provider satisfies the interface.
var _ Provider = (*openaiProvider)(nil)
