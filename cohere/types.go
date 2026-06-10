package cohere

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"time"
)

const Version = "3.0.36"

type Provider interface {
	Model(modelID string) LanguageModel
	LanguageModel(modelID string) LanguageModel
	EmbeddingModel(modelID string) EmbeddingModel
	Embedding(modelID string) EmbeddingModel
	TextEmbeddingModel(modelID string) EmbeddingModel
	TextEmbedding(modelID string) EmbeddingModel
	RerankingModel(modelID string) RerankingModel
	Reranking(modelID string) RerankingModel
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

type RerankingModel interface {
	ModelID() string
	Provider() string
	DoRerank(ctx context.Context, opts RerankOptions) (*RerankResult, error)
}

type CohereChatLanguageModel = cohereChatLanguageModel
type CohereEmbeddingModel = cohereEmbeddingModel
type CohereRerankingModel = cohereRerankingModel

type Message interface{ IsMessage() }
type UserContent interface{ IsUserContent() }
type AssistantContent interface{ IsAssistantContent() }
type ToolContent interface{ IsToolContent() }
type Content interface{ IsContent() }
type StreamPart interface{ IsStreamPart() }
type Delta interface{ IsDelta() }

type SystemMessage struct {
	Content         string
	ProviderOptions ProviderMetadata
}

func (SystemMessage) IsMessage() {}

type UserMessage struct {
	Content         []UserContent
	ProviderOptions ProviderMetadata
}

func (UserMessage) IsMessage() {}

type AssistantMessage struct {
	Content         []AssistantContent
	ProviderOptions ProviderMetadata
}

func (AssistantMessage) IsMessage() {}

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

type SourceContent struct {
	SourceType       string
	ID               string
	MediaType        string
	Title            string
	ProviderMetadata ProviderMetadata
}

func (SourceContent) IsContent() {}

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

type StreamOptions struct {
	GenerateOptions
	IncludeRawChunks bool
}

type EmbedOptions struct {
	Values          []string
	Headers         http.Header
	ProviderOptions ProviderOptions
}

type RerankOptions struct {
	Query           string
	Documents       RerankDocuments
	TopN            *int
	Headers         http.Header
	ProviderOptions ProviderOptions
}

type RerankDocuments struct {
	Type   string
	Values []any
}

type RankedDocument struct {
	Index          int
	RelevanceScore float64
}

func TextDocuments(values ...string) RerankDocuments {
	out := make([]any, len(values))
	for i, v := range values {
		out[i] = v
	}
	return RerankDocuments{Type: "text", Values: out}
}

func ObjectDocuments(values ...any) RerankDocuments {
	return RerankDocuments{Type: "object", Values: append([]any(nil), values...)}
}

type Tool struct {
	Type        string
	ID          string
	Name        string
	Description string
	InputSchema any
	Strict      *bool
}

type ToolChoice struct{ Type, ToolName string }

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
type EmbedResult struct {
	Embeddings       [][]float64
	Usage            *EmbeddingUsage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}
type RerankResult struct {
	Ranking  []RankedDocument
	Warnings []Warning
	Response ResponseMetadata
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

type ProviderMetadata map[string]any
type MessageMetadata map[string]any
type ProviderOptions map[string]map[string]any

type Warning struct{ Type, Feature, Details, Message string }
type FinishReason struct{ Unified, Raw string }
type TokenCounts struct{ Total, NoCache, CacheRead, CacheWrite *int }
type OutputTokenCounts struct{ Total, Text, Reasoning *int }
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
type StreamTextDelta struct{ ID, Text string }
type StreamTextEnd struct{ ID string }
type StreamReasoningStart struct{ ID string }
type StreamReasoningDelta struct{ ID, Text string }
type StreamReasoningEnd struct{ ID string }
type StreamToolInputStart struct{ ID, ToolName string }
type StreamToolInputDelta struct{ ID, Delta string }
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
