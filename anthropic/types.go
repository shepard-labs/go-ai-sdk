package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
)

const Version = "0.1.0"

type ModelID string

const (
	ModelClaudeHaiku45  ModelID = "claude-haiku-4-5"
	ModelClaudeSonnet46 ModelID = "claude-sonnet-4-6"
	ModelClaudeOpus48   ModelID = "claude-opus-4-8"
	ModelClaudeFable5   ModelID = "claude-fable-5"
)

type ModelCapabilities struct {
	MaxOutputTokens  int
	StructuredOutput bool
	RejectsSampling  bool
}

type Provider interface {
	Model(modelID string, opts ...ModelOptions) LanguageModel
	LanguageModel(modelID string, opts ...ModelOptions) LanguageModel
	Chat(modelID string, opts ...ModelOptions) LanguageModel
	Messages(modelID string, opts ...ModelOptions) LanguageModel
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

type Message interface{ IsMessage() }
type UserContent interface{ IsUserContent() }
type AssistantContent interface{ IsAssistantContent() }
type ResponseContent interface{ IsResponseContent() }
type Content interface{ IsContent() }
type StreamPart interface{ IsStreamPart() }
type Delta interface{ IsDelta() }

type SystemMessage struct {
	Content      string
	CacheControl *CacheControl
}

func (SystemMessage) IsMessage() {}

type UserMessage struct{ Content []UserContent }

func (UserMessage) IsMessage() {}

type AssistantMessage struct{ Content []AssistantContent }

func (AssistantMessage) IsMessage() {}

type TextContent struct {
	Text      string
	Citations []Citation
}

func (TextContent) IsUserContent()      {}
func (TextContent) IsAssistantContent() {}
func (TextContent) IsResponseContent()  {}
func (TextContent) IsContent()          {}

type ImageContent struct{ Source ImageSource }

func (ImageContent) IsUserContent() {}

type ImageSource struct {
	Type      string
	MediaType string
	Data      string
	URL       string
}

type DocumentContent struct{ Source DocumentSource }

func (DocumentContent) IsUserContent() {}

type DocumentSource struct {
	Type      string
	MediaType string
	Data      string
	URL       string
}

type ThinkingContent struct {
	Thinking  string
	Signature string
}

func (ThinkingContent) IsAssistantContent() {}
func (ThinkingContent) IsResponseContent()  {}

type RedactedThinkingContent struct{ Data string }

func (RedactedThinkingContent) IsAssistantContent() {}
func (RedactedThinkingContent) IsResponseContent()  {}

type CompactionContent struct{ Text string }

func (CompactionContent) IsAssistantContent() {}
func (CompactionContent) IsResponseContent()  {}
func (CompactionContent) IsContent()          {}

type ToolCallContent struct {
	Type             string
	ToolCallID       string
	ToolName         string
	Input            json.RawMessage
	ProviderExecuted bool
	Dynamic          bool
	ProviderMetadata ProviderMetadata
}

func (ToolCallContent) IsAssistantContent() {}
func (ToolCallContent) IsResponseContent()  {}
func (ToolCallContent) IsContent()          {}

type ServerToolUseContent struct {
	ID               string
	Name             string
	Input            json.RawMessage
	ProviderMetadata ProviderMetadata
}

func (ServerToolUseContent) IsAssistantContent() {}
func (ServerToolUseContent) IsResponseContent()  {}

type MCPToolUseContent struct {
	ID               string
	Name             string
	ServerName       string
	Input            json.RawMessage
	ProviderMetadata ProviderMetadata
}

func (MCPToolUseContent) IsAssistantContent() {}
func (MCPToolUseContent) IsResponseContent()  {}

type ToolResultContent struct {
	ToolCallID       string
	Result           []ToolResultPart
	IsError          bool
	ProviderExecuted bool
	Dynamic          bool
	ProviderMetadata ProviderMetadata
}

func (ToolResultContent) IsUserContent() {}
func (ToolResultContent) IsContent()     {}

type ToolResultPart interface{ IsToolResultPart() }

type ToolResultText struct{ Text string }

func (ToolResultText) IsToolResultPart() {}

type ToolResultImage struct{ Source ImageSource }

func (ToolResultImage) IsToolResultPart() {}

type ToolResultDocument struct{ Source DocumentSource }

func (ToolResultDocument) IsToolResultPart() {}

type ToolResultReference struct{ ID string }

func (ToolResultReference) IsToolResultPart() {}

type WebFetchError struct{ Type, ErrorCode string }

func (WebFetchError) IsToolResultPart() {}

type WebFetchResult struct {
	Type        string
	URL         string
	RetrievedAt string
	Content     DocumentContent
}

func (WebFetchResult) IsToolResultPart() {}

type WebSearchError struct{ Type, ErrorCode string }

func (WebSearchError) IsToolResultPart() {}

type WebSearchResult struct {
	Type             string
	URL              string
	Title            string
	EncryptedContent string
	PageAge          string
}

func (WebSearchResult) IsToolResultPart() {}

type CodeExecutionError struct{ Type, ErrorCode string }

func (CodeExecutionError) IsToolResultPart() {}

type CodeExecutionOutput struct {
	Type string
	Text string
}

type CodeExecutionResult struct {
	Type       string
	Stdout     string
	Stderr     string
	ReturnCode int
	Content    []CodeExecutionOutput
}

func (CodeExecutionResult) IsToolResultPart() {}

type EncryptedCodeExecutionResult struct {
	Type            string
	EncryptedStdout string
	Stderr          string
	ReturnCode      int
	Content         []CodeExecutionOutput
}

func (EncryptedCodeExecutionResult) IsToolResultPart() {}

type BashCodeExecutionError struct{ Type, ErrorCode string }

func (BashCodeExecutionError) IsToolResultPart() {}

type BashCodeExecutionOutput struct {
	Type string
	Text string
}

type BashCodeExecutionResult struct {
	Type       string
	Content    []BashCodeExecutionOutput
	Stdout     string
	Stderr     string
	ReturnCode int
}

func (BashCodeExecutionResult) IsToolResultPart() {}

type ToolSearchError struct{ Type, ErrorCode string }

func (ToolSearchError) IsToolResultPart() {}

type ToolReference struct {
	Name        string
	Description string
}

type ToolSearchResult struct {
	Type           string
	ToolReferences []ToolReference
}

func (ToolSearchResult) IsToolResultPart() {}

type AdvisorError struct{ Type, ErrorCode string }

func (AdvisorError) IsToolResultPart() {}

type AdvisorResult struct {
	Type string
	Text string
}

func (AdvisorResult) IsToolResultPart() {}

type AdvisorRedactedResult struct {
	Type             string
	EncryptedContent string
}

func (AdvisorRedactedResult) IsToolResultPart() {}

type MessagesResponse struct {
	ID                string
	Model             string
	Role              string
	Content           []ResponseContent
	StopReason        string
	StopSequence      string
	Usage             ResponseUsage
	Container         *ContainerInfo
	Skills            []SkillInfo
	ContextManagement *ContextManagementResponse
}

type ResponseUsage struct {
	InputTokens              int
	OutputTokens             int
	CacheCreationInputTokens int
	CacheReadInputTokens     int
	ServiceTier              string
	ServerToolUse            *UsageIteration
	Executor                 *ExecutorIteration
	Advisor                  *AdvisorIteration
	Iterations               []UsageIteration
}

type UsageIteration struct{ InputTokens, OutputTokens int }
type ExecutorIteration struct{ InputTokens, OutputTokens int }
type AdvisorIteration struct{ InputTokens, OutputTokens int }

type Citation struct {
	Type            string `json:"type,omitempty"`
	Text            string `json:"text,omitempty"`
	CitedText       string `json:"cited_text,omitempty"`
	URL             string `json:"url,omitempty"`
	Title           string `json:"title,omitempty"`
	EncryptedIndex  string `json:"encrypted_index,omitempty"`
	DocumentIndex   int    `json:"document_index,omitempty"`
	DocumentTitle   string `json:"document_title,omitempty"`
	StartPageNumber int    `json:"start_page_number,omitempty"`
	EndPageNumber   int    `json:"end_page_number,omitempty"`
	StartChar       int    `json:"start_char,omitempty"`
	EndChar         int    `json:"end_char,omitempty"`
	StartCharIndex  int    `json:"start_char_index,omitempty"`
	EndCharIndex    int    `json:"end_char_index,omitempty"`
}

type ContainerInfo struct{ ID string }

type ContextManagementResponse struct {
	Edits        []ContextManagementEditResponse
	AppliedEdits []ContextManagementEditResponse
}

type ReasoningContent struct {
	Type      string
	Text      string
	Signature string
}

func (ReasoningContent) IsContent() {}

type SourceContent struct {
	SourceType string
	ID         string
	URL        string
	Title      string
	MediaType  string
	Filename   string
}

func (SourceContent) IsContent() {}

type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonToolCalls     FinishReason = "tool-calls"
	FinishReasonContentFilter FinishReason = "content-filter"
	FinishReasonError         FinishReason = "error"
	FinishReasonUnknown       FinishReason = "unknown"
)

type Usage struct {
	InputTokens  TokenUsage
	OutputTokens TokenUsage
	TotalTokens  int
	Iterations   []UsageIteration
}

type TokenUsage struct {
	Total      int
	CacheRead  int
	CacheWrite int
}

type GenerateOptions struct {
	Messages         []Message
	Tools            []Tool
	ToolChoice       *ToolChoice
	ToolOptions      ToolOptions
	MaxTokens        int
	Temperature      *float64
	TopK             *int
	TopP             *float64
	StopSequences    []string
	ResponseFormat   *ResponseFormat
	StructuredOutput *StructuredOutput
	FrequencyPenalty *float64
	PresencePenalty  *float64
	Seed             *int
	Thinking         *ThinkingConfig
}

type StreamOptions = GenerateOptions

type GenerateRequest struct {
	Messages []Message
	Tools    []Tool
	Options  ModelOptions
}

type GenerateResult struct {
	Content          []Content
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	MessageMetadata  MessageMetadata
}

type GenerateResponse = GenerateResult

type Warning struct {
	Type    string
	Message string
}

type ProviderMetadata map[string]any
type MessageMetadata map[string]any

type StreamResult struct {
	Stream   <-chan StreamPart
	Parts    <-chan StreamPart
	Request  []byte
	Response *StreamResponse
}

type StreamResponse struct {
	ID      string
	ModelID string
	Headers http.Header
	Body    []byte
}

type StreamStart struct{}

func (StreamStart) IsStreamPart() {}

type StreamTextStart struct{ ID string }

func (StreamTextStart) IsStreamPart() {}

type StreamTextDelta struct {
	ID        string
	Text      string
	Citations []Citation
}

func (StreamTextDelta) IsStreamPart() {}

type StreamTextEnd struct{ ID string }

func (StreamTextEnd) IsStreamPart() {}

type StreamReasoningStart struct{ ID string }

func (StreamReasoningStart) IsStreamPart() {}

type StreamReasoningDelta struct {
	ID    string
	Delta Delta
}

func (StreamReasoningDelta) IsStreamPart() {}

type StreamReasoningEnd struct{ ID string }

func (StreamReasoningEnd) IsStreamPart() {}

type StreamToolInputStart struct {
	ID               string
	ToolName         string
	ProviderExecuted bool
	Dynamic          bool
}

func (StreamToolInputStart) IsStreamPart() {}

type StreamToolInputDelta struct {
	ID    string
	Delta Delta
}

func (StreamToolInputDelta) IsStreamPart() {}

type StreamToolInputEnd struct{ ID string }

func (StreamToolInputEnd) IsStreamPart() {}

type StreamToolCall struct{ ToolCallContent }

func (StreamToolCall) IsStreamPart() {}

type StreamToolResult struct{ ToolResultContent }

func (StreamToolResult) IsStreamPart() {}

type StreamSource struct {
	SourceContent
	ProviderMetadata ProviderMetadata
}

func (StreamSource) IsStreamPart() {}

type StreamResponseMetadata struct {
	ID      string
	ModelID string
}

func (StreamResponseMetadata) IsStreamPart() {}

type StreamFinish struct {
	FinishReason     FinishReason
	Usage            Usage
	ProviderMetadata ProviderMetadata
}

func (StreamFinish) IsStreamPart() {}

type StreamError struct{ Err error }

func (StreamError) IsStreamPart() {}

type StreamRaw struct{ Event any }

func (StreamRaw) IsStreamPart() {}

type ThinkingDelta struct{ Thinking string }

func (ThinkingDelta) IsDelta() {}

type SignatureDelta struct{ Signature string }

func (SignatureDelta) IsDelta() {}

type InputJSONDelta struct{ PartialJSON string }

func (InputJSONDelta) IsDelta() {}

type Tool struct {
	ID                  string                   `json:"id,omitempty"`
	Name                string                   `json:"name,omitempty"`
	Description         string                   `json:"description,omitempty"`
	InputSchema         any                      `json:"input_schema,omitempty"`
	ArgsSchema          any                      `json:"args_schema,omitempty"`
	OutputSchema        any                      `json:"output_schema,omitempty"`
	CacheControl        *CacheControl            `json:"cache_control,omitempty"`
	EagerInputStreaming *bool                    `json:"eager_input_streaming,omitempty"`
	Strict              *bool                    `json:"strict,omitempty"`
	DeferLoading        *bool                    `json:"defer_loading,omitempty"`
	AllowedCallers      []ToolCallCaller         `json:"allowed_callers,omitempty"`
	InputExamples       []any                    `json:"input_examples,omitempty"`
	Type                string                   `json:"type,omitempty"`
	DisplayWidthPx      *int                     `json:"display_width_px,omitempty"`
	DisplayHeightPx     *int                     `json:"display_height_px,omitempty"`
	DisplayNumber       *int                     `json:"display_number,omitempty"`
	EnableZoom          *bool                    `json:"enable_zoom,omitempty"`
	MaxCharacters       *int                     `json:"max_characters,omitempty"`
	MaxUses             *int                     `json:"max_uses,omitempty"`
	AllowedDomains      []string                 `json:"allowed_domains,omitempty"`
	BlockedDomains      []string                 `json:"blocked_domains,omitempty"`
	Citations           *CitationsConfig         `json:"citations,omitempty"`
	MaxContentTokens    *int                     `json:"max_content_tokens,omitempty"`
	UserLocation        *UserLocation            `json:"user_location,omitempty"`
	AdvisorModel        string                   `json:"advisor_model,omitempty"`
	AdvisorMaxUses      *int                     `json:"advisor_max_uses,omitempty"`
	AdvisorCaching      *CachingConfig           `json:"advisor_caching,omitempty"`
	MCPServerName       string                   `json:"mcp_server_name,omitempty"`
	DefaultConfig       *MCPToolConfig           `json:"default_config,omitempty"`
	Configs             map[string]MCPToolConfig `json:"configs,omitempty"`
	Dynamic             bool                     `json:"dynamic,omitempty"`
	ProviderExecuted    bool                     `json:"provider_executed,omitempty"`
}

type MCPToolConfig struct {
	Enabled      *bool `json:"enabled,omitempty"`
	DeferLoading *bool `json:"defer_loading,omitempty"`
}

type ToolFactory func() Tool

type Tools struct{ ToolNameMapping ToolNameMapping }

type ToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

type ToolNameMapping map[string]string

type CacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type CacheControlledTextContent struct {
	Text         string
	CacheControl *CacheControl
}

func (CacheControlledTextContent) IsUserContent()      {}
func (CacheControlledTextContent) IsAssistantContent() {}

type CacheControlledImageContent struct {
	Source       ImageSource
	CacheControl *CacheControl
}

func (CacheControlledImageContent) IsUserContent() {}

type CacheControlledDocumentContent struct {
	Source       DocumentSource
	CacheControl *CacheControl
}

func (CacheControlledDocumentContent) IsUserContent() {}

type ThinkingType string
type ThinkingDisplay string

const (
	ThinkingTypeAdaptive ThinkingType = "adaptive"
	ThinkingTypeEnabled  ThinkingType = "enabled"
	ThinkingTypeDisabled ThinkingType = "disabled"
)

type ThinkingConfig struct {
	Type         ThinkingType
	BudgetTokens int
	Display      ThinkingDisplay
}

type StructuredOutputMode string

const (
	StructuredOutputModeOutputFormat StructuredOutputMode = "outputFormat"
	StructuredOutputModeJSONTool     StructuredOutputMode = "jsonTool"
	StructuredOutputModeAuto         StructuredOutputMode = "auto"
)

type OutputConfig struct {
	Mode       StructuredOutputMode `json:"-"`
	Effort     string               `json:"effort,omitempty"`
	Format     *ResponseFormat      `json:"format,omitempty"`
	Schema     any                  `json:"schema,omitempty"`
	TaskBudget *TokenTaskBudget     `json:"task_budget,omitempty"`
}

type TokenTaskBudget struct {
	Type  string `json:"type"`
	Total int    `json:"total"`
}

type StructuredOutput struct {
	Name        string
	Description string
	Schema      any
}

type Metadata struct {
	UserID string `json:"user_id,omitempty"`
}
type ToolCallCaller string

type CitationsConfig struct{ Enabled bool }

type UserLocation struct {
	Type     string
	City     string
	Region   string
	Country  string
	Timezone string
}

type CachingConfig struct{ Enabled bool }

type ResponseFormat struct {
	Type   string `json:"type,omitempty"`
	Schema any    `json:"schema,omitempty"`
}
