// Package internal holds typed JSON request/response envelopes shared across
// the google package's chat, embedding, image, video, speech, and files
// implementations. These types are not part of the public API.
package internal

import "encoding/json"

// ---- Request envelopes ----

// APIGenerateContentRequest is the wire body for
// POST .../models/{model}:generateContent.
type APIGenerateContentRequest struct {
	Contents          []APIContent         `json:"contents"`
	SystemInstruction *APIContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *APIGenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings    []APISafetySetting   `json:"safetySettings,omitempty"`
	Tools             []APITool            `json:"tools,omitempty"`
	ToolConfig        *APIToolConfig       `json:"toolConfig,omitempty"`
	CachedContent     string               `json:"cachedContent,omitempty"`
	Labels            map[string]string    `json:"labels,omitempty"`
	ServiceTier       string               `json:"serviceTier,omitempty"`
}

// APIContent represents a single content turn (role + parts).
type APIContent struct {
	Role  string    `json:"role"` // "user" | "model"
	Parts []APIPart `json:"parts"`
}

// APIPart is a single multimodal part within an APIContent.
type APIPart struct {
	Text                string                  `json:"text,omitempty"`
	Thought             *bool                   `json:"thought,omitempty"`
	ThoughtSignature    string                  `json:"thoughtSignature,omitempty"`
	InlineData          *APIInlineData          `json:"inlineData,omitempty"`
	FileData            *APIFileData            `json:"fileData,omitempty"`
	FunctionCall        *APIFunctionCall        `json:"functionCall,omitempty"`
	FunctionResponse    *APIFunctionResponse    `json:"functionResponse,omitempty"`
	ToolCall            *APIServerToolCall      `json:"toolCall,omitempty"`
	ToolResponse        *APIServerToolResponse  `json:"toolResponse,omitempty"`
	ExecutableCode      *APIExecutableCode      `json:"executableCode,omitempty"`
	CodeExecutionResult *APICodeExecutionResult `json:"codeExecutionResult,omitempty"`
}

// APIInlineData carries base64-encoded inline binary data.
type APIInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// APIFileData carries a file URI reference.
type APIFileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// APIFunctionCall carries a function call from the model.
type APIFunctionCall struct {
	ID           string          `json:"id,omitempty"`
	Name         string          `json:"name"`
	Args         json.RawMessage `json:"args,omitempty"`
	PartialArgs  []APIPartialArg `json:"partialArgs,omitempty"`
	WillContinue *bool           `json:"willContinue,omitempty"`
}

// APIFunctionResponse carries a function-call response from the user.
type APIFunctionResponse struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
	Parts    []APIPart       `json:"parts,omitempty"`
}

// APIServerToolCall carries a server-side (provider-executed) tool call.
type APIServerToolCall struct {
	ToolType string          `json:"toolType"`
	ID       string          `json:"id"`
	Args     json.RawMessage `json:"args,omitempty"`
}

// APIServerToolResponse carries a server-side tool response.
type APIServerToolResponse struct {
	ToolType string          `json:"toolType"`
	ID       string          `json:"id"`
	Response json.RawMessage `json:"response,omitempty"`
}

// APIExecutableCode carries a code block from the code-execution tool.
type APIExecutableCode struct {
	Language string `json:"language"`
	Code     string `json:"code"`
}

// APICodeExecutionResult carries the output of an executed code block.
type APICodeExecutionResult struct {
	Outcome string `json:"outcome"`
	Output  string `json:"output"`
}

// APIPartialArg is the wire shape for a single streaming partial-args entry.
type APIPartialArg struct {
	JSONPath     string   `json:"jsonPath"`
	WillContinue bool     `json:"willContinue,omitempty"`
	StringValue  *string  `json:"stringValue,omitempty"`
	NumberValue  *float64 `json:"numberValue,omitempty"`
	BoolValue    *bool    `json:"boolValue,omitempty"`
	NullValue    *string  `json:"nullValue,omitempty"`
}

// APIGenerationConfig carries the generationConfig body field.
type APIGenerationConfig struct {
	MaxOutputTokens    *int               `json:"maxOutputTokens,omitempty"`
	Temperature        *float64           `json:"temperature,omitempty"`
	TopK               *int               `json:"topK,omitempty"`
	TopP               *float64           `json:"topP,omitempty"`
	FrequencyPenalty   *float64           `json:"frequencyPenalty,omitempty"`
	PresencePenalty    *float64           `json:"presencePenalty,omitempty"`
	StopSequences      []string           `json:"stopSequences,omitempty"`
	Seed               *int               `json:"seed,omitempty"`
	ResponseMimeType   string             `json:"responseMimeType,omitempty"`
	ResponseSchema     any                `json:"responseSchema,omitempty"`
	ResponseModalities []string           `json:"responseModalities,omitempty"`
	AudioTimestamp     *bool              `json:"audioTimestamp,omitempty"`
	ThinkingConfig     *APIThinkingConfig `json:"thinkingConfig,omitempty"`
	MediaResolution    string             `json:"mediaResolution,omitempty"`
	ImageConfig        *APIImageConfig    `json:"imageConfig,omitempty"`
	SpeechConfig       *APISpeechConfig   `json:"speechConfig,omitempty"`
}

// APIThinkingConfig controls the model's thinking/reasoning budget.
type APIThinkingConfig struct {
	IncludeThoughts *bool  `json:"includeThoughts,omitempty"`
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
}

// APIImageConfig carries image-generation config (aspect ratio and size).
type APIImageConfig struct {
	AspectRatio string `json:"aspectRatio,omitempty"`
	ImageSize   string `json:"imageSize,omitempty"`
}

// APISpeechConfig carries TTS voice configuration.
type APISpeechConfig struct {
	VoiceConfig             *APISingleVoiceConfig       `json:"voiceConfig,omitempty"`
	MultiSpeakerVoiceConfig *APIMultiSpeakerVoiceConfig `json:"multiSpeakerVoiceConfig,omitempty"`
}

// APISingleVoiceConfig names a single prebuilt voice.
type APISingleVoiceConfig struct {
	PrebuiltVoiceConfig APIPrebuiltVoiceConfig `json:"prebuiltVoiceConfig"`
}

// APIMultiSpeakerVoiceConfig lists per-speaker voice assignments.
type APIMultiSpeakerVoiceConfig struct {
	SpeakerVoiceConfigs []APISpeakerVoiceConfig `json:"speakerVoiceConfigs"`
}

// APISpeakerVoiceConfig binds a speaker label to a voice.
type APISpeakerVoiceConfig struct {
	Speaker     string                 `json:"speaker"`
	VoiceConfig APIPrebuiltVoiceConfig `json:"voiceConfig"`
}

// APIPrebuiltVoiceConfig names a prebuilt TTS voice.
type APIPrebuiltVoiceConfig struct {
	VoiceName string `json:"voiceName"`
}

// APISafetySetting configures a harm-category threshold.
type APISafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold,omitempty"`
}

// APITool is a heterogeneous tools[] entry. The wire shape is one of:
//
//	{ functionDeclarations: [...] }
//	{ googleSearch: {...} }
//	{ urlContext: {} }
//	{ codeExecution: {} }
//	{ fileSearch: {...} }
//	{ googleMaps: {} }
//	{ enterpriseWebSearch: {} }
//	{ retrieval: { vertex_rag_store: {...} } }
//
// Body is serialised / deserialised verbatim via custom MarshalJSON / UnmarshalJSON.
type APITool struct {
	Body map[string]any `json:"-"`
}

// MarshalJSON emits Body as the top-level JSON object.
func (t APITool) MarshalJSON() ([]byte, error) {
	if t.Body == nil {
		return []byte("{}"), nil
	}
	return json.Marshal(t.Body)
}

// UnmarshalJSON populates Body from the top-level JSON object.
func (t *APITool) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &t.Body)
}

// APIToolConfig carries tool-call configuration.
type APIToolConfig struct {
	FunctionCallingConfig            *APIFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
	RetrievalConfig                  *APIRetrievalConfig       `json:"retrievalConfig,omitempty"`
	IncludeServerSideToolInvocations *bool                     `json:"includeServerSideToolInvocations,omitempty"`
}

// APIFunctionCallingConfig controls function-calling mode.
type APIFunctionCallingConfig struct {
	Mode                        string   `json:"mode,omitempty"` // "AUTO" | "ANY" | "NONE" | "VALIDATED"
	AllowedFunctionNames        []string `json:"allowedFunctionNames,omitempty"`
	StreamFunctionCallArguments *bool    `json:"streamFunctionCallArguments,omitempty"`
}

// APIRetrievalConfig carries retrieval configuration (e.g. Maps latLng).
type APIRetrievalConfig struct {
	LatLng *APILatLng `json:"latLng,omitempty"`
}

// APILatLng is a geographic coordinate pair.
type APILatLng struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// ---- Response envelopes ----

// APIGenerateContentResponse is the wire shape of the :generateContent response.
type APIGenerateContentResponse struct {
	Candidates     []APICandidate     `json:"candidates,omitempty"`
	PromptFeedback *APIPromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *APIUsageMetadata  `json:"usageMetadata,omitempty"`
	ModelVersion   string             `json:"modelVersion,omitempty"`
	ResponseID     string             `json:"responseId,omitempty"`
}

// APICandidate is a single response candidate.
type APICandidate struct {
	Content            APIContent             `json:"content"`
	FinishReason       string                 `json:"finishReason,omitempty"`
	SafetyRatings      []APISafetyRating      `json:"safetyRatings,omitempty"`
	CitationMetadata   *APICitationMetadata   `json:"citationMetadata,omitempty"`
	GroundingMetadata  *APIGroundingMetadata  `json:"groundingMetadata,omitempty"`
	UrlContextMetadata *APIURLContextMetadata `json:"urlContextMetadata,omitempty"`
	Index              int                    `json:"index"`
}

// APIPromptFeedback carries prompt-level safety feedback.
type APIPromptFeedback struct {
	BlockReason   string            `json:"blockReason,omitempty"`
	SafetyRatings []APISafetyRating `json:"safetyRatings,omitempty"`
}

// APISafetyRating carries a per-category safety rating.
type APISafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

// APICitationMetadata carries citation source metadata.
type APICitationMetadata struct {
	CitationSources []APICitationSource `json:"citationSources,omitempty"`
}

// APICitationSource describes a single citation.
type APICitationSource struct {
	StartIndex *int   `json:"startIndex,omitempty"`
	EndIndex   *int   `json:"endIndex,omitempty"`
	URI        string `json:"uri,omitempty"`
	Title      string `json:"title,omitempty"`
	License    string `json:"license,omitempty"`
}

// APIGroundingMetadata carries grounding (web-search) metadata.
type APIGroundingMetadata struct {
	WebSearchQueries   []string              `json:"webSearchQueries,omitempty"`
	ImageSearchQueries []string              `json:"imageSearchQueries,omitempty"`
	RetrievalQueries   []string              `json:"retrievalQueries,omitempty"`
	SearchEntryPoint   *APISearchEntryPoint  `json:"searchEntryPoint,omitempty"`
	GroundingChunks    []APIGroundingChunk   `json:"groundingChunks,omitempty"`
	GroundingSupports  []APIGroundingSupport `json:"groundingSupports,omitempty"`
	RetrievalMetadata  *APIRetrievalMetadata `json:"retrievalMetadata,omitempty"`
}

// APISearchEntryPoint carries the search entry point rendered snippet.
type APISearchEntryPoint struct {
	RenderedContent string `json:"renderedContent,omitempty"`
}

// APIGroundingChunk is a single grounding source (web, image, retrieved context, or maps).
type APIGroundingChunk struct {
	Web              *APIWebChunk       `json:"web,omitempty"`
	Image            *APIImageChunk     `json:"image,omitempty"`
	RetrievedContext *APIRetrievedChunk `json:"retrievedContext,omitempty"`
	Maps             *APIMapsChunk      `json:"maps,omitempty"`
}

// APIWebChunk is a web grounding chunk.
type APIWebChunk struct {
	URI   string `json:"uri,omitempty"`
	Title string `json:"title,omitempty"`
}

// APIImageChunk is an image grounding chunk.
type APIImageChunk struct {
	SourceURI string `json:"sourceUri,omitempty"`
	ImageURI  string `json:"imageUri,omitempty"`
	Title     string `json:"title,omitempty"`
	Domain    string `json:"domain,omitempty"`
}

// APIRetrievedChunk is a retrieved-context grounding chunk.
type APIRetrievedChunk struct {
	URI             string `json:"uri,omitempty"`
	Title           string `json:"title,omitempty"`
	Text            string `json:"text,omitempty"`
	FileSearchStore string `json:"fileSearchStore,omitempty"`
}

// APIMapsChunk is a Maps grounding chunk.
type APIMapsChunk struct {
	URI     string `json:"uri,omitempty"`
	Title   string `json:"title,omitempty"`
	Text    string `json:"text,omitempty"`
	PlaceID string `json:"placeId,omitempty"`
}

// APIGroundingSupport maps content segments to grounding chunks.
type APIGroundingSupport struct {
	Segment               *APISegment `json:"segment,omitempty"`
	GroundingChunkIndices []int       `json:"groundingChunkIndices,omitempty"`
	ConfidenceScores      []float64   `json:"confidenceScores,omitempty"`
}

// APISegment describes a text segment by start/end index.
type APISegment struct {
	StartIndex int `json:"startIndex"`
	EndIndex   int `json:"endIndex"`
}

// APIRetrievalMetadata carries retrieval performance metadata.
type APIRetrievalMetadata struct {
	GoogleSearchDynamicRetrievalScore float64 `json:"googleSearchDynamicRetrievalScore,omitempty"`
}

// APIURLContextMetadata carries URL-context tool metadata.
type APIURLContextMetadata struct {
	URLMetadata []string `json:"urlMetadata,omitempty"`
}

// APIUsageMetadata carries token usage from the response.
type APIUsageMetadata struct {
	PromptTokenCount        int                     `json:"promptTokenCount"`
	CachedContentTokenCount int                     `json:"cachedContentTokenCount"`
	CandidatesTokenCount    int                     `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int                     `json:"thoughtsTokenCount"`
	TotalTokenCount         int                     `json:"totalTokenCount"`
	PromptTokensDetails     []APIModalityTokenCount `json:"promptTokensDetails,omitempty"`
	CandidatesTokensDetails []APIModalityTokenCount `json:"candidatesTokensDetails,omitempty"`
	TrafficType             string                  `json:"trafficType,omitempty"`
}

// APIModalityTokenCount carries per-modality token count.
type APIModalityTokenCount struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

// ---- Embedding envelopes ----

// APIEmbedContentRequest is the wire body for :embedContent (single value).
type APIEmbedContentRequest struct {
	Model                string     `json:"model"`
	Content              APIContent `json:"content"`
	OutputDimensionality *int       `json:"outputDimensionality,omitempty"`
	TaskType             string     `json:"taskType,omitempty"`
}

// APIBatchEmbedContentsRequest is the wire body for :batchEmbedContents.
type APIBatchEmbedContentsRequest struct {
	Requests []APIEmbedContentRequest `json:"requests"`
}

// APIEmbedContentResponse is the wire response for :embedContent.
type APIEmbedContentResponse struct {
	Embedding APIEmbeddingValues `json:"embedding"`
}

// APIBatchEmbedContentsResponse is the wire response for :batchEmbedContents.
type APIBatchEmbedContentsResponse struct {
	Embeddings []APIEmbeddingValues `json:"embeddings"`
}

// APIEmbeddingValues carries a single embedding vector.
type APIEmbeddingValues struct {
	Values []float64 `json:"values"`
}

// ---- Imagen / Veo / Files envelopes (stubs for later milestones) ----

// APIImagenRequest is the :predict request body for Imagen models.
type APIImagenRequest struct {
	Instances  []map[string]any `json:"instances"`
	Parameters map[string]any   `json:"parameters,omitempty"`
}

// APIImagenResponse is the :predict response for Imagen models.
type APIImagenResponse struct {
	Predictions []APIImagenPrediction `json:"predictions,omitempty"`
}

// APIImagenPrediction carries one generated image (base64).
type APIImagenPrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
}

// APIVideoOperationRequest is the :predictLongRunning request body.
type APIVideoOperationRequest struct {
	Instances  []map[string]any `json:"instances"`
	Parameters map[string]any   `json:"parameters,omitempty"`
}

// APIVideoOperationResponse is the LRO response shape for video operations.
// Wire shape: { "done": bool, "name": string, "metadata": any, "response": { "predictions": [...] } }
type APIVideoOperationResponse struct {
	Done     bool                     `json:"done"`
	Name     string                   `json:"name"`
	Metadata map[string]any           `json:"metadata,omitempty"`
	Response *APIVideoOperationResult `json:"response,omitempty"`
}

// APIVideoOperationResult wraps the predictions array inside a completed LRO.
type APIVideoOperationResult struct {
	Predictions []APIVideoPrediction `json:"predictions,omitempty"`
}

// APIVideoPrediction carries one generated video.
type APIVideoPrediction struct {
	BytesBase64Encoded string      `json:"bytesBase64Encoded,omitempty"`
	Video              APIVideoURI `json:"video,omitempty"`
}

// APIVideoURI carries a video download URI.
type APIVideoURI struct {
	URI string `json:"uri,omitempty"`
}

// APIFileResponse is the file metadata returned by the Files API.
type APIFileResponse struct {
	File APIFileMetadata `json:"file"`
}

// APIFileMetadata carries file metadata from the Files API.
type APIFileMetadata struct {
	Name           string `json:"name"`
	DisplayName    string `json:"displayName,omitempty"`
	MimeType       string `json:"mimeType,omitempty"`
	SizeBytes      string `json:"sizeBytes,omitempty"`
	URI            string `json:"uri,omitempty"`
	State          string `json:"state,omitempty"` // "PROCESSING" | "ACTIVE" | "FAILED"
	CreateTime     string `json:"createTime,omitempty"`
	UpdateTime     string `json:"updateTime,omitempty"`
	ExpirationTime string `json:"expirationTime,omitempty"`
	SHA256Hash     string `json:"sha256Hash,omitempty"`
}
