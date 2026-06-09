package openrouter

type DataCollection string
type Quantization string
type ProviderSort string

const (
	DataCollectionAllow DataCollection = "allow"
	DataCollectionDeny  DataCollection = "deny"

	QuantizationInt4    Quantization = "int4"
	QuantizationInt8    Quantization = "int8"
	QuantizationFP4     Quantization = "fp4"
	QuantizationFP6     Quantization = "fp6"
	QuantizationFP8     Quantization = "fp8"
	QuantizationFP16    Quantization = "fp16"
	QuantizationBF16    Quantization = "bf16"
	QuantizationFP32    Quantization = "fp32"
	QuantizationUnknown Quantization = "unknown"

	ProviderSortPrice      ProviderSort = "price"
	ProviderSortThroughput ProviderSort = "throughput"
	ProviderSortLatency    ProviderSort = "latency"
)

type ProviderRouting struct {
	Order             []string       `json:"order,omitempty"`
	AllowFallbacks    *bool          `json:"allow_fallbacks,omitempty"`
	RequireParameters *bool          `json:"require_parameters,omitempty"`
	DataCollection    DataCollection `json:"data_collection,omitempty"`
	Only              []string       `json:"only,omitempty"`
	Ignore            []string       `json:"ignore,omitempty"`
	Quantizations     []Quantization `json:"quantizations,omitempty"`
	Sort              ProviderSort   `json:"sort,omitempty"`
	MaxPrice          *MaxPrice      `json:"max_price,omitempty"`
	ZDR               *bool          `json:"zdr,omitempty"`
}

type EmbeddingProviderRouting struct {
	Order             []string       `json:"order,omitempty"`
	AllowFallbacks    *bool          `json:"allow_fallbacks,omitempty"`
	RequireParameters *bool          `json:"require_parameters,omitempty"`
	DataCollection    DataCollection `json:"data_collection,omitempty"`
	Only              []string       `json:"only,omitempty"`
	Ignore            []string       `json:"ignore,omitempty"`
	Sort              ProviderSort   `json:"sort,omitempty"`
	MaxPrice          *MaxPrice      `json:"max_price,omitempty"`
}

type ImageProviderRouting = EmbeddingProviderRouting

type MaxPrice struct {
	Prompt     any `json:"prompt,omitempty"`
	Completion any `json:"completion,omitempty"`
	Image      any `json:"image,omitempty"`
	Audio      any `json:"audio,omitempty"`
	Request    any `json:"request,omitempty"`
}

type ReasoningDetail struct {
	Type      string `json:"type"`
	Summary   string `json:"summary,omitempty"`
	Data      string `json:"data,omitempty"`
	Text      string `json:"text,omitempty"`
	Signature string `json:"signature,omitempty"`
	ID        string `json:"id,omitempty"`
	Format    string `json:"format,omitempty"`
	Index     *int   `json:"index,omitempty"`
}

type FileAnnotation struct {
	Type     string `json:"type,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
}

type OpenRouterUsageAccounting struct {
	PromptTokens          int     `json:"promptTokens,omitempty"`
	CachedTokens          int     `json:"cachedTokens,omitempty"`
	CacheWriteTokens      int     `json:"cacheWriteTokens,omitempty"`
	CompletionTokens      int     `json:"completionTokens,omitempty"`
	ReasoningTokens       int     `json:"reasoningTokens,omitempty"`
	TotalTokens           int     `json:"totalTokens,omitempty"`
	Cost                  float64 `json:"cost,omitempty"`
	UpstreamInferenceCost float64 `json:"upstreamInferenceCost,omitempty"`
	Raw                   any     `json:"raw,omitempty"`
}
