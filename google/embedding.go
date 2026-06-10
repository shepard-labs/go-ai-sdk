package google

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// googleEmbeddingModel implements the Google Generative AI embedding model.
//
// It dispatches between :embedContent (single value) and :batchEmbedContents
// (multiple values) endpoints and supports per-value multimodal content
// overrides via providerOptions.google.content.
type googleEmbeddingModel struct {
	provider *googleProvider
	modelID  string
}

func (m *googleEmbeddingModel) ModelID() string  { return m.modelID }
func (m *googleEmbeddingModel) Provider() string { return m.provider.name + ".embedding" }

func (m *googleEmbeddingModel) MaxEmbeddingsPerCall() int {
	return m.provider.maxEmbeddingsPerCall
}

func (m *googleEmbeddingModel) SupportsParallelCalls() bool {
	return m.provider.supportsParallelEmbeddingCalls
}

// DoEmbed sends one or more text values to the embedding endpoint and returns
// the resulting vectors.
//
// When len(opts.Values) == 1, the :embedContent endpoint is used; otherwise
// the :batchEmbedContents endpoint is used. An error is returned when the
// caller supplies more values than MaxEmbeddingsPerCall allows, or when
// opts.ProviderOptions["google"].content is set and its length does not
// match len(opts.Values).
func (m *googleEmbeddingModel) DoEmbed(ctx context.Context, opts EmbedOptions) (*EmbedResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	if len(opts.Values) > m.MaxEmbeddingsPerCall() {
		return nil, TooManyEmbeddingValuesForCallError{
			Provider:             m.Provider(),
			ModelID:              m.modelID,
			MaxEmbeddingsPerCall: m.MaxEmbeddingsPerCall(),
			Values:               len(opts.Values),
		}
	}

	rawOpts := cloneProviderOptions(opts.ProviderOptions)
	embOpts := embeddingOptionsFromProviderOptions(rawOpts)
	if embOpts.Content != nil && len(embOpts.Content) != len(opts.Values) {
		return nil, InvalidPromptError{
			Message: fmt.Sprintf("providerOptions.google.content length (%d) must match values length (%d)",
				len(embOpts.Content), len(opts.Values)),
		}
	}

	perCallHeaders := cloneHeader(opts.Headers)

	if len(opts.Values) == 1 {
		body, err := convertEmbeddingRequest(m.modelID, opts.Values[0], embOpts)
		if err != nil {
			return nil, err
		}
		resp, err := m.provider.executeJSON(ctx, m.embedContentPath(), body, perCallHeaders)
		if err != nil {
			return nil, err
		}
		embeddings, metadata, err := parseEmbeddingResponse(resp.Body, true)
		if err != nil {
			return nil, err
		}
		return &EmbedResult{
			Embeddings:       embeddings,
			Usage:            nil,
			Warnings:         nil,
			ProviderMetadata: metadata,
			Request:          RequestMetadata{Body: append([]byte(nil), body...)},
			Response:         ResponseMetadata{Headers: cloneHeader(resp.Headers), Body: append([]byte(nil), resp.Body...)},
		}, nil
	}

	body, err := convertBatchEmbeddingRequest(m.modelID, opts.Values, embOpts)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, m.batchEmbedContentsPath(), body, perCallHeaders)
	if err != nil {
		return nil, err
	}
	embeddings, metadata, err := parseEmbeddingResponse(resp.Body, false)
	if err != nil {
		return nil, err
	}
	return &EmbedResult{
		Embeddings:       embeddings,
		Usage:            nil,
		Warnings:         nil,
		ProviderMetadata: metadata,
		Request:          RequestMetadata{Body: append([]byte(nil), body...)},
		Response:         ResponseMetadata{Headers: cloneHeader(resp.Headers), Body: append([]byte(nil), resp.Body...)},
	}, nil
}

// embedContentPath returns the :embedContent endpoint path for this model.
func (m *googleEmbeddingModel) embedContentPath() string {
	return "/" + getModelPath(m.modelID) + ":embedContent"
}

// batchEmbedContentsPath returns the :batchEmbedContents endpoint path for this model.
func (m *googleEmbeddingModel) batchEmbedContentsPath() string {
	return "/" + getModelPath(m.modelID) + ":batchEmbedContents"
}

// convertEmbeddingRequest builds the request body for the :embedContent
// endpoint (single value).
func convertEmbeddingRequest(modelID, value string, opts EmbeddingModelOptions) ([]byte, error) {
	content, err := buildEmbedContent(value, opts.Content, 0)
	if err != nil {
		return nil, err
	}
	req := internal.APIEmbedContentRequest{
		Model:   "models/" + modelID,
		Content: content,
	}
	if opts.OutputDimensionality != nil {
		req.OutputDimensionality = opts.OutputDimensionality
	}
	if opts.TaskType != "" {
		req.TaskType = opts.TaskType
	}
	return json.Marshal(req)
}

// convertBatchEmbeddingRequest builds the request body for the
// :batchEmbedContents endpoint (multiple values).
func convertBatchEmbeddingRequest(modelID string, values []string, opts EmbeddingModelOptions) ([]byte, error) {
	requests := make([]internal.APIEmbedContentRequest, len(values))
	for i, v := range values {
		var contentParts [][]ContentPart
		if opts.Content != nil {
			contentParts = opts.Content
		}
		content, err := buildEmbedContent(v, contentParts, i)
		if err != nil {
			return nil, err
		}
		req := internal.APIEmbedContentRequest{
			Model:   "models/" + modelID,
			Content: content,
		}
		if opts.OutputDimensionality != nil {
			req.OutputDimensionality = opts.OutputDimensionality
		}
		if opts.TaskType != "" {
			req.TaskType = opts.TaskType
		}
		requests[i] = req
	}
	return json.Marshal(internal.APIBatchEmbedContentsRequest{Requests: requests})
}

// buildEmbedContent builds an APIContent for an embedding value. The text
// value is always prepended to the parts list. When perValueParts is set
// and perValueParts[i] is non-nil, its parts are appended after the text part.
func buildEmbedContent(value string, perValueParts [][]ContentPart, index int) (internal.APIContent, error) {
	parts := make([]internal.APIPart, 0, 1)
	parts = append(parts, internal.APIPart{Text: value})
	if perValueParts != nil {
		if index < 0 || index >= len(perValueParts) {
			return internal.APIContent{}, fmt.Errorf("embedding content index %d out of range (len %d)", index, len(perValueParts))
		}
		extra := perValueParts[index]
		for _, p := range extra {
			if wire := convertContentPart(p); wire != nil {
				parts = append(parts, *wire)
			}
		}
	}
	return internal.APIContent{Parts: parts}, nil
}

// convertContentPart converts a ContentPart into an APIPart. A nil pointer is
// returned for parts that have no wire representation (e.g. a ContentPart with
// no fields set); callers should skip those.
func convertContentPart(p ContentPart) *internal.APIPart {
	if p.InlineData != nil {
		return &internal.APIPart{InlineData: &internal.APIInlineData{
			MimeType: p.InlineData.MimeType,
			Data:     p.InlineData.Data,
		}}
	}
	if p.FileData != nil {
		return &internal.APIPart{FileData: &internal.APIFileData{
			MimeType: p.FileData.MimeType,
			FileURI:  p.FileData.FileURI,
		}}
	}
	if p.Text != "" {
		return &internal.APIPart{Text: p.Text}
	}
	return nil
}

// parseEmbeddingResponse decodes a Google embedding response body. When
// single is true, the response is expected to be a single embedding
// { "embedding": { "values": [...] } }; when false, a batch
// { "embeddings": [{ "values": [...] }, ...] }.
func parseEmbeddingResponse(body []byte, single bool) ([][]float64, ProviderMetadata, error) {
	if single {
		var decoded internal.APIEmbedContentResponse
		if err := json.Unmarshal(body, &decoded); err != nil {
			return nil, nil, InvalidResponseDataError{Message: "failed to decode :embedContent response", Data: body}
		}
		embeddings := [][]float64{append([]float64(nil), decoded.Embedding.Values...)}
		return embeddings, nil, nil
	}
	var decoded internal.APIBatchEmbedContentsResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, nil, InvalidResponseDataError{Message: "failed to decode :batchEmbedContents response", Data: body}
	}
	embeddings := make([][]float64, len(decoded.Embeddings))
	for i, item := range decoded.Embeddings {
		embeddings[i] = append([]float64(nil), item.Values...)
	}
	return embeddings, nil, nil
}

// embeddingOptionsFromProviderOptions extracts EmbeddingModelOptions from
// the "google" ProviderOptions key.
func embeddingOptionsFromProviderOptions(opts ProviderOptions) EmbeddingModelOptions {
	if opts == nil {
		return EmbeddingModelOptions{}
	}
	raw, ok := opts[ProviderName]
	if !ok {
		return EmbeddingModelOptions{}
	}
	var out EmbeddingModelOptions
	if v, ok := raw["outputDimensionality"]; ok {
		if n, ok := toInt(v); ok {
			out.OutputDimensionality = &n
		}
	}
	if v, ok := raw["taskType"]; ok {
		if s, ok := v.(string); ok {
			out.TaskType = s
		}
	}
	if v, ok := raw["content"]; ok {
		if parts, ok := toContentPartMatrix(v); ok {
			out.Content = parts
		}
	}
	return out
}

// toInt coerces a provider-options value to int. Returns (n, false) when the
// value is not an integer-shaped number.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int8:
		return int(n), true
	case int16:
		return int(n), true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case uint:
		return int(n), true
	case uint8:
		return int(n), true
	case uint16:
		return int(n), true
	case uint32:
		return int(n), true
	case uint64:
		return int(n), true
	case float64:
		return int(n), true
	case *int:
		if n == nil {
			return 0, false
		}
		return *n, true
	}
	return 0, false
}

// toContentPartMatrix coerces a provider-options value to [][]ContentPart.
// The shape is a 2-level slice: each outer entry is the per-value parts list
// for one input value, and may be nil (text-only). Each inner entry is a
// ContentPart.
func toContentPartMatrix(v any) ([][]ContentPart, bool) {
	outer, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([][]ContentPart, len(outer))
	for i, elem := range outer {
		if elem == nil {
			out[i] = nil
			continue
		}
		inner, ok := elem.([]any)
		if !ok {
			return nil, false
		}
		parts := make([]ContentPart, 0, len(inner))
		for _, item := range inner {
			p, ok := toContentPart(item)
			if !ok {
				return nil, false
			}
			parts = append(parts, p)
		}
		out[i] = parts
	}
	return out, true
}

// toContentPart coerces a provider-options value to a ContentPart. Accepted
// shapes:
//
//	{ "text": "..." }
//	{ "inlineData": { "mimeType": "...", "data": "..." } }
//	{ "fileData":   { "mimeType": "...", "fileUri": "..." } }
func toContentPart(v any) (ContentPart, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return ContentPart{}, false
	}
	var p ContentPart
	if s, ok := stringFromMap(m, "text"); ok {
		p.Text = s
	}
	if raw, ok := m["inlineData"]; ok {
		if id, ok := toInlineDataPart(raw); ok {
			p.InlineData = &id
		}
	}
	if raw, ok := m["fileData"]; ok {
		if fd, ok := toFileDataPart(raw); ok {
			p.FileData = &fd
		}
	}
	return p, true
}

func toInlineDataPart(v any) (InlineDataPart, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return InlineDataPart{}, false
	}
	mime, _ := stringFromMap(m, "mimeType")
	data, _ := stringFromMap(m, "data")
	return InlineDataPart{MimeType: mime, Data: data}, true
}

func toFileDataPart(v any) (FileDataPart, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return FileDataPart{}, false
	}
	mime, _ := stringFromMap(m, "mimeType")
	uri, _ := stringFromMap(m, "fileUri")
	return FileDataPart{MimeType: mime, FileURI: uri}, true
}

func stringFromMap(m map[string]any, key string) (string, bool) {
	v, ok := m[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// cloneProviderOptions returns a deep copy of a ProviderOptions map.
func cloneProviderOptions(opts ProviderOptions) ProviderOptions {
	if opts == nil {
		return nil
	}
	out := make(ProviderOptions, len(opts))
	for key, values := range opts {
		cloned := make(map[string]any, len(values))
		for k, v := range values {
			cloned[k] = v
		}
		out[key] = cloned
	}
	return out
}
