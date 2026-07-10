package openrouter

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"time"
)

type imageModel struct {
	provider *openRouterProvider
	modelID  string
	options  ImageOptions
}

func (m *imageModel) ModelID() string       { return m.modelID }
func (m *imageModel) Provider() string      { return "openrouter.image" }
func (m *imageModel) MaxImagesPerCall() int { return defaultMaxImages }

func (m *imageModel) DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	if opts.Mask != nil {
		return nil, UnsupportedFunctionalityError{Message: "OpenRouter image generation does not support masks"}
	}
	body := map[string]any{"model": m.modelID, "prompt": opts.Prompt}
	if opts.N > 0 {
		body["n"] = opts.N
	}
	if opts.Size != "" {
		body["size"] = opts.Size
	}
	if opts.AspectRatio != "" {
		body["aspect_ratio"] = opts.AspectRatio
	}
	if opts.Quality != "" {
		body["quality"] = opts.Quality
	}
	if opts.Background != "" {
		body["background"] = opts.Background
	}
	if opts.OutputCompression != nil {
		body["output_compression"] = *opts.OutputCompression
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if m.options.User != "" {
		body["user"] = m.options.User
	}
	if m.options.Provider != nil {
		body["provider"] = m.options.Provider
	}
	if len(opts.InputFiles) > 0 {
		references := make([]map[string]any, 0, len(opts.InputFiles))
		for _, f := range opts.InputFiles {
			reference, err := imageInputReference(f)
			if err != nil {
				return nil, err
			}
			references = append(references, reference)
		}
		body["input_references"] = references
	}
	mergeBody(body, m.provider.extraBody)
	mergeBody(body, m.options.ExtraBody)
	mergeOpenRouterOptions(body, opts.ProviderOptions)
	var resp imageResponse
	_, h, err := m.provider.postJSONOnce(ctx, "/images", body, opts.Headers, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 {
		return nil, NoContentGeneratedError{Message: "OpenRouter returned no images"}
	}
	images := []string{}
	imageData := make([]ImageData, 0, len(resp.Data))
	for _, img := range resp.Data {
		if strings.TrimSpace(img.Base64) == "" {
			continue
		}
		images = append(images, img.Base64)
		imageData = append(imageData, ImageData{Base64: img.Base64, MediaType: img.MediaType})
	}
	if len(images) == 0 {
		return nil, NoContentGeneratedError{Message: "OpenRouter returned no usable images"}
	}
	rb, _ := json.Marshal(body)
	var usage *ImageUsage
	cost := 0.0
	if resp.Usage != nil {
		usage = &ImageUsage{InputTokens: resp.Usage.PromptTokens, OutputTokens: resp.Usage.CompletionTokens, TotalTokens: resp.Usage.TotalTokens}
		cost = resp.Usage.Cost
	}
	metadata := ProviderMetadata{"openrouter": map[string]any{"created": resp.Created, "cost": cost}}
	return &ImageGenerateResult{Images: images, ImageData: imageData, Usage: usage, ProviderMetadata: metadata, Request: RequestMetadata{Body: rb}, Response: ImageResponseMetadata{ID: resp.ID, ModelID: resp.Model, Headers: h, Timestamp: time.Now()}}, nil
}

func imageInputReference(f FileContent) (map[string]any, error) {
	url, media := fileURL(f)
	if !strings.HasPrefix(url, "data:") && !strings.HasPrefix(url, "http") {
		if b, ok := f.Data.([]byte); ok {
			if media == "" {
				media = "image/png"
			}
			url = "data:" + media + ";base64," + base64.StdEncoding.EncodeToString(b)
		}
	}
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}, nil
}

type imageResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Data    []struct {
		Base64    string `json:"b64_json"`
		MediaType string `json:"media_type"`
	} `json:"data"`
	Usage *struct {
		PromptTokens     int     `json:"prompt_tokens"`
		CompletionTokens int     `json:"completion_tokens"`
		TotalTokens      int     `json:"total_tokens"`
		Cost             float64 `json:"cost"`
	} `json:"usage"`
}
