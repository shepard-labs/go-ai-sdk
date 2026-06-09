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
	warnings := []Warning{}
	if opts.N > 1 {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "OpenRouter returns one image per image generation call"})
	}
	if opts.Size != "" {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "Use AspectRatio for OpenRouter image sizing"})
	}
	var content any = opts.Prompt
	if len(opts.InputFiles) > 0 {
		parts := make([]apiPart, 0, len(opts.InputFiles)+1)
		for _, f := range opts.InputFiles {
			p, err := imageInputPart(f)
			if err != nil {
				return nil, err
			}
			parts = append(parts, p)
		}
		parts = append(parts, apiPart{Type: "text", Text: opts.Prompt})
		content = parts
	}
	body := map[string]any{"model": m.modelID, "messages": []map[string]any{{"role": "user", "content": content}}, "modalities": []string{"image", "text"}}
	if opts.AspectRatio != "" {
		body["image_config"] = map[string]any{"aspect_ratio": opts.AspectRatio}
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
	mergeBody(body, m.provider.extraBody)
	mergeBody(body, m.options.ExtraBody)
	mergeOpenRouterOptions(body, opts.ProviderOptions)
	var resp chatResponse
	_, h, err := m.provider.postJSON(ctx, "/chat/completions", body, opts.Headers, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, NoContentGeneratedError{Message: "OpenRouter returned no image choices"}
	}
	images := []string{}
	for _, img := range resp.Choices[0].Message.Images {
		_, data := dataURLParts(img.ImageURL.URL, "image/jpeg")
		images = append(images, data)
	}
	rb, _ := json.Marshal(body)
	var usage *ImageUsage
	if resp.Usage != nil {
		usage = &ImageUsage{InputTokens: resp.Usage.PromptTokens, OutputTokens: resp.Usage.CompletionTokens, TotalTokens: resp.Usage.TotalTokens}
	}
	return &ImageGenerateResult{Images: images, Warnings: warnings, Usage: usage, Request: RequestMetadata{Body: rb}, Response: ImageResponseMetadata{ID: resp.ID, ModelID: resp.Model, Headers: h, Timestamp: time.Now()}}, nil
}

func imageInputPart(f FileContent) (apiPart, error) {
	url, media := fileURL(f)
	if !strings.HasPrefix(url, "data:") && !strings.HasPrefix(url, "http") {
		if b, ok := f.Data.([]byte); ok {
			if media == "" {
				media = "image/png"
			}
			url = "data:" + media + ";base64," + base64.StdEncoding.EncodeToString(b)
		}
	}
	return apiPart{Type: "image_url", ImageURL: &apiURLPayload{URL: url}}, nil
}
