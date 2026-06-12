package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// openaiImageModel implements ImageModel.
type openaiImageModel struct {
	provider *openaiProvider
	modelID  string
}

func newImageModel(p *openaiProvider, modelID string) ImageModel {
	return &openaiImageModel{provider: p, modelID: modelID}
}

func (m *openaiImageModel) ModelID() string  { return m.modelID }
func (m *openaiImageModel) Provider() string { return "openai.image" }

// MaxImagesPerCall is per-model: dall-e-3 = 1, others = 10.
func (m *openaiImageModel) MaxImagesPerCall() int {
	if m.modelID == "dall-e-3" {
		return 1
	}
	return 10
}

func (m *openaiImageModel) DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	if len(opts.Files) > 0 {
		return m.doEdit(ctx, opts)
	}
	return m.doGenerateImages(ctx, opts)
}

// hasDefaultResponseFormat returns true for model families that always
// return b64_json and ignore the response_format field.
func hasDefaultResponseFormat(modelID string) bool {
	prefixes := []string{
		"chatgpt-image-",
		"gpt-image-1-mini",
		"gpt-image-1.5",
		"gpt-image-1",
		"gpt-image-2",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(modelID, p) {
			return true
		}
	}
	return false
}

func (m *openaiImageModel) doGenerateImages(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	body := map[string]any{
		"model":  m.modelID,
		"prompt": opts.Prompt,
		"n":      1,
	}
	if opts.Size != "" {
		body["size"] = opts.Size
	}
	if !hasDefaultResponseFormat(m.modelID) {
		body["response_format"] = "b64_json"
	}
	var warnings []Warning
	if v, ok := opts.ProviderOptions["openai"]; ok {
		if _, has := v["aspectRatio"]; has {
			warnings = append(warnings, Warning{Type: "unsupported", Feature: "aspectRatio", Message: "aspectRatio is not supported by the OpenAI image API"})
		}
		if _, has := v["seed"]; has {
			warnings = append(warnings, Warning{Type: "unsupported", Feature: "seed", Message: "seed is not supported by the OpenAI image API"})
		}

		if n, ok := v["n"]; ok {
			body["n"] = n
		}
		// Per-model fields. The openai model layer is the source of
		// truth for which fields apply to which model.
		if quality, ok := v["quality"].(string); ok && quality != "" {
			if err := validateImageQuality(m.modelID, quality); err != nil {
				return nil, err
			}
			body["quality"] = quality
		}
		if style, ok := v["style"].(string); ok && style != "" {
			if m.modelID != "dall-e-3" {
				return nil, InvalidPromptError{Message: "style is only supported for dall-e-3"}
			}
			body["style"] = style
		}
		if background, ok := v["background"].(string); ok && background != "" {
			if !strings.HasPrefix(m.modelID, "gpt-image-") {
				return nil, InvalidPromptError{Message: "background is only supported for gpt-image-*"}
			}
			body["background"] = background
		}
		if outputFormat, ok := v["outputFormat"].(string); ok && outputFormat != "" {
			if !strings.HasPrefix(m.modelID, "gpt-image-") {
				return nil, InvalidPromptError{Message: "outputFormat is only supported for gpt-image-*"}
			}
			body["output_format"] = outputFormat
		}
		if oc, ok := v["outputCompression"].(int); ok {
			if !strings.HasPrefix(m.modelID, "gpt-image-") {
				return nil, InvalidPromptError{Message: "outputCompression is only supported for gpt-image-*"}
			}
			body["output_compression"] = oc
		}
		if mod, ok := v["moderation"].(string); ok && mod != "" {
			if !strings.HasPrefix(m.modelID, "gpt-image-") {
				return nil, InvalidPromptError{Message: "moderation is only supported for gpt-image-*"}
			}
			body["moderation"] = mod
		}
		if user, ok := v["user"].(string); ok {
			body["user"] = user
		}
	}
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointImageGenerations, encoded, opts.Headers)
	if err != nil {
		return nil, err
	}
	result, err := m.parseImageResponse(resp.Body, encoded)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(result.Warnings, warnings...)
	return result, nil
}

func (m *openaiImageModel) doEdit(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	if len(opts.Files) == 0 {
		return nil, InvalidPromptError{Message: "image edit requires at least one file in opts.Files"}
	}
	build := func(w *multipart.Writer) error {
		if err := w.WriteField("model", m.modelID); err != nil {
			return err
		}
		if err := w.WriteField("prompt", opts.Prompt); err != nil {
			return err
		}
		if err := w.WriteField("n", "1"); err != nil {
			return err
		}
		if opts.Size != "" {
			if err := w.WriteField("size", opts.Size); err != nil {
				return err
			}
		}
		if !hasDefaultResponseFormat(m.modelID) {
			if err := w.WriteField("response_format", "b64_json"); err != nil {
				return err
			}
		}
		for i, f := range opts.Files {
			data, name, _ := imageFileData(f)
			formField := "image"
			if i > 0 {
				formField = "image[]"
			}
			part, err := w.CreateFormFile(formField, name)
			if err != nil {
				return err
			}
			if _, err := part.Write(data); err != nil {
				return err
			}
		}
		if opts.Mask != nil {
			data, name, _ := imageFileData(*opts.Mask)
			part, err := w.CreateFormFile("mask", name)
			if err != nil {
				return err
			}
			if _, err := part.Write(data); err != nil {
				return err
			}
		}
		// inputFidelity is edits-only (gpt-image-*).
		if v, ok := opts.ProviderOptions["openai"]; ok {
			if fidelity, ok := v["inputFidelity"].(string); ok && fidelity != "" {
				if !strings.HasPrefix(m.modelID, "gpt-image-") {
					return InvalidPromptError{Message: "inputFidelity is only supported for gpt-image-*"}
				}
				if err := w.WriteField("input_fidelity", fidelity); err != nil {
					return err
				}
			}
		}
		return nil
	}
	resp, err := m.provider.executeMultipart(ctx, endpointImageEdits, opts.Headers, build)
	if err != nil {
		return nil, err
	}
	return m.parseImageResponse(resp.Body, nil)
}

// validateImageQuality enforces per-model quality enum.
func validateImageQuality(modelID, q string) error {
	switch {
	case modelID == "dall-e-3":
		if q != "standard" && q != "hd" {
			return InvalidPromptError{Message: fmt.Sprintf("dall-e-3 quality must be standard|hd, got %q", q)}
		}
	case modelID == "dall-e-2":
		if q != "standard" && q != "hd" {
			return InvalidPromptError{Message: fmt.Sprintf("dall-e-2 quality must be standard|hd, got %q", q)}
		}
	case strings.HasPrefix(modelID, "gpt-image-"):
		switch q {
		case "low", "medium", "high", "auto":
			// ok
		default:
			return InvalidPromptError{Message: fmt.Sprintf("gpt-image-* quality must be low|medium|high|auto, got %q", q)}
		}
	}
	return nil
}

func imageFileData(f openaicompatible.ImageFile) (data []byte, filename, contentType string) {
	switch f.Type {
	case "url":
		return []byte(f.URL), "image", "text/uri-list"
	case "base64":
		if decoded, err := base64.StdEncoding.DecodeString(f.Base64); err == nil {
			return decoded, "image.png", "image/png"
		}
		return []byte(f.Base64), "image.txt", "text/plain"
	default:
		ct := f.MediaType
		if ct == "" {
			ct = "application/octet-stream"
		}
		filename := "image"
		if len(f.Data) > 4 {
			switch {
			case bytes.HasPrefix(f.Data, []byte{0x89, 'P', 'N', 'G'}):
				filename = "image.png"
			case bytes.HasPrefix(f.Data, []byte{0xff, 0xd8}):
				filename = "image.jpg"
			case bytes.HasPrefix(f.Data, []byte("GIF87a")), bytes.HasPrefix(f.Data, []byte("GIF89a")):
				filename = "image.gif"
			case bytes.HasPrefix(f.Data, []byte("RIFF")):
				filename = "image.webp"
			}
		}
		return f.Data, filename, ct
	}
}

func (m *openaiImageModel) parseImageResponse(body, requestBody []byte) (*ImageGenerateResult, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse: " + err.Error()}
	}
	result := &ImageGenerateResult{
		Response: ImageResponseMetadata{ModelID: m.modelID, Headers: http.Header{}},
	}
	if requestBody != nil {
		result.Request = RequestMetadata{Body: requestBody}
	}
	data, _ := raw["data"].([]any)
	images := make([]string, 0, len(data))
	imageMeta := make([]map[string]any, 0, len(data))
	for _, item := range data {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		entry := map[string]any{}
		if rp, ok := obj["revised_prompt"].(string); ok {
			entry["revisedPrompt"] = rp
		}
		if b64, ok := obj["b64_json"].(string); ok {
			decoded, err := base64.StdEncoding.DecodeString(b64)
			if err == nil {
				images = append(images, string(decoded))
			} else {
				images = append(images, b64)
			}
		} else if url, ok := obj["url"].(string); ok {
			images = append(images, url)
		}
		imageMeta = append(imageMeta, entry)
	}
	result.Images = images
	if len(imageMeta) > 0 {
		if result.ProviderMetadata == nil {
			result.ProviderMetadata = ProviderMetadata{"openai": map[string]any{}}
		}
		pm, _ := result.ProviderMetadata["openai"].(map[string]any)
		if pm == nil {
			pm = map[string]any{}
		}
		pm["images"] = imageMeta
		result.ProviderMetadata["openai"] = pm
	}
	// Distribute input_tokens_details.
	if usage, ok := raw["usage"].(map[string]any); ok {
		if details, ok := usage["input_tokens_details"].(map[string]any); ok {
			var imageTok, textTok float64
			if v, ok := details["image_tokens"].(float64); ok {
				imageTok = v
			}
			if v, ok := details["text_tokens"].(float64); ok {
				textTok = v
			}
			n := len(data)
			if n > 0 {
				imagePerImg, textPerImg := distributeTokenDetails(int(imageTok), int(textTok), n)
				// Stash per-image token detail into providerMetadata.openai.images.
				for i, entry := range imageMeta {
					if i < n {
						entry["imageTokens"] = imagePerImg[i]
						entry["textTokens"] = textPerImg[i]
					}
				}
				result.Usage = &ImageUsage{}
				if pt, ok := usage["input_tokens"].(float64); ok {
					result.Usage.InputTokens = int(pt)
				}
				if ot, ok := usage["output_tokens"].(float64); ok {
					result.Usage.OutputTokens = int(ot)
				}
				result.Usage.TotalTokens = result.Usage.InputTokens + result.Usage.OutputTokens
			}
		}
	}
	return result, nil
}

// distributeTokenDetails splits a total image/text token count across N
// images, with the remainder going to the last image so the sum equals
// the original total.
func distributeTokenDetails(totalImageTokens, totalTextTokens, n int) (imagePerImg, textPerImg []int) {
	if n <= 0 {
		return nil, nil
	}
	imagePerImg = make([]int, n)
	textPerImg = make([]int, n)
	imgBase := totalImageTokens / n
	imgRem := totalImageTokens % n
	txtBase := totalTextTokens / n
	txtRem := totalTextTokens % n
	for i := 0; i < n; i++ {
		imagePerImg[i] = imgBase
		textPerImg[i] = txtBase
	}
	if imgRem > 0 {
		imagePerImg[n-1] += imgRem
	}
	if txtRem > 0 {
		textPerImg[n-1] += txtRem
	}
	return imagePerImg, textPerImg
}
