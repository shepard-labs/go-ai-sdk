package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// googleImageModel implements the Google Generative AI image model interface.
//
// DoGenerate dispatches on model ID: gemini-*-image / nano-banana* delegates
// to a LanguageModel with responseModalities: ["IMAGE"]; everything else goes
// to the Imagen :predict endpoint.
type googleImageModel struct {
	provider *googleProvider
	modelID  string
	settings ImageModelSettings
}

// ModelID returns the model's ID string.
func (m *googleImageModel) ModelID() string { return m.modelID }

// Provider returns the provider name suffix.
func (m *googleImageModel) Provider() string { return m.provider.name + ".image" }

// MaxImagesPerCall returns the maximum number of images per call (10 for
// Gemini image, 4 for Imagen). An explicit [ImageModelSettings.MaxImagesPerCall]
// overrides the model-family default.
func (m *googleImageModel) MaxImagesPerCall() int {
	if m.settings.MaxImagesPerCall != nil {
		return *m.settings.MaxImagesPerCall
	}
	if isGeminiImageModel(m.modelID) {
		return 10
	}
	return 4
}

// DoGenerate performs an image generation call. It dispatches to the
// Gemini-image path or the Imagen path based on the model ID.
func (m *googleImageModel) DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	if isGeminiImageModel(m.modelID) {
		return m.doGenerateGemini(ctx, opts)
	}
	return m.doGenerateImagen(ctx, opts)
}

// ---- Gemini image path ----

// doGenerateGemini delegates to a [LanguageModel] with responseModalities set
// to ["IMAGE"] and extracts image data from the model's inlineData parts.
func (m *googleImageModel) doGenerateGemini(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	if opts.Mask != nil {
		return nil, UnsupportedFunctionalityError{
			Functionality: "image editing via mask is not supported for Gemini image models",
		}
	}
	if opts.N > 1 {
		return nil, UnsupportedFunctionalityError{
			Functionality: "Gemini image models return at most one image per call",
		}
	}

	var warnings []Warning
	if opts.Size != "" {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "size",
			Details: "Gemini image models do not accept a size parameter; use aspectRatio or providerOptions.google.imageConfig.imageSize instead.",
		})
	}

	// Parse recognized image-model provider options.
	rawOpts := cloneProviderOptions(opts.ProviderOptions)
	vertexLike := isVertexLike(m.provider.baseURL, m.provider.useVertexAIHeaders, rawOpts)
	merged := mergeGoogleNamespaces(rawOpts, vertexLike)
	imageOpts, recognizedKeys := imageModelOptionsFromProviderOptions(merged)

	// Build the LanguageModel call. The "google" namespace in the provider
	// options is rewritten to add responseModalities and imageConfig (aspect
	// ratio) and to strip the image-model-only recognized keys.
	passthroughGoogle := cloneMap(merged)
	if passthroughGoogle == nil {
		passthroughGoogle = map[string]any{}
	}
	for _, k := range recognizedKeys {
		delete(passthroughGoogle, k)
	}
	passthroughGoogle["responseModalities"] = []string{"IMAGE"}
	aspectRatio := opts.AspectRatio
	if aspectRatio == "" {
		aspectRatio = imageOpts.AspectRatio
	}
	if aspectRatio != "" {
		passthroughGoogle["imageConfig"] = map[string]any{"aspectRatio": aspectRatio}
	}

	lmProviderOptions := ProviderOptions{}
	if vertexLike {
		if v, ok := rawOpts["googleVertex"]; ok {
			lmProviderOptions["googleVertex"] = cloneMap(v)
		}
		if v, ok := rawOpts["vertex"]; ok {
			lmProviderOptions["vertex"] = cloneMap(v)
		}
	}
	lmProviderOptions["google"] = passthroughGoogle

	lmOpts := GenerateOptions{
		Headers:         cloneHeader(opts.Headers),
		ProviderOptions: lmProviderOptions,
	}

	// Build user prompt from the prompt text plus any file inputs.
	userContents, fileWarnings, err := buildImagePromptContents(opts)
	if err != nil {
		return nil, err
	}
	warnings = append(warnings, fileWarnings...)
	lmOpts.Messages = []Message{UserMessage{Content: userContents}}

	// Forward the seed field (the language model call accepts it).
	if opts.Seed != nil {
		lmOpts.Seed = opts.Seed
	}

	// Wire up googleSearch as a provider tool when set.
	if imageOpts.GoogleSearch != nil {
		lmOpts.Tools = []Tool{{
			Type:       "provider",
			ID:         "google.google_search",
			Name:       "google_search",
			ArgsSchema: *imageOpts.GoogleSearch,
		}}
	}

	// Instantiate a language model and call DoGenerate.
	lm := m.provider.languageModel(m.modelID)
	result, err := lm.DoGenerate(ctx, lmOpts)
	if err != nil {
		return nil, err
	}
	warnings = append(warnings, result.Warnings...)

	images, imageMetadata := extractImageInlineData(result.Content)

	providerMetadata := result.ProviderMetadata
	if providerMetadata == nil {
		providerMetadata = ProviderMetadata{}
	}
	gm, ok := providerMetadata["google"].(map[string]any)
	if !ok {
		gm = map[string]any{}
		providerMetadata["google"] = gm
	}
	imageMetas := make([]map[string]any, len(imageMetadata))
	for i, entry := range imageMetadata {
		if entry == nil {
			imageMetas[i] = map[string]any{}
			continue
		}
		imageMetas[i] = entry
	}
	gm["images"] = imageMetas

	return &ImageGenerateResult{
		Images:           images,
		Warnings:         warnings,
		ProviderMetadata: providerMetadata,
		Request:          result.Request,
		Response:         result.Response,
	}, nil
}

// buildImagePromptContents assembles the user prompt's content parts from
// the text prompt and the input files. URL files become image fileData parts
// with mediaType "image/*"; data/base64 files become image inlineData parts.
func buildImagePromptContents(opts ImageGenerateOptions) ([]UserContent, []Warning, error) {
	var contents []UserContent
	if opts.Prompt != "" {
		contents = append(contents, TextContent{Text: opts.Prompt})
	}
	for _, f := range opts.Files {
		uc, err := imageFileToUserContent(f)
		if err != nil {
			return nil, nil, err
		}
		contents = append(contents, uc)
	}
	return contents, nil, nil
}

// imageFileToUserContent converts an [ImageFile] into a [UserContent] for
// embedding into the LanguageModel prompt. URL/reference files become
// [ImageContent] with fileData (mime "image/*"); data files become
// [ImageContent] with inlineData (base64 of the bytes).
func imageFileToUserContent(f ImageFile) (UserContent, error) {
	mediaType := f.MediaType
	if mediaType == "" {
		mediaType = "image/*"
	}
	switch f.Type {
	case "url", "reference":
		return ImageContent{
			Source: ImageSource{Type: f.Type, MediaType: mediaType, URL: f.URL},
		}, nil
	case "data":
		return ImageContent{
			Source: ImageSource{Type: "data", MediaType: mediaType, Data: base64.StdEncoding.EncodeToString(f.Data)},
		}, nil
	case "base64":
		// Validate the base64 (matches the openaicompatible check).
		if _, err := base64.StdEncoding.DecodeString(f.Base64); err != nil {
			return nil, InvalidPromptError{Message: "invalid base64 image file: " + err.Error()}
		}
		return ImageContent{
			Source: ImageSource{Type: "data", MediaType: mediaType, Data: f.Base64},
		}, nil
	default:
		return nil, UnsupportedFunctionalityError{
			Functionality: "unsupported image file type " + f.Type,
		}
	}
}

// extractImageInlineData walks a slice of [Content] and returns the base64
// payload of every inline-data part whose media type starts with "image/".
// The returned metadata slice mirrors the image list, one entry per image,
// carrying the image's mediaType. Parts without an image media type are
// skipped.
func extractImageInlineData(contents []Content) ([]string, []map[string]any) {
	var images []string
	var metas []map[string]any
	for _, c := range contents {
		img, ok := c.(ImageContent)
		if !ok {
			continue
		}
		if img.Source.Type != "data" {
			continue
		}
		if !strings.HasPrefix(img.Source.MediaType, "image/") {
			continue
		}
		images = append(images, img.Source.Data)
		metas = append(metas, map[string]any{"mimeType": img.Source.MediaType})
	}
	return images, metas
}

// imageModelOptionsFromProviderOptions parses the typed [ImageModelOptions]
// view out of the merged google provider-options namespace. The list of
// recognized keys is returned so callers can strip them before forwarding
// the remainder to the language model.
func imageModelOptionsFromProviderOptions(merged map[string]any) (ImageModelOptions, []string) {
	out := ImageModelOptions{}
	var recognized []string
	if v, ok := merged["personGeneration"]; ok {
		if s, ok := v.(string); ok {
			out.PersonGeneration = s
			recognized = append(recognized, "personGeneration")
		}
	}
	if v, ok := merged["aspectRatio"]; ok {
		if s, ok := v.(string); ok {
			out.AspectRatio = s
			recognized = append(recognized, "aspectRatio")
		}
	}
	if v, ok := merged["googleSearch"]; ok {
		if args, ok := googleSearchArgsFromAny(v); ok {
			out.GoogleSearch = &args
			recognized = append(recognized, "googleSearch")
		}
	}
	return out, recognized
}

// googleSearchArgsFromAny coerces an arbitrary value to [GoogleSearchArgs].
// Returns (zero, false) when the value is not a map[string]any.
func googleSearchArgsFromAny(v any) (GoogleSearchArgs, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return GoogleSearchArgs{}, false
	}
	var out GoogleSearchArgs
	if raw, ok := m["searchTypes"]; ok {
		if sm, ok := raw.(map[string]any); ok {
			t := &GoogleSearchTypes{}
			if w, ok := sm["webSearch"].(map[string]any); ok {
				t.WebSearch = w
			}
			if i, ok := sm["imageSearch"].(map[string]any); ok {
				t.ImageSearch = i
			}
			if t.WebSearch != nil || t.ImageSearch != nil {
				out.SearchTypes = t
			}
		}
	}
	if raw, ok := m["timeRangeFilter"]; ok {
		if tm, ok := raw.(map[string]any); ok {
			f := &TimeRangeFilter{}
			if s, ok := tm["startTime"].(string); ok {
				f.StartTime = s
			}
			if s, ok := tm["endTime"].(string); ok {
				f.EndTime = s
			}
			if f.StartTime != "" || f.EndTime != "" {
				out.TimeRangeFilter = f
			}
		}
	}
	return out, true
}

// cloneMap returns a shallow copy of m. Returns nil for nil input.
func cloneMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// ---- Imagen path ----

// doGenerateImagen calls the :predict endpoint with the instances/parameters
// body shape. Editing (files / mask) is rejected; size and seed are warned;
// googleSearch is warned and stripped.
func (m *googleImageModel) doGenerateImagen(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	if len(opts.Files) > 0 {
		return nil, UnsupportedFunctionalityError{
			Functionality: "image editing (files) is not supported for Imagen; use the @ai-sdk/google-vertex provider",
		}
	}
	if opts.Mask != nil {
		return nil, UnsupportedFunctionalityError{
			Functionality: "image editing (mask) is not supported for Imagen; use the @ai-sdk/google-vertex provider",
		}
	}

	var warnings []Warning
	if opts.Size != "" {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "size",
			Details: "Imagen does not accept a size parameter; use aspectRatio or providerOptions.google.imageSize instead.",
		})
	}
	if opts.Seed != nil {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "seed",
		})
	}

	// Parse recognized image-model provider options.
	rawOpts := cloneProviderOptions(opts.ProviderOptions)
	vertexLike := isVertexLike(m.provider.baseURL, m.provider.useVertexAIHeaders, rawOpts)
	merged := mergeGoogleNamespaces(rawOpts, vertexLike)
	imageOpts, recognizedKeys := imageModelOptionsFromProviderOptions(merged)
	if imageOpts.GoogleSearch != nil {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "googleSearch",
			Details: "Google Search grounding is only supported on Gemini image models.",
		})
		recognizedKeys = append(recognizedKeys, "googleSearch")
	}

	// Build the parameters object.
	parameters := map[string]any{}
	if opts.N > 0 {
		parameters["sampleCount"] = opts.N
	}
	aspectRatio := opts.AspectRatio
	if aspectRatio == "" {
		aspectRatio = imageOpts.AspectRatio
	}
	if aspectRatio != "" {
		parameters["aspectRatio"] = aspectRatio
	}
	if imageOpts.PersonGeneration != "" {
		parameters["personGeneration"] = imageOpts.PersonGeneration
	}
	// Spread the rest of the google namespace into parameters, minus
	// the image-model-recognized keys and the structural keys we manage
	// ourselves.
	recognized := map[string]struct{}{}
	for _, k := range recognizedKeys {
		recognized[k] = struct{}{}
	}
	for k, v := range merged {
		if _, ok := recognized[k]; ok {
			continue
		}
		if isImagenBodyReservedKey(k) {
			continue
		}
		parameters[k] = v
	}

	body := internal.APIImagenRequest{
		Instances:  []map[string]any{{"prompt": opts.Prompt}},
		Parameters: parameters,
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	resp, err := m.provider.executeJSON(ctx, "/"+getModelPath(m.modelID)+":predict", bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}

	images, err := parsePredictImageResponse(resp.Body)
	if err != nil {
		return nil, err
	}

	imageMetas := make([]map[string]any, len(images))
	for i := range images {
		imageMetas[i] = map[string]any{}
	}

	return &ImageGenerateResult{
		Images:   images,
		Warnings: warnings,
		ProviderMetadata: ProviderMetadata{
			"google": map[string]any{
				"images": imageMetas,
			},
		},
		Request:  RequestMetadata{Body: append([]byte(nil), bodyBytes...)},
		Response: responseMetadata(resp.Headers, append([]byte(nil), resp.Body...), "", m.modelID),
	}, nil
}

// isImagenBodyReservedKey reports whether k is a key the wire body manages
// explicitly and that must not be spread from passthrough. (Most Imagen
// passthrough keys — language, negativePrompt, addWatermark, etc. — are fine
// to forward as-is.)
func isImagenBodyReservedKey(k string) bool {
	switch k {
	case "prompt", "sampleCount", "aspectRatio", "instances", "parameters":
		return true
	}
	return false
}

// parsePredictImageResponse decodes the :predict response for Imagen.
// Wire shape: { "predictions": [{ "bytesBase64Encoded": <string> }] }.
func parsePredictImageResponse(body []byte) ([]string, error) {
	var decoded internal.APIImagenResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to decode :predict response", Data: string(body)}
	}
	images := make([]string, len(decoded.Predictions))
	for i, p := range decoded.Predictions {
		images[i] = p.BytesBase64Encoded
	}
	return images, nil
}

// isGeminiImageModel reports whether the model ID belongs to the Gemini image
// family (vs. Imagen). Matches `gemini-.*-image`, `nano-banana*` prefixes.
func isGeminiImageModel(modelID string) bool {
	if hasPrefix(modelID, "gemini-") && strings.Contains(modelID, "image") {
		return true
	}
	if hasPrefix(modelID, "nano-banana") {
		return true
	}
	return false
}

// hasPrefix reports whether s begins with prefix. Reimplemented locally to
// keep the image model self-contained (used only by isGeminiImageModel).
func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}
