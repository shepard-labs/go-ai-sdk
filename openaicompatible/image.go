package openaicompatible

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strconv"
	"time"
)

var imageNow = time.Now

type imageResponse struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
	} `json:"data"`
}

type imageBlob struct {
	FieldName string
	Data      []byte
	MediaType string
}

func (m *openAICompatibleImageModel) DoGenerate(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	if len(opts.Files) == 0 {
		return m.doGenerateImage(ctx, opts)
	}
	return m.doEditImage(ctx, opts)
}

func (m *openAICompatibleImageModel) doGenerateImage(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	providerOptions := mergeImageProviderOptions(m.provider.name, cloneProviderOptions(opts.ProviderOptions))
	bodyMap := map[string]any{
		"model":           m.modelID,
		"prompt":          opts.Prompt,
		"n":               opts.N,
		"response_format": "b64_json",
	}
	if opts.Size != "" {
		bodyMap["size"] = opts.Size
	}
	for k, v := range providerOptions {
		if v != nil {
			bodyMap[k] = v
		}
	}
	body, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointImageGenerations, body, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}
	return m.parseImageResponse(resp, RequestMetadata{Body: append([]byte(nil), body...)}, imageWarnings(opts))
}

func (m *openAICompatibleImageModel) doEditImage(ctx context.Context, opts ImageGenerateOptions) (*ImageGenerateResult, error) {
	providerOptions := mergeImageProviderOptions(m.provider.name, cloneProviderOptions(opts.ProviderOptions))
	fields, formFields, err := imageEditFields(m.modelID, opts, providerOptions)
	if err != nil {
		return nil, err
	}
	blobs, err := m.imageBlobs(ctx, opts)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeMultipart(ctx, endpointImageEdits, cloneHeader(opts.Headers), func(w *multipart.Writer) error {
		for _, field := range fields {
			if err := w.WriteField(field.name, field.value); err != nil {
				return err
			}
		}
		for _, blob := range blobs {
			if err := writeImagePart(w, blob); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return m.parseImageResponse(resp, RequestMetadata{FormFields: formFields}, imageWarnings(opts))
}

func (m *openAICompatibleImageModel) parseImageResponse(resp *apiResponse, request RequestMetadata, warnings []Warning) (*ImageGenerateResult, error) {
	var decoded imageResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, err
	}
	images := make([]string, len(decoded.Data))
	for i, item := range decoded.Data {
		images[i] = item.B64JSON
	}
	return &ImageGenerateResult{
		Images:   images,
		Warnings: warnings,
		Request:  request,
		Response: ImageResponseMetadata{Timestamp: imageNow(), ModelID: m.modelID, Headers: cloneHeader(resp.Headers)},
	}, nil
}

func imageWarnings(opts ImageGenerateOptions) []Warning {
	var warnings []Warning
	if opts.AspectRatio != "" {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "aspectRatio", Details: "This model does not support aspect ratio. Use size instead."})
	}
	if opts.Seed != nil {
		warnings = append(warnings, Warning{Type: "unsupported", Feature: "seed"})
	}
	return warnings
}

type imageField struct{ name, value string }

func imageEditFields(modelID string, opts ImageGenerateOptions, providerOptions map[string]any) ([]imageField, map[string][]string, error) {
	var fields []imageField
	formFields := map[string][]string{}
	add := func(name, value string) {
		fields = append(fields, imageField{name: name, value: value})
		formFields[name] = append(formFields[name], value)
	}
	add("model", modelID)
	if opts.Prompt != "" {
		add("prompt", opts.Prompt)
	}
	add("n", strconv.Itoa(opts.N))
	if opts.Size != "" {
		add("size", opts.Size)
	}
	for k, v := range providerOptions {
		value, ok, err := formFieldValue(v)
		if err != nil {
			return nil, nil, err
		}
		if ok {
			add(k, value)
		}
	}
	return fields, formFields, nil
}

func formFieldValue(v any) (string, bool, error) {
	if v == nil {
		return "", false, nil
	}
	switch value := v.(type) {
	case string:
		return value, true, nil
	case bool:
		return strconv.FormatBool(value), true, nil
	case int:
		return strconv.Itoa(value), true, nil
	case int8:
		return strconv.FormatInt(int64(value), 10), true, nil
	case int16:
		return strconv.FormatInt(int64(value), 10), true, nil
	case int32:
		return strconv.FormatInt(int64(value), 10), true, nil
	case int64:
		return strconv.FormatInt(value, 10), true, nil
	case uint:
		return strconv.FormatUint(uint64(value), 10), true, nil
	case uint8:
		return strconv.FormatUint(uint64(value), 10), true, nil
	case uint16:
		return strconv.FormatUint(uint64(value), 10), true, nil
	case uint32:
		return strconv.FormatUint(uint64(value), 10), true, nil
	case uint64:
		return strconv.FormatUint(value, 10), true, nil
	case float32:
		return strconv.FormatFloat(float64(value), 'f', -1, 32), true, nil
	case float64:
		return strconv.FormatFloat(value, 'f', -1, 64), true, nil
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return "", false, err
		}
		return string(encoded), true, nil
	}
}

func (m *openAICompatibleImageModel) imageBlobs(ctx context.Context, opts ImageGenerateOptions) ([]imageBlob, error) {
	blobs := make([]imageBlob, 0, len(opts.Files)+1)
	for _, file := range opts.Files {
		data, mediaType, err := m.convertImageFile(ctx, file)
		if err != nil {
			return nil, err
		}
		blobs = append(blobs, imageBlob{FieldName: "image", Data: data, MediaType: mediaType})
	}
	if opts.Mask != nil {
		data, mediaType, err := m.convertImageFile(ctx, *opts.Mask)
		if err != nil {
			return nil, err
		}
		blobs = append(blobs, imageBlob{FieldName: "mask", Data: data, MediaType: mediaType})
	}
	return blobs, nil
}

func (m *openAICompatibleImageModel) convertImageFile(ctx context.Context, file ImageFile) ([]byte, string, error) {
	switch file.Type {
	case "bytes":
		return append([]byte(nil), file.Data...), file.MediaType, nil
	case "base64":
		data, err := base64.StdEncoding.DecodeString(file.Base64)
		if err != nil {
			return nil, "", err
		}
		return data, file.MediaType, nil
	case "url":
		data, mediaType, err := m.provider.downloadImageURL(ctx, file.URL)
		if err != nil {
			return nil, "", err
		}
		return data, mediaType, nil
	default:
		return nil, "", fmt.Errorf("openaicompatible: unsupported image file type %q", file.Type)
	}
}

func writeImagePart(w *multipart.Writer, blob imageBlob) error {
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, blob.FieldName, blob.FieldName))
	if blob.MediaType != "" {
		header.Set("Content-Type", blob.MediaType)
	}
	part, err := w.CreatePart(header)
	if err != nil {
		return err
	}
	_, err = part.Write(blob.Data)
	return err
}

func (p *openAICompatibleProvider) executeMultipart(ctx context.Context, path string, perCall http.Header, build func(*multipart.Writer) error) (*apiResponse, error) {
	if p.err != nil {
		return nil, p.err
	}
	requestURL, err := p.requestURL(path)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 0; attempt <= p.retry.MaxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var body bytes.Buffer
		writer := multipart.NewWriter(&body)
		if err := build(writer); err != nil {
			return nil, err
		}
		if err := writer.Close(); err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body.Bytes()))
		if err != nil {
			return nil, err
		}
		for k, values := range p.headersForCall(perCall) {
			for _, v := range values {
				req.Header.Add(k, v)
			}
		}
		req.Header.Set("Content-Type", writer.FormDataContentType())
		if p.generateID != nil {
			req.Header.Set("x-request-id", p.generateID())
		}
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if !shouldRetry(resp, err) || attempt == p.retry.MaxRetries {
				return nil, err
			}
			if waitErr := p.waitBeforeRetry(ctx, attempt, resp, err); waitErr != nil {
				return nil, waitErr
			}
			continue
		}
		if resp == nil {
			lastErr = &APICallError{Message: "fetcher returned nil response and nil error", Retryable: true}
			if attempt == p.retry.MaxRetries {
				return nil, lastErr
			}
			if waitErr := p.waitBeforeRetry(ctx, attempt, nil, lastErr); waitErr != nil {
				return nil, waitErr
			}
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			apiErr := p.readAPIError(resp)
			lastErr = apiErr
			if apiErr.Retryable && attempt < p.retry.MaxRetries {
				if waitErr := p.waitBeforeRetry(ctx, attempt, resp, apiErr); waitErr != nil {
					return nil, waitErr
				}
				continue
			}
			return nil, apiErr
		}
		return p.readResponse(resp)
	}
	return nil, lastErr
}

func (p *openAICompatibleProvider) downloadImageURL(ctx context.Context, imageURL string) ([]byte, string, error) {
	var lastErr error
	for attempt := 0; attempt <= p.retry.MaxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, "", err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
		if err != nil {
			return nil, "", err
		}
		if ua := p.headers.Get("User-Agent"); ua != "" {
			req.Header.Set("User-Agent", ua)
		}
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if !shouldRetry(resp, err) || attempt == p.retry.MaxRetries {
				return nil, "", err
			}
			if waitErr := p.waitBeforeRetry(ctx, attempt, resp, err); waitErr != nil {
				return nil, "", waitErr
			}
			continue
		}
		if resp == nil {
			lastErr = &APICallError{Message: "fetcher returned nil response and nil error", Retryable: true}
			if attempt == p.retry.MaxRetries {
				return nil, "", lastErr
			}
			if waitErr := p.waitBeforeRetry(ctx, attempt, nil, lastErr); waitErr != nil {
				return nil, "", waitErr
			}
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			apiErr := p.readAPIError(resp)
			lastErr = apiErr
			if apiErr.Retryable && attempt < p.retry.MaxRetries {
				if waitErr := p.waitBeforeRetry(ctx, attempt, resp, apiErr); waitErr != nil {
					return nil, "", waitErr
				}
				continue
			}
			return nil, "", apiErr
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, "", err
		}
		return body, resp.Header.Get("Content-Type"), nil
	}
	return nil, "", lastErr
}

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
