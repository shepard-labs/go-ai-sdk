package openai

import (
	"context"
	"encoding/json"
	"mime/multipart"
	"strconv"
)

// openaiFilesClient implements Files.
type openaiFilesClient struct {
	provider *openaiProvider
}

func (c *openaiFilesClient) UploadFile(ctx context.Context, opts FilesUploadOptions) (*FilesUploadResult, error) {
	if c.provider.err != nil {
		return nil, c.provider.err
	}
	purpose := "assistants"
	var expiresAfterSeconds *int
	if v, ok := opts.ProviderOptions["openai"]; ok {
		if p, ok := v["purpose"].(string); ok && p != "" {
			purpose = p
		}
		if e, ok := v["expiresAfter"].(float64); ok {
			secs := int(e)
			expiresAfterSeconds = &secs
		} else if e, ok := v["expiresAfter"].(int); ok {
			expiresAfterSeconds = &e
		}
	}
	build := func(w *multipart.Writer) error {
		if err := w.WriteField("purpose", purpose); err != nil {
			return err
		}
		if expiresAfterSeconds != nil {
			if err := w.WriteField("expires_after[anchor]", "created_at"); err != nil {
				return err
			}
			if err := w.WriteField("expires_after[seconds]", strconv.Itoa(*expiresAfterSeconds)); err != nil {
				return err
			}
		}
		filename := opts.Filename
		if filename == "" {
			filename = "file"
		}
		part, err := w.CreateFormFile("file", filename)
		if err != nil {
			return err
		}
		_, err = part.Write(opts.Data)
		return err
	}
	resp, err := c.provider.executeMultipart(ctx, endpointFiles, nil, build)
	if err != nil {
		return nil, err
	}
	var raw filesAPIResponse
	if err := json.Unmarshal(resp.Body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse files response: " + err.Error()}
	}
	return mapFilesResponse(raw, opts.Filename, opts.MediaType), nil
}

type filesAPIResponse struct {
	ID        string  `json:"id"`
	Object    *string `json:"object"`
	Bytes     *int    `json:"bytes"`
	CreatedAt *int64  `json:"created_at"`
	Filename  *string `json:"filename"`
	Purpose   *string `json:"purpose"`
	Status    *string `json:"status"`
	ExpiresAt *int64  `json:"expires_at"`
}

func mapFilesResponse(raw filesAPIResponse, filename, mediaType string) *FilesUploadResult {
	result := &FilesUploadResult{
		ProviderReference: ProviderReference{"openai": raw.ID},
		ProviderMetadata: ProviderMetadata{
			"openai": map[string]any{},
		},
		Warnings: []Warning{},
	}
	if raw.Filename != nil {
		result.Filename = *raw.Filename
	} else if filename != "" {
		result.Filename = filename
	}
	if mediaType != "" {
		result.MediaType = mediaType
	}
	pm := result.ProviderMetadata["openai"].(map[string]any)
	if raw.Filename != nil {
		pm["filename"] = *raw.Filename
	}
	if raw.Purpose != nil {
		pm["purpose"] = *raw.Purpose
	}
	if raw.Bytes != nil {
		pm["bytes"] = *raw.Bytes
	}
	if raw.CreatedAt != nil {
		pm["createdAt"] = *raw.CreatedAt
	}
	if raw.Status != nil {
		pm["status"] = *raw.Status
	}
	if raw.ExpiresAt != nil {
		pm["expiresAt"] = *raw.ExpiresAt
	}
	return result
}
