package openai

import (
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
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

// Upload reads the full content from r and delegates to UploadFile so both
// paths share the same multipart encoding (purpose default, expires_after
// brackets, filename fallback).
func (c *openaiFilesClient) Upload(ctx context.Context, r io.Reader, filename, mediaType string, providerOptions ProviderOptions) (*FilesUploadResult, error) {
	if c.provider.err != nil {
		return nil, c.provider.err
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return c.UploadFile(ctx, FilesUploadOptions{
		Data:            data,
		MediaType:       mediaType,
		Filename:        filename,
		ProviderOptions: providerOptions,
	})
}

// GetFileMetadata retrieves a file object (GET /files/{id}). The response
// shares the upload response shape, so it maps through mapFilesResponse.
func (c *openaiFilesClient) GetFileMetadata(ctx context.Context, fileID string) (*FilesUploadResult, error) {
	if c.provider.err != nil {
		return nil, c.provider.err
	}
	if fileID == "" {
		return nil, InvalidPromptError{Message: "fileID must not be empty"}
	}
	resp, err := c.provider.executeBuffered(ctx, http.MethodGet, filesFilePath(fileID), nil, nil)
	if err != nil {
		return nil, err
	}
	var raw filesAPIResponse
	if err := json.Unmarshal(resp.Body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse files response: " + err.Error()}
	}
	return mapFilesResponse(raw, "", ""), nil
}

// DownloadFile downloads file content (GET /files/{id}/content) and
// returns the raw bytes.
func (c *openaiFilesClient) DownloadFile(ctx context.Context, fileID string) ([]byte, error) {
	if c.provider.err != nil {
		return nil, c.provider.err
	}
	if fileID == "" {
		return nil, InvalidPromptError{Message: "fileID must not be empty"}
	}
	resp, err := c.provider.executeBuffered(ctx, http.MethodGet, filesFilePath(fileID)+"/content", nil, nil)
	if err != nil {
		return nil, err
	}
	return resp.Body, nil
}

// DeleteFile deletes a file (DELETE /files/{id}) and reports the server's
// deleted flag.
func (c *openaiFilesClient) DeleteFile(ctx context.Context, fileID string) (bool, error) {
	if c.provider.err != nil {
		return false, c.provider.err
	}
	if fileID == "" {
		return false, InvalidPromptError{Message: "fileID must not be empty"}
	}
	resp, err := c.provider.executeBuffered(ctx, http.MethodDelete, filesFilePath(fileID), nil, nil)
	if err != nil {
		return false, err
	}
	var raw struct {
		ID      string `json:"id"`
		Deleted bool   `json:"deleted"`
	}
	if err := json.Unmarshal(resp.Body, &raw); err != nil {
		return false, InvalidResponseDataError{Message: "failed to parse files response: " + err.Error()}
	}
	return raw.Deleted, nil
}

// filesFilePath escapes a file ID into its /files/{id} path.
func filesFilePath(fileID string) string {
	return endpointFiles + "/" + url.PathEscape(fileID)
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
