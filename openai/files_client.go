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
	purpose, expiresAfterSeconds := filesUploadFields(opts.ProviderOptions)
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

// copyWithContext copies src to dst in chunks, aborting promptly when ctx
// is done so a cancelled Upload never leaves the pipe writer blocked
// forever. A blocked pipe write already fails once the pipe is closed
// with the ctx error; this covers the copy itself between chunks.
func copyWithContext(ctx context.Context, dst io.Writer, src io.Reader) (int64, error) {
	buf := make([]byte, 32*1024)
	var total int64
	for {
		select {
		case <-ctx.Done():
			return total, ctx.Err()
		default:
		}
		n, rerr := src.Read(buf)
		if n > 0 {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return total, werr
			}
			total += int64(n)
		}
		if rerr != nil {
			if rerr == io.EOF {
				return total, nil
			}
			return total, rerr
		}
	}
}

// filesUploadFields extracts the shared upload form fields (purpose
// default, expires_after brackets) from provider options.
func filesUploadFields(providerOptions ProviderOptions) (string, *int) {
	purpose := "assistants"
	var expiresAfterSeconds *int
	if v, ok := providerOptions["openai"]; ok {
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
	return purpose, expiresAfterSeconds
}

// Upload streams the multipart body via io.Pipe (chunked, no full
// buffering of the reader) and maps the response through the same shape
// as UploadFile. The UploadFile path stays fully buffered (and retryable);
// the streamed body here can only be consumed once, so it is not retried.
func (c *openaiFilesClient) Upload(ctx context.Context, r io.Reader, filename, mediaType string, providerOptions ProviderOptions) (*FilesUploadResult, error) {
	if c.provider.err != nil {
		return nil, c.provider.err
	}
	purpose, expiresAfterSeconds := filesUploadFields(providerOptions)
	if filename == "" {
		filename = "file"
	}
	pr, pw := io.Pipe()
	writer := multipart.NewWriter(pw)
	// Abort the pipe if ctx is cancelled while the writer is blocked on
	// it (e.g. fetch failed early and no one drains pr anymore), so the
	// writer goroutine below never blocks forever.
	stopWatch := context.AfterFunc(ctx, func() {
		_ = pw.CloseWithError(ctx.Err())
	})
	defer stopWatch()
	go func() {
		err := func() error {
			if err := writer.WriteField("purpose", purpose); err != nil {
				return err
			}
			if expiresAfterSeconds != nil {
				if err := writer.WriteField("expires_after[anchor]", "created_at"); err != nil {
					return err
				}
				if err := writer.WriteField("expires_after[seconds]", strconv.Itoa(*expiresAfterSeconds)); err != nil {
					return err
				}
			}
			part, err := writer.CreateFormFile("file", filename)
			if err != nil {
				return err
			}
			if _, err := copyWithContext(ctx, part, r); err != nil {
				return err
			}
			return writer.Close()
		}()
		_ = pw.CloseWithError(err)
	}()
	resp, err := c.provider.executeMultipartReader(ctx, endpointFiles, nil, pr, writer.FormDataContentType())
	if err != nil {
		// No one will drain pr now; release a writer blocked on the pipe
		// so it terminates instead of leaking.
		_ = pr.CloseWithError(err)
		return nil, err
	}
	var raw filesAPIResponse
	if err := json.Unmarshal(resp.Body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse files response: " + err.Error()}
	}
	return mapFilesResponse(raw, filename, mediaType), nil
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
