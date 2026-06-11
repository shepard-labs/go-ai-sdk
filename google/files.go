package google

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

const defaultFilesPollIntervalMs = 2000
const defaultFilesPollTimeoutMs = 300_000

// googleFiles implements the Files interface for the Google Generative AI provider.
type googleFiles struct {
	provider *googleProvider
}

// filesUploadOptionsFromProviderOptions parses the typed FilesUploadProviderOptions view
// from the merged google provider-options namespace.
func filesUploadOptionsFromProviderOptions(merged map[string]any) FilesUploadProviderOptions {
	out := FilesUploadProviderOptions{}
	if merged == nil {
		return out
	}
	if v, ok := merged["displayName"].(string); ok {
		out.DisplayName = v
	}
	if v, ok := merged["pollIntervalMs"].(float64); ok {
		ms := int(v)
		out.PollIntervalMs = &ms
	}
	if v, ok := merged["pollTimeoutMs"].(float64); ok {
		ms := int(v)
		out.PollTimeoutMs = &ms
	}
	return out
}

// baseOrigin derives the upload origin by stripping the trailing /v1beta suffix
// from the base URL.
// Upload performs a Google resumable file upload: initiates, uploads bytes,
// and polls until the file is ACTIVE or FAILED.
func (f *googleFiles) Upload(ctx context.Context, data []byte, opts FilesUploadOptions) (*FilesUploadResult, error) {
	if f.provider.err != nil {
		return nil, f.provider.err
	}

	if ctx.Err() != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, &APICallError{
				Message:   "file upload timed out after " + formatMs(defaultFilesPollTimeoutMs),
				Type:      "GOOGLE_FILES_UPLOAD_TIMEOUT",
				Retryable: false,
			}
		}
		return nil, &APICallError{
			Message:   "file upload aborted",
			Type:      "GOOGLE_FILES_UPLOAD_ABORTED",
			Retryable: false,
		}
	}

	rawOpts := cloneProviderOptions(opts.ProviderOptions)
	vertexLike := isVertexLike(f.provider.baseURL, f.provider.useVertexAIHeaders, rawOpts)
	merged := mergeGoogleNamespaces(rawOpts, vertexLike)
	filesOpts := filesUploadOptionsFromProviderOptions(merged)

	// Warn on unsupported filename option.
	if opts.Filename != "" {
		f.provider.logger.Warn("unsupported filename option", "feature", "filename", "details", "filename is not used; displayName is set via providerOptions.google.displayName.")
	}

	mediaType := opts.MediaType
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}

	// ---- Step 1: initiate resumable upload ----
	initPath := "/upload/v1beta/files"
	initBody := map[string]any{}
	if filesOpts.DisplayName != "" {
		initBody["file"] = map[string]any{"display_name": filesOpts.DisplayName}
	}
	initBodyBytes, err := json.Marshal(initBody)
	if err != nil {
		return nil, err
	}

	initHeaders := cloneHeader(opts.Headers)
	initHeaders.Set("X-Goog-Upload-Protocol", "resumable")
	initHeaders.Set("X-Goog-Upload-Command", "start")
	initHeaders.Set("X-Goog-Upload-Header-Content-Length", fmt.Sprintf("%d", len(data)))
	initHeaders.Set("X-Goog-Upload-Header-Content-Type", mediaType)

	resp, err := f.provider.executeBuffered(ctx, http.MethodPost, initPath, initBodyBytes, initHeaders)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, &APICallError{
				Message:   "file upload timed out after " + formatMs(defaultFilesPollTimeoutMs),
				Type:      "GOOGLE_FILES_UPLOAD_TIMEOUT",
				Retryable: false,
			}
		}
		if ctx.Err() == context.Canceled {
			return nil, &APICallError{
				Message:   "file upload aborted",
				Type:      "GOOGLE_FILES_UPLOAD_ABORTED",
				Retryable: false,
			}
		}
		return nil, &APICallError{
			Message:   "files upload init request failed",
			Type:      "GOOGLE_FILES_UPLOAD_ERROR",
			Retryable: false,
			Cause:     err,
		}
	}

	uploadURL := resp.Headers.Get("X-Goog-Upload-Url")
	if uploadURL == "" {
		return nil, &APICallError{
			Message:   "X-Goog-Upload-Url header missing in upload init response",
			Type:      "GOOGLE_FILES_UPLOAD_ERROR",
			Retryable: false,
			Status:    resp.Status,
			Headers:   resp.Headers,
			RequestID: resp.RequestID,
			Body:      resp.Body,
		}
	}

	// ---- Step 2: upload bytes ----
	uploadReq, err := http.NewRequestWithContext(ctx, http.MethodPost, uploadURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	for k, values := range f.provider.headersForCall(cloneHeader(opts.Headers)) {
		for _, v := range values {
			uploadReq.Header.Add(k, v)
		}
	}
	uploadReq.Header.Set("X-Goog-Upload-Offset", "0")
	uploadReq.Header.Set("X-Goog-Upload-Command", "upload, finalize")

	f.provider.logger.Debug("google files upload bytes", "url", uploadURL)
	uploadResp, err := f.provider.fetch.Do(uploadReq)
	if err != nil {
		return nil, &APICallError{
			Message:   "files upload bytes request failed",
			Type:      "GOOGLE_FILES_UPLOAD_ERROR",
			Retryable: false,
			Cause:     err,
		}
	}
	defer uploadResp.Body.Close()

	if uploadResp.StatusCode < 200 || uploadResp.StatusCode >= 300 {
		body, _, _ := readLimited(uploadResp.Body, f.provider.maxErrorResponseBytes)
		return nil, &APICallError{
			Message:   "files upload bytes returned non-success status",
			Type:      "GOOGLE_FILES_UPLOAD_ERROR",
			Retryable: false,
			Status:    uploadResp.StatusCode,
			Headers:   cloneHeader(uploadResp.Header),
			RequestID: uploadResp.Header.Get("x-request-id"),
			Body:      body,
		}
	}

	uploadBody, _, err := readLimited(uploadResp.Body, f.provider.maxResponseBodyBytes)
	if err != nil {
		return nil, err
	}

	var fileResp internal.APIFileResponse
	if err := json.Unmarshal(uploadBody, &fileResp); err != nil {
		return nil, &APICallError{
			Message:   "failed to parse file upload response",
			Type:      "GOOGLE_FILES_UPLOAD_ERROR",
			Retryable: false,
			Status:    uploadResp.StatusCode,
			Headers:   cloneHeader(uploadResp.Header),
			RequestID: uploadResp.Header.Get("x-request-id"),
			Body:      uploadBody,
		}
	}

	// ---- Step 3: poll until ACTIVE or FAILED ----
	if fileResp.File.State == "ACTIVE" {
		return buildFilesResult(&fileResp.File, mediaType), nil
	}
	if fileResp.File.State == "FAILED" {
		return nil, &APICallError{
			Message:   "file upload failed on the server",
			Type:      "GOOGLE_FILES_UPLOAD_FAILED",
			Retryable: false,
		}
	}

	pollInterval := defaultFilesPollIntervalMs
	if filesOpts.PollIntervalMs != nil {
		pollInterval = *filesOpts.PollIntervalMs
	}
	pollTimeout := defaultFilesPollTimeoutMs
	if filesOpts.PollTimeoutMs != nil {
		pollTimeout = *filesOpts.PollTimeoutMs
	}

	deadline := time.NewTimer(time.Duration(pollTimeout) * time.Millisecond)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Duration(pollInterval) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline.C:
			return nil, &APICallError{
				Message:   "file upload timed out after " + formatMs(pollTimeout),
				Type:      "GOOGLE_FILES_UPLOAD_TIMEOUT",
				Retryable: false,
			}
		case <-ctx.Done():
			return nil, &APICallError{
				Message:   "file upload aborted",
				Type:      "GOOGLE_FILES_UPLOAD_ABORTED",
				Retryable: false,
			}
		case <-ticker.C:
			if ctxErr := ctx.Err(); ctxErr != nil {
				if ctxErr == context.DeadlineExceeded {
					return nil, &APICallError{
						Message:   "file upload timed out after " + formatMs(pollTimeout),
						Type:      "GOOGLE_FILES_UPLOAD_TIMEOUT",
						Retryable: false,
					}
				}
				return nil, &APICallError{
					Message:   "file upload aborted",
					Type:      "GOOGLE_FILES_UPLOAD_ABORTED",
					Retryable: false,
				}
			}
			getResp, err := f.provider.executeGet(ctx, "/"+fileResp.File.Name, cloneHeader(opts.Headers))
			if err != nil {
				return nil, err
			}
			var pollResp internal.APIFileResponse
			if err := json.Unmarshal(getResp.Body, &pollResp); err != nil {
				return nil, &APICallError{
					Message:   "failed to parse file status response",
					Type:      "GOOGLE_FILES_UPLOAD_ERROR",
					Retryable: false,
					Status:    getResp.Status,
					Headers:   getResp.Headers,
					RequestID: getResp.RequestID,
					Body:      getResp.Body,
				}
			}
			if pollResp.File.State == "ACTIVE" {
				return buildFilesResult(&pollResp.File, mediaType), nil
			}
			if pollResp.File.State == "FAILED" {
				return nil, &APICallError{
					Message:   "file upload failed on the server",
					Type:      "GOOGLE_FILES_UPLOAD_FAILED",
					Retryable: false,
					Status:    getResp.Status,
					Headers:   getResp.Headers,
					RequestID: getResp.RequestID,
					Body:      getResp.Body,
				}
			}
		case <-ctx.Done():
			return nil, &APICallError{
				Message:   "file upload aborted",
				Type:      "GOOGLE_FILES_UPLOAD_ABORTED",
				Retryable: false,
			}
		}
	}
}

// buildFilesResult assembles the FilesUploadResult from file metadata.
func buildFilesResult(file *internal.APIFileMetadata, mediaType string) *FilesUploadResult {
	mimeType := file.MimeType
	if mimeType == "" {
		mimeType = mediaType
	}
	meta := map[string]any{
		"name":        file.Name,
		"displayName": file.DisplayName,
		"mimeType":    file.MimeType,
		"sizeBytes":   file.SizeBytes,
		"state":       file.State,
		"uri":         file.URI,
	}
	if file.CreateTime != "" {
		meta["createTime"] = file.CreateTime
	}
	if file.UpdateTime != "" {
		meta["updateTime"] = file.UpdateTime
	}
	if file.ExpirationTime != "" {
		meta["expirationTime"] = file.ExpirationTime
	}
	if file.SHA256Hash != "" {
		meta["sha256Hash"] = file.SHA256Hash
	}
	return &FilesUploadResult{
		ProviderReference: map[string]any{"google": file.URI},
		MediaType:         mimeType,
		ProviderMetadata:  ProviderMetadata{"google": meta},
	}
}
