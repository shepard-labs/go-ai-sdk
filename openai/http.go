package openai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math/rand"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// httpAPIResponse is the captured response of a buffered HTTP call.
type httpAPIResponse struct {
	Status    int
	Headers   http.Header
	RequestID string
	Body      []byte
	Truncated bool
}

// httpStreamResponse is the captured stream of an HTTP call.
type httpStreamResponse struct {
	Body    io.ReadCloser
	Headers http.Header
}

// executeJSON performs a POST to {baseURL}{path} with the given JSON body
// and returns the response.
func (p *openaiProvider) executeJSON(ctx context.Context, path string, body []byte, perCall http.Header) (*httpAPIResponse, error) {
	if p.err != nil {
		return nil, p.err
	}
	return p.executeBuffered(ctx, http.MethodPost, path, body, perCall)
}

// executeStream performs a POST and returns a streaming response.
func (p *openaiProvider) executeStream(ctx context.Context, path string, body []byte, perCall http.Header) (*httpStreamResponse, error) {
	if p.err != nil {
		return nil, p.err
	}
	return p.executeStreamBuffered(ctx, http.MethodPost, path, body, perCall)
}

// executeMultipart performs a multipart POST.
func (p *openaiProvider) executeMultipart(ctx context.Context, path string, perCall http.Header, build func(*multipart.Writer) error) (*httpAPIResponse, error) {
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
			if attempt == p.retry.MaxRetries {
				return nil, err
			}
			if waitErr := p.waitBeforeRetry(ctx, attempt); waitErr != nil {
				return nil, waitErr
			}
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			apiErr := p.readAPIError(resp)
			lastErr = apiErr
			if attempt < p.retry.MaxRetries {
				if waitErr := p.waitBeforeRetry(ctx, attempt); waitErr != nil {
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

func (p *openaiProvider) requestURL(path string) (string, error) {
	u, err := url.Parse(p.baseURL + path)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

func (p *openaiProvider) headersForCall(perCall http.Header) http.Header {
	headers := p.compatHeaders().Clone()
	for k, values := range perCall {
		headers.Del(k)
		for _, v := range values {
			headers.Add(k, v)
		}
	}
	return headers
}

// compatHeaders reuses the openaicompatible package's headers (which carry
// auth, org, project, user-agent, plus the user-supplied spread Headers
// stored on the provider) and returns a clone the openai package
// uses for its own requests.
func (p *openaiProvider) compatHeaders() http.Header {
	headers := http.Header{}
	if p.apiKey != "" {
		headers.Set("Authorization", "Bearer "+p.apiKey)
	}
	if p.organization != "" {
		headers.Set("OpenAI-Organization", p.organization)
	}
	if p.project != "" {
		headers.Set("OpenAI-Project", p.project)
	}
	headers.Set("User-Agent", userAgent)
	// Apply the spread ProviderSettings.Headers (overriding auth/org/project
	// when explicitly set, per the spec).
	for k, values := range p.headers {
		headers.Del(k)
		for _, v := range values {
			headers.Add(k, v)
		}
	}
	return headers
}

func (p *openaiProvider) executeBuffered(ctx context.Context, method, path string, body []byte, perCall http.Header) (*httpAPIResponse, error) {
	requestURL, err := p.requestURL(path)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 0; attempt <= p.retry.MaxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, method, requestURL, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		for k, values := range p.headersForCall(perCall) {
			for _, v := range values {
				req.Header.Add(k, v)
			}
		}
		if len(body) > 0 {
			req.Header.Set("Content-Type", "application/json")
		}
		if p.generateID != nil {
			req.Header.Set("x-request-id", p.generateID())
		}
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if attempt == p.retry.MaxRetries {
				return nil, err
			}
			if waitErr := p.waitBeforeRetry(ctx, attempt); waitErr != nil {
				return nil, waitErr
			}
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			apiErr := p.readAPIError(resp)
			lastErr = apiErr
			if attempt < p.retry.MaxRetries {
				if waitErr := p.waitBeforeRetry(ctx, attempt); waitErr != nil {
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

func (p *openaiProvider) executeStreamBuffered(ctx context.Context, method, path string, body []byte, perCall http.Header) (*httpStreamResponse, error) {
	requestURL, err := p.requestURL(path)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 0; attempt <= p.retry.MaxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, method, requestURL, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		for k, values := range p.headersForCall(perCall) {
			for _, v := range values {
				req.Header.Add(k, v)
			}
		}
		if len(body) > 0 {
			req.Header.Set("Content-Type", "application/json")
		}
		if p.generateID != nil {
			req.Header.Set("x-request-id", p.generateID())
		}
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if attempt == p.retry.MaxRetries {
				return nil, err
			}
			if waitErr := p.waitBeforeRetry(ctx, attempt); waitErr != nil {
				return nil, waitErr
			}
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			apiErr := p.readAPIError(resp)
			lastErr = apiErr
			if attempt < p.retry.MaxRetries {
				if waitErr := p.waitBeforeRetry(ctx, attempt); waitErr != nil {
					return nil, waitErr
				}
				continue
			}
			return nil, apiErr
		}
		return &httpStreamResponse{Body: resp.Body, Headers: resp.Header.Clone()}, nil
	}
	return nil, lastErr
}

func (p *openaiProvider) readResponse(resp *http.Response) (*httpAPIResponse, error) {
	defer resp.Body.Close()
	body, truncated, err := readLimited(resp.Body, p.maxResponseBodyBytes)
	if err != nil {
		return nil, err
	}
	meta := &httpAPIResponse{Status: resp.StatusCode, Headers: resp.Header.Clone(), RequestID: resp.Header.Get("x-request-id"), Body: body, Truncated: truncated}
	if truncated {
		return nil, &APICallError{Message: "response body exceeded maximum size", Status: resp.StatusCode, Headers: meta.Headers, RequestID: meta.RequestID, Body: body, Truncated: true}
	}
	return meta, nil
}

func (p *openaiProvider) readAPIError(resp *http.Response) *APICallError {
	defer resp.Body.Close()
	body, truncated, err := readLimited(resp.Body, p.maxErrorResponseBytes)
	if err != nil {
		return &APICallError{Message: err.Error(), Status: resp.StatusCode, Headers: resp.Header.Clone(), Retryable: false, Cause: err}
	}
	return buildAPICallError(resp, body, truncated)
}

func readLimited(r io.Reader, limit int64) ([]byte, bool, error) {
	if limit <= 0 {
		limit = 32 << 20
	}
	lr := io.LimitReader(r, limit+1)
	body, err := io.ReadAll(lr)
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) > limit {
		return body[:limit], true, nil
	}
	return body, false, nil
}

func (p *openaiProvider) waitBeforeRetry(ctx context.Context, attempt int) error {
	delay := p.retryDelay(attempt)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (p *openaiProvider) retryDelay(attempt int) time.Duration {
	delay := p.retry.BaseDelay
	for range attempt {
		delay *= 2
		if delay >= p.retry.MaxDelay {
			delay = p.retry.MaxDelay
			break
		}
	}
	if p.retry.Jitter && delay > 0 {
		return time.Duration(rand.Int63n(int64(delay)))
	}
	return delay
}

func buildAPICallError(resp *http.Response, body []byte, truncated bool) *APICallError {
	status := 0
	headers := http.Header{}
	requestID := ""
	if resp != nil {
		status = resp.StatusCode
		headers = resp.Header.Clone()
		requestID = headers.Get("x-request-id")
	}
	var decoded openaiErrorBody
	_ = decodeJSON(body, &decoded)
	// Some OpenAI error types don't surface an HTTP status (the server
	// returns 200 but the body is an error). Derive a status code from
	// the error type so callers can react accordingly.
	if status < 400 {
		if derived, ok := openAIErrorTypeStatusCode(decoded.Error.Type); ok {
			status = derived
		}
	}
	retryable := defaultRetryableStatus(status)
	apiErr := &APICallError{
		Message:   decoded.Error.Message,
		Type:      decoded.Error.Type,
		Code:      decoded.Error.Code,
		Param:     decoded.Error.Param,
		Status:    status,
		Retryable: retryable,
		Headers:   headers,
		RequestID: requestID,
		Body:      append([]byte(nil), body...),
		Truncated: truncated,
	}
	if apiErr.Message == "" {
		apiErr.Message = strings.TrimSpace(string(body))
		if apiErr.Message == "" {
			apiErr.Message = "API call failed with status " + strconv.Itoa(status)
		}
	}
	return apiErr
}

func defaultRetryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

// openAIErrorTypeStatusCode maps known OpenAI error.type strings to
// HTTP-like status codes, used when the server returns a non-error
// status code but the body carries an error payload.
func openAIErrorTypeStatusCode(t string) (int, bool) {
	switch t {
	case "insufficient_quota", "rate_limit_exceeded":
		return http.StatusTooManyRequests, true
	case "invalid_request_error", "invalid_api_key", "invalid_organization",
		"invalid_project", "invalid_user", "invalid_grant", "invalid_scope":
		return http.StatusBadRequest, true
	case "authentication_error":
		return http.StatusUnauthorized, true
	case "permission_error", "forbidden":
		return http.StatusForbidden, true
	case "not_found":
		return http.StatusNotFound, true
	case "conflict":
		return http.StatusConflict, true
	case "unprocessable_entity":
		return http.StatusUnprocessableEntity, true
	case "server_error", "api_error", "internal_server_error":
		return http.StatusInternalServerError, true
	case "service_unavailable", "overloaded":
		return http.StatusServiceUnavailable, true
	}
	return 0, false
}

// openaiErrorBody is the standard OpenAI error response body.
type openaiErrorBody struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   any    `json:"param"`
		Code    any    `json:"code"`
	} `json:"error"`
}

// decodeJSON decodes JSON body into v, ignoring errors.
func decodeJSON(body []byte, v any) error {
	if len(body) == 0 {
		return errors.New("empty body")
	}
	return jsonUnmarshal(body, v)
}

// Static errors to avoid the unused import warning.
var _ = http.MethodPost
var _ = io.Discard
