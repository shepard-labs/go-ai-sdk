package openaicompatible

import (
	"bytes"
	"context"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

const (
	endpointChatCompletions  = "/chat/completions"
	endpointCompletions      = "/completions"
	endpointEmbeddings       = "/embeddings"
	endpointImageGenerations = "/images/generations"
	endpointImageEdits       = "/images/edits"
)

type apiResponse struct {
	Status    int
	Headers   http.Header
	RequestID string
	Body      []byte
	Truncated bool
}

type httpStreamResponse struct {
	Body    io.ReadCloser
	Headers http.Header
}

func (p *openAICompatibleProvider) requestURL(path string) (string, error) {
	u, err := url.Parse(p.baseURL + path)
	if err != nil {
		return "", err
	}
	if len(p.queryParams) > 0 {
		q := url.Values{}
		for k, v := range p.queryParams {
			q.Set(k, v)
		}
		u.RawQuery = q.Encode()
	}
	return u.String(), nil
}

func (p *openAICompatibleProvider) headersForCall(perCall http.Header) http.Header {
	headers := cloneHeader(p.headers)
	for k, values := range perCall {
		headers.Del(k)
		for _, v := range values {
			headers.Add(k, v)
		}
	}
	return headers
}

func (p *openAICompatibleProvider) executeJSON(ctx context.Context, path string, body []byte, perCall http.Header) (*apiResponse, error) {
	if p.err != nil {
		return nil, p.err
	}
	return p.executeBuffered(ctx, http.MethodPost, path, body, perCall)
}

func (p *openAICompatibleProvider) executeStream(ctx context.Context, path string, body []byte, perCall http.Header) (*httpStreamResponse, error) {
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
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body))
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
		p.logger.Debug("openaicompatible stream request start", "method", http.MethodPost, "url", requestURL, "attempt", attempt+1)
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if !shouldRetry(resp, err) || attempt == p.retry.MaxRetries {
				p.logger.Error("openaicompatible stream request failure", "error", err)
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
		return &httpStreamResponse{Body: resp.Body, Headers: cloneHeader(resp.Header)}, nil
	}
	return nil, lastErr
}

func (p *openAICompatibleProvider) executeBuffered(ctx context.Context, method, path string, body []byte, perCall http.Header) (*apiResponse, error) {
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
		p.logger.Debug("openaicompatible request start", "method", method, "url", requestURL, "attempt", attempt+1)
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if !shouldRetry(resp, err) || attempt == p.retry.MaxRetries {
				p.logger.Error("openaicompatible request failure", "error", err)
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
		if shouldRetry(resp, nil) && attempt < p.retry.MaxRetries {
			drainAndClose(resp.Body)
			if waitErr := p.waitBeforeRetry(ctx, attempt, resp, nil); waitErr != nil {
				return nil, waitErr
			}
			continue
		}
		return p.readResponse(resp)
	}
	return nil, lastErr
}

func (p *openAICompatibleProvider) readResponse(resp *http.Response) (*apiResponse, error) {
	defer resp.Body.Close()
	body, truncated, err := readLimited(resp.Body, p.maxResponseBodyBytes)
	if err != nil {
		return nil, err
	}
	meta := &apiResponse{Status: resp.StatusCode, Headers: cloneHeader(resp.Header), RequestID: resp.Header.Get("x-request-id"), Body: body, Truncated: truncated}
	if truncated {
		return nil, &APICallError{Message: "response body exceeded maximum size", Status: resp.StatusCode, Headers: meta.Headers, RequestID: meta.RequestID, Body: body, Truncated: true, Retryable: false}
	}
	return meta, nil
}

func (p *openAICompatibleProvider) readAPIError(resp *http.Response) *APICallError {
	defer resp.Body.Close()
	body, truncated, err := readLimited(resp.Body, p.maxErrorResponseBytes)
	if err != nil {
		return &APICallError{Message: err.Error(), Status: resp.StatusCode, Headers: cloneHeader(resp.Header), RequestID: resp.Header.Get("x-request-id"), Retryable: false, Cause: err}
	}
	apiErr := buildAPICallError(resp, body, truncated, p.errorStructure)
	p.logger.Error("openaicompatible API error", "status", resp.StatusCode, "message", apiErr.Message)
	return apiErr
}

func readLimited(r io.Reader, limit int64) ([]byte, bool, error) {
	if limit <= 0 {
		limit = defaultMaxResponseBodyBytes
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

func shouldRetry(resp *http.Response, err error) bool {
	if err != nil {
		return true
	}
	if resp == nil {
		return true
	}
	return defaultRetryableStatus(resp.StatusCode)
}

func drainAndClose(body io.ReadCloser) {
	if body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(body, 1024))
	_ = body.Close()
}

func (p *openAICompatibleProvider) waitBeforeRetry(ctx context.Context, attempt int, resp *http.Response, err error) error {
	delay := p.retryDelay(attempt, resp)
	p.logger.Warn("openaicompatible request retry", "attempt", attempt+1, "delay", delay, "error", err)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (p *openAICompatibleProvider) retryDelay(attempt int, resp *http.Response) time.Duration {
	if resp != nil {
		if delay, ok := retryAfterDelay(resp.Header.Get("Retry-After")); ok {
			return delay
		}
	}
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

func retryAfterDelay(value string) (time.Duration, bool) {
	if value == "" {
		return 0, false
	}
	if seconds, err := strconv.Atoi(value); err == nil {
		return time.Duration(seconds) * time.Second, true
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	delay := time.Until(when)
	if delay < 0 {
		delay = 0
	}
	return delay, true
}
