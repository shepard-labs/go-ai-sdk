package google

// HTTP, retry, and streaming plumbing for the google package.
//
// The behavior here is intentionally byte-for-byte identical to
// openaicompatible/http.go (same retry status set 408/409/429/5xx; same
// Retry-After parsing; same exponential backoff with jitter; same body-size
// cap; same drainAndClose semantics). The implementation is copied from that
// file and adapted to the googleProvider struct. A follow-up PR can promote
// the shared logic into an internal package.

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

func (p *googleProvider) requestURL(path string) (string, error) {
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

func (p *googleProvider) headersForCall(perCall http.Header) http.Header {
	headers := cloneHeader(p.headers)
	for k, values := range perCall {
		headers.Del(k)
		for _, v := range values {
			headers.Add(k, v)
		}
	}
	return headers
}

func (p *googleProvider) executeJSON(ctx context.Context, path string, body []byte, perCall http.Header) (*apiResponse, error) {
	if p.err != nil {
		return nil, p.err
	}
	return p.executeBuffered(ctx, http.MethodPost, path, body, perCall)
}

func (p *googleProvider) executeGet(ctx context.Context, path string, perCall http.Header) (*apiResponse, error) {
	if p.err != nil {
		return nil, p.err
	}
	return p.executeBuffered(ctx, http.MethodGet, path, nil, perCall)
}

func (p *googleProvider) executeStream(ctx context.Context, path string, body []byte, perCall http.Header) (*httpStreamResponse, error) {
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
		p.logger.Debug("google stream request start", "method", http.MethodPost, "url", requestURL, "attempt", attempt+1)
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if !shouldRetry(resp, err) || attempt == p.retry.MaxRetries {
				p.logger.Error("google stream request failure", "error", err)
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

func (p *googleProvider) executeBuffered(ctx context.Context, method, path string, body []byte, perCall http.Header) (*apiResponse, error) {
	requestURL, err := p.requestURL(path)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 0; attempt <= p.retry.MaxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var bodyReader io.Reader
		if len(body) > 0 {
			bodyReader = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, method, requestURL, bodyReader)
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
		p.logger.Debug("google request start", "method", method, "url", requestURL, "attempt", attempt+1)
		resp, err := p.fetch.Do(req)
		if err != nil {
			lastErr = err
			if !shouldRetry(resp, err) || attempt == p.retry.MaxRetries {
				p.logger.Error("google request failure", "error", err)
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
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return p.readResponse(resp)
	}
	return nil, lastErr
}

func (p *googleProvider) readResponse(resp *http.Response) (*apiResponse, error) {
	defer resp.Body.Close()
	body, truncated, err := readLimited(resp.Body, p.maxResponseBodyBytes)
	if err != nil {
		return nil, err
	}
	meta := &apiResponse{
		Status:    resp.StatusCode,
		Headers:   cloneHeader(resp.Header),
		RequestID: resp.Header.Get("x-request-id"),
		Body:      body,
		Truncated: truncated,
	}
	if truncated {
		return nil, &APICallError{
			Message:   "response body exceeded maximum size",
			Status:    resp.StatusCode,
			Headers:   meta.Headers,
			RequestID: meta.RequestID,
			Body:      body,
			Truncated: true,
			Retryable: false,
		}
	}
	return meta, nil
}

func (p *googleProvider) readAPIError(resp *http.Response) *APICallError {
	defer resp.Body.Close()
	body, truncated, err := readLimited(resp.Body, p.maxErrorResponseBytes)
	if err != nil {
		return &APICallError{
			Message:   err.Error(),
			Status:    resp.StatusCode,
			Headers:   cloneHeader(resp.Header),
			RequestID: resp.Header.Get("x-request-id"),
			Retryable: false,
			Cause:     err,
		}
	}
	apiErr := buildAPICallError(resp, body, truncated, p.errorStructure)
	p.logger.Error("google API error", "status", resp.StatusCode, "message", apiErr.Message)
	return apiErr
}

// readLimited reads up to limit bytes from r. If the body exceeds limit,
// the excess is silently discarded and truncated is true.
func readLimited(r io.Reader, limit int64) ([]byte, bool, error) {
	const defaultLimit int64 = 32 << 20
	if limit <= 0 {
		limit = defaultLimit
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

// shouldRetry reports whether the request should be retried based on the
// response status or transport error. Retries on 408, 409, 429, and 5xx; any
// transport error is also retried.
func shouldRetry(resp *http.Response, err error) bool {
	if err != nil {
		return true
	}
	if resp == nil {
		return true
	}
	return defaultRetryableStatus(resp.StatusCode)
}

// drainAndClose discards up to 1 KiB of body and closes it.
func drainAndClose(body io.ReadCloser) {
	if body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(body, 1024))
	_ = body.Close()
}

func (p *googleProvider) waitBeforeRetry(ctx context.Context, attempt int, resp *http.Response, err error) error {
	delay := p.retryDelay(attempt, resp)
	p.logger.Warn("google request retry", "attempt", attempt+1, "delay", delay, "error", err)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (p *googleProvider) retryDelay(attempt int, resp *http.Response) time.Duration {
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

// retryAfterDelay parses a Retry-After header value (integer seconds or
// HTTP-date) and returns the resulting duration.
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

// defaultHTTPClient returns a pre-configured *http.Client with connection pool
// settings tuned for API usage.
func defaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 100
	transport.MaxIdleConnsPerHost = 20
	transport.IdleConnTimeout = 90 * time.Second
	transport.TLSHandshakeTimeout = 10 * time.Second
	transport.ExpectContinueTimeout = time.Second
	transport.ResponseHeaderTimeout = 2 * time.Minute
	return &http.Client{Transport: transport}
}
