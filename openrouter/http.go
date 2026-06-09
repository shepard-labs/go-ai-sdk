package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

func cloneHeader(h http.Header) http.Header {
	out := make(http.Header)
	for k, values := range h {
		for _, v := range values {
			out.Add(k, v)
		}
	}
	return out
}

func mergeHeader(dst, src http.Header) {
	for k, values := range src {
		dst.Del(k)
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func mergeBody(dst map[string]any, src map[string]any) {
	for k, v := range src {
		dst[k] = v
	}
}

func mergeOpenRouterOptions(dst map[string]any, opts ProviderOptions) {
	or := cloneMap(opts.OpenRouter())
	if or == nil {
		return
	}
	if v, ok := or["cacheControl"]; ok {
		if _, exists := or["cache_control"]; !exists {
			or["cache_control"] = v
		}
		delete(or, "cacheControl")
	}
	mergeBody(dst, or)
}

func (p *openRouterProvider) postJSON(ctx context.Context, path string, body map[string]any, headers http.Header, out any) ([]byte, http.Header, error) {
	requestBody, err := json.Marshal(body)
	if err != nil {
		return nil, nil, err
	}
	return p.doJSON(ctx, http.MethodPost, path, requestBody, headers, out)
}

func (p *openRouterProvider) getJSON(ctx context.Context, path string, headers http.Header, out any) ([]byte, http.Header, error) {
	return p.doJSON(ctx, http.MethodGet, path, nil, headers, out)
}

func (p *openRouterProvider) doJSON(ctx context.Context, method, path string, body []byte, headers http.Header, out any) ([]byte, http.Header, error) {
	var lastErr error
	var lastResp *http.Response
	for attempt := 0; attempt <= p.retry.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := p.retryDelay(attempt-1, lastResp)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, nil, ctx.Err()
			}
		}
		var rdr io.Reader
		if body != nil {
			rdr = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, method, p.baseURL+path, rdr)
		if err != nil {
			return nil, nil, err
		}
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		mergeHeader(req.Header, p.requestHeaders(headers))
		resp, err := p.fetch.Do(req)
		lastResp = resp
		if err != nil {
			lastErr = err
			if attempt < p.retry.MaxRetries {
				continue
			}
			return nil, nil, &APICallError{Message: err.Error(), Retryable: true, Cause: err}
		}
		raw, readErr := readLimited(resp.Body, limitForStatus(resp.StatusCode, p.maxResponseBodyBytes, p.maxErrorResponseBytes))
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, resp.Header, readErr
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			err := parseAPIError(resp.StatusCode, raw)
			err.Retryable = retryableStatus(resp.StatusCode)
			if err.Retryable && attempt < p.retry.MaxRetries {
				lastErr = err
				continue
			}
			return raw, resp.Header, err
		}
		if apiErr := apiErrorPayload(200, raw); apiErr != nil {
			return raw, resp.Header, apiErr
		}
		if out != nil && len(strings.TrimSpace(string(raw))) > 0 {
			if err := json.Unmarshal(raw, out); err != nil {
				return raw, resp.Header, InvalidResponseDataError{Message: err.Error()}
			}
		}
		return raw, resp.Header, nil
	}
	return nil, nil, lastErr
}

func readLimited(r io.Reader, limit int64) ([]byte, error) {
	if limit <= 0 {
		return io.ReadAll(r)
	}
	return io.ReadAll(io.LimitReader(r, limit))
}

func limitForStatus(status int, okLimit, errLimit int64) int64 {
	if status >= 200 && status < 300 {
		return okLimit
	}
	return errLimit
}

func parseAPIError(status int, body []byte) *APICallError {
	if err := apiErrorPayload(status, body); err != nil {
		return err
	}
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = http.StatusText(status)
	}
	return &APICallError{StatusCode: status, Message: msg, Body: body}
}

func apiErrorPayload(status int, body []byte) *APICallError {
	var payload struct {
		Error *struct {
			Message  string `json:"message"`
			Code     any    `json:"code"`
			Type     string `json:"type"`
			Param    string `json:"param"`
			Metadata struct {
				Raw          string `json:"raw"`
				ProviderName string `json:"provider_name"`
			} `json:"metadata"`
		} `json:"error"`
		UserID string `json:"user_id"`
	}
	if json.Unmarshal(body, &payload) != nil || payload.Error == nil {
		return nil
	}
	msg := payload.Error.Message
	if msg == "" {
		msg = fmt.Sprint(payload.Error.Code)
	}
	return &APICallError{StatusCode: status, Message: msg, Code: payload.Error.Code, Type: payload.Error.Type, Param: payload.Error.Param, Raw: payload.Error.Metadata.Raw, ProviderName: payload.Error.Metadata.ProviderName, UserID: payload.UserID, Body: body, Retryable: retryableStatus(status)}
}

func retryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500 || status == http.StatusRequestTimeout
}

func (p *openRouterProvider) retryDelay(attempt int, resp *http.Response) time.Duration {
	if resp != nil {
		if d, ok := retryAfterDelay(resp.Header.Get("Retry-After")); ok {
			return d
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
	if seconds, err := time.ParseDuration(value + "s"); err == nil {
		return seconds, true
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	d := time.Until(when)
	if d < 0 {
		d = 0
	}
	return d, true
}
