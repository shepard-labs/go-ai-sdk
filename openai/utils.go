package openai

import (
	"math/rand"
	"net/http"
	"time"
)

// defaultOpenAIRetry returns a default retry options struct, applying user
// overrides from opts.
func defaultOpenAIRetry(opts *RetryOptions) RetryOptions {
	retry := RetryOptions{MaxRetries: 2, BaseDelay: 200 * time.Millisecond, MaxDelay: 2 * time.Second, Jitter: true}
	if opts == nil {
		return retry
	}
	retry.MaxRetries = opts.MaxRetries
	if opts.BaseDelay > 0 {
		retry.BaseDelay = opts.BaseDelay
	}
	if opts.MaxDelay > 0 {
		retry.MaxDelay = opts.MaxDelay
	}
	retry.Jitter = opts.Jitter
	if opts.MaxRetries < 0 {
		retry.MaxRetries = 0
	}
	if retry.MaxDelay < retry.BaseDelay {
		retry.MaxDelay = retry.BaseDelay
	}
	return retry
}

func openaiDefaultIfZero(value, fallback int64) int64 {
	if value > 0 {
		return value
	}
	return fallback
}

func openaiDefaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 100
	transport.MaxIdleConnsPerHost = 20
	transport.IdleConnTimeout = 90 * time.Second
	transport.TLSHandshakeTimeout = 10 * time.Second
	transport.ExpectContinueTimeout = time.Second
	transport.ResponseHeaderTimeout = 2 * time.Minute
	return &http.Client{Transport: transport}
}

type openaiNoopLogger struct{}

func (openaiNoopLogger) Debug(string, ...any) {}
func (openaiNoopLogger) Info(string, ...any)  {}
func (openaiNoopLogger) Warn(string, ...any)  {}
func (openaiNoopLogger) Error(string, ...any) {}

// retryAfterDelay parses a Retry-After header value.
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
	delay := time.Until(when)
	if delay < 0 {
		delay = 0
	}
	return delay, true
}

// retryJitter is a helper that returns a random sub-duration of d.
func retryJitter(d time.Duration, jitter bool) time.Duration {
	if jitter && d > 0 {
		return time.Duration(rand.Int63n(int64(d)))
	}
	return d
}

// intPtr returns a pointer to v.
func intPtr(v int) *int { return &v }

// floatPtr returns a pointer to v.
func floatPtr(v float64) *float64 { return &v }

// stringPtr returns a pointer to v.
func stringPtr(v string) *string { return &v }

// boolPtr returns a pointer to v.
func boolPtr(v bool) *bool { return &v }

// intValueOrZero dereferences v, returning 0 if v is nil.
func intValueOrZero(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}

// derefString returns the value of s, or fallback if s is nil.
func derefString(s *string, fallback string) string {
	if s == nil {
		return fallback
	}
	return *s
}
