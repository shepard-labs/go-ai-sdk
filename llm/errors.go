package llm

import (
	"errors"
	"fmt"
	"time"
)

// APIError is a structured error from a provider API call.
type APIError struct {
	Provider   string
	StatusCode int
	Code       string
	Message    string
	RequestID  string
	RetryAfter *time.Duration
	Temporary  bool
}

func (e *APIError) Error() string {
	if e.RequestID != "" {
		return fmt.Sprintf("llm: %s API error %d (%s): %s [request_id=%s]", e.Provider, e.StatusCode, e.Code, e.Message, e.RequestID)
	}
	return fmt.Sprintf("llm: %s API error %d (%s): %s", e.Provider, e.StatusCode, e.Code, e.Message)
}

// IsRateLimit reports whether err is (or wraps) a rate-limit error.
func IsRateLimit(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 429
	}
	return false
}

// IsAuth reports whether err is (or wraps) an authentication/authorization error.
func IsAuth(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 401 || apiErr.StatusCode == 403
	}
	return false
}

// IsInvalidRequest reports whether err is (or wraps) a 400-class invalid-request error.
func IsInvalidRequest(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		sc := apiErr.StatusCode
		return sc >= 400 && sc < 500 && sc != 429 && sc != 401 && sc != 403
	}
	return false
}

// IsUnsupported reports whether err is (or wraps) an UnsupportedFeatureError.
func IsUnsupported(err error) bool {
	var ufErr *UnsupportedFeatureError
	return errors.As(err, &ufErr)
}

// IsTemporary reports whether err is transient and safe to retry.
func IsTemporary(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.Temporary {
		return true
	}
	return IsRateLimit(err)
}

// RetryAfter returns the suggested retry delay from a rate-limit error, if present.
func RetryAfter(err error) (time.Duration, bool) {
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter != nil {
		return *apiErr.RetryAfter, true
	}
	return 0, false
}
