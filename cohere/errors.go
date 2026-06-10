package cohere

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf8"
)

var ErrMissingAPIKey = errors.New("cohere: API key is required")

type APIError struct{ Message string }

func (e APIError) Error() string { return e.Message }

type APICallError struct {
	Message   string
	Status    int
	Retryable bool
	Headers   http.Header
	RequestID string
	Body      []byte
	Truncated bool
	Cause     error
}

func (e *APICallError) Error() string {
	if e == nil {
		return ""
	}
	if e.Status != 0 {
		return fmt.Sprintf("cohere: API call failed with status %d: %s", e.Status, e.Message)
	}
	return "cohere: API call failed: " + e.Message
}
func (e *APICallError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

type UnsupportedFunctionalityError struct{ Functionality, Message string }

func (e UnsupportedFunctionalityError) Error() string {
	if e.Message != "" {
		return "cohere: unsupported functionality: " + e.Message
	}
	return "cohere: unsupported functionality: " + e.Functionality
}

type InvalidPromptError struct{ Message string }

func (e InvalidPromptError) Error() string { return "cohere: invalid prompt: " + e.Message }

type InvalidResponseDataError struct {
	Message string
	Data    any
}

func (e InvalidResponseDataError) Error() string {
	return "cohere: invalid response data: " + e.Message
}

type TooManyEmbeddingValuesForCallError struct {
	Provider, ModelID    string
	MaxEmbeddingsPerCall int
	Values               []string
}

func (e TooManyEmbeddingValuesForCallError) Error() string {
	return fmt.Sprintf("cohere: too many embedding values for provider %s model %s: max %d, got %d", e.Provider, e.ModelID, e.MaxEmbeddingsPerCall, len(e.Values))
}

type NoSuchModelError struct{ ModelID, ModelType string }

func (e NoSuchModelError) Error() string {
	return fmt.Sprintf("cohere: no such %s model: %s", e.ModelType, e.ModelID)
}

func buildAPICallError(resp *http.Response, body []byte, truncated bool) *APICallError {
	status, headers, requestID := 0, http.Header{}, ""
	if resp != nil {
		status = resp.StatusCode
		headers = cloneHeader(resp.Header)
		requestID = headers.Get("x-request-id")
	}
	msg := ""
	var decoded APIError
	parseErr := json.Unmarshal(body, &decoded)
	if parseErr == nil {
		msg = decoded.Message
	}
	if msg == "" {
		msg = rawBodyFallbackMessage(status, body)
	}
	return &APICallError{Message: msg, Status: status, Retryable: defaultRetryableStatus(status), Headers: headers, RequestID: requestID, Body: append([]byte(nil), body...), Truncated: truncated, Cause: parseErr}
}

func rawBodyFallbackMessage(status int, body []byte) string {
	trimmed := strings.TrimSpace(string(body))
	if trimmed != "" && utf8.ValidString(trimmed) {
		return trimmed
	}
	return fmt.Sprintf("API call failed with status %d", status)
}
func defaultRetryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}
