package google

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf8"
)

// ErrMissingBaseURL is returned when ProviderSettings.BaseURL is empty and no
// default can be applied (should not occur in normal usage since CreateGoogle
// falls back to defaultBaseURL).
var ErrMissingBaseURL = errors.New("google: BaseURL is required")

// ErrMissingAPIKey is returned by [CreateGoogle] when no API key is provided via
// ProviderSettings.APIKey and GOOGLE_GENERATIVE_AI_API_KEY is not set.
var ErrMissingAPIKey = errors.New("google: APIKey is required (or set GOOGLE_GENERATIVE_AI_API_KEY env)")

// APIError holds a parsed Google API error object from the response body.
type APIError struct {
	Type    string
	Message string
	Param   any
	Code    any
}

// Error implements the error interface.
func (e APIError) Error() string {
	if e.Type == "" {
		return e.Message
	}
	return e.Type + ": " + e.Message
}

// APICallError represents a failed Google API call. It carries the HTTP status,
// parsed error fields, raw body, retryability flag, and the original cause when
// available. Use errors.As to extract it from wrapped errors.
type APICallError struct {
	Message   string
	Type      string
	Code      any
	Param     any
	Status    int
	Retryable bool
	Headers   http.Header
	RequestID string
	Body      []byte
	Truncated bool
	Cause     error
}

// Error implements the error interface.
func (e *APICallError) Error() string {
	if e == nil {
		return ""
	}
	if e.Status != 0 {
		return fmt.Sprintf("google: API call failed with status %d: %s", e.Status, e.Message)
	}
	return "google: API call failed: " + e.Message
}

// Unwrap returns the underlying cause error.
func (e *APICallError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

// InvalidPromptError is returned when the prompt structure is invalid, e.g. a
// system message that appears after user messages.
type InvalidPromptError struct{ Message string }

// Error implements the error interface.
func (e InvalidPromptError) Error() string { return "google: invalid prompt: " + e.Message }

// InvalidResponseDataError is returned when the provider returns a response that
// cannot be parsed according to the expected schema.
type InvalidResponseDataError struct {
	Message string
	Data    any
}

// Error implements the error interface.
func (e InvalidResponseDataError) Error() string {
	return "google: invalid response data: " + e.Message
}

// UnsupportedFunctionalityError is returned when the caller requests a capability
// that this provider or model does not support.
type UnsupportedFunctionalityError struct{ Functionality string }

// Error implements the error interface.
func (e UnsupportedFunctionalityError) Error() string {
	return "google: unsupported functionality: " + e.Functionality
}

// TooManyEmbeddingValuesForCallError is returned when the caller supplies more
// embedding input values than MaxEmbeddingsPerCall allows.
type TooManyEmbeddingValuesForCallError struct {
	Provider             string
	ModelID              string
	MaxEmbeddingsPerCall int
	Values               int
}

// Error implements the error interface.
func (e TooManyEmbeddingValuesForCallError) Error() string {
	return fmt.Sprintf("google: too many embedding values for provider %s model %s: max %d, got %d",
		e.Provider, e.ModelID, e.MaxEmbeddingsPerCall, e.Values)
}

// ErrorParseInput is the argument passed to a custom ProviderErrorStructure.Parse
// function.
type ErrorParseInput struct {
	Status    int
	Headers   http.Header
	Body      []byte
	Truncated bool
}

// ProviderErrorStructure provides optional hooks for customizing error parsing.
// When Parse is set it is tried first; on failure the default Google error parser
// is used as a fallback. IsRetryable overrides the default retry logic.
type ProviderErrorStructure struct {
	Parse          func(ErrorParseInput) (any, error)
	ErrorToMessage func(any) string
	IsRetryable    func(*http.Response, any) bool
}

// googleAPIErrorBody is the wire shape of a Google error response body.
//
//	{ "error": { "code": <int>, "message": <string>, "status": <string> } }
type googleAPIErrorBody struct {
	Error struct {
		Code    any    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

func buildAPICallError(resp *http.Response, body []byte, truncated bool, structure ProviderErrorStructure) *APICallError {
	status := 0
	headers := http.Header{}
	requestID := ""
	if resp != nil {
		status = resp.StatusCode
		headers = cloneHeader(resp.Header)
		requestID = headers.Get("x-request-id")
	}
	input := ErrorParseInput{Status: status, Headers: headers, Body: body, Truncated: truncated}
	parsed, parseErr := parseGoogleError(input, structure)
	apiErr, _ := parsed.(APIError)
	message := ""
	if parseErr == nil && parsed != nil && structure.ErrorToMessage != nil {
		message = structure.ErrorToMessage(parsed)
	}
	if message == "" {
		message = apiErr.Message
	}
	if message == "" {
		message = rawBodyFallbackMessage(status, body)
	}
	retryable := defaultRetryableStatus(status)
	if parseErr == nil && parsed != nil && structure.IsRetryable != nil {
		retryable = structure.IsRetryable(resp, parsed)
	}
	return &APICallError{
		Message:   message,
		Type:      apiErr.Type,
		Code:      apiErr.Code,
		Param:     apiErr.Param,
		Status:    status,
		Retryable: retryable,
		Headers:   headers,
		RequestID: requestID,
		Body:      append([]byte(nil), body...),
		Truncated: truncated,
		Cause:     parseErr,
	}
}

func parseGoogleError(input ErrorParseInput, structure ProviderErrorStructure) (any, error) {
	if structure.Parse != nil {
		if parsed, err := structure.Parse(input); err == nil {
			return parsed, nil
		}
	}
	return defaultParseGoogleError(input)
}

func defaultParseGoogleError(input ErrorParseInput) (any, error) {
	var decoded googleAPIErrorBody
	if err := json.Unmarshal(input.Body, &decoded); err != nil {
		return APIError{}, err
	}
	if decoded.Error.Message == "" {
		return APIError{}, errors.New("google: error message missing")
	}
	return APIError{Message: decoded.Error.Message, Type: decoded.Error.Status, Code: decoded.Error.Code}, nil
}

func rawBodyFallbackMessage(status int, body []byte) string {
	trimmed := strings.TrimSpace(string(body))
	if len(trimmed) > 0 && utf8.ValidString(trimmed) {
		return trimmed
	}
	return fmt.Sprintf("API call failed with status %d", status)
}

func defaultRetryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, // 408
		http.StatusConflict,        // 409
		http.StatusTooManyRequests: // 429
		return true
	}
	return status >= 500
}
