package typesafe

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

// ErrMissingAPIKey is reported by Provider.Err and DoDecide when neither an
// explicit API key nor TYPESAFE_AI_API_KEY is configured.
var ErrMissingAPIKey = errors.New("typesafe: API key is required")

// APICallError reports a failed TypeSafe API call.
type APICallError struct {
	Message   string
	Status    int
	Retryable bool
	Headers   http.Header
	Body      []byte
	Truncated bool
	Cause     error
}

func (e *APICallError) Error() string {
	if e == nil {
		return ""
	}
	if e.Status != 0 {
		return fmt.Sprintf("typesafe: API call failed with status %d: %s", e.Status, e.Message)
	}
	return "typesafe: API call failed: " + e.Message
}

func (e *APICallError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

// InvalidArgumentError reports a request argument that failed local
// validation before any HTTP call.
type InvalidArgumentError struct {
	Argument string
	Message  string
}

func (e *InvalidArgumentError) Error() string {
	if e.Argument != "" {
		return fmt.Sprintf("typesafe: invalid argument %s: %s", e.Argument, e.Message)
	}
	return "typesafe: invalid argument: " + e.Message
}

// UnsupportedFunctionalityError reports input the TypeSafe API cannot
// accept, such as non-image or non-inline file state.
type UnsupportedFunctionalityError struct {
	Functionality string
	Message       string
}

func (e *UnsupportedFunctionalityError) Error() string {
	if e.Message != "" {
		return "typesafe: unsupported functionality: " + e.Message
	}
	return "typesafe: unsupported functionality: " + e.Functionality
}

// InvalidResponseDataError reports a malformed 2xx response body.
type InvalidResponseDataError struct {
	Message string
	Data    any
}

func (e *InvalidResponseDataError) Error() string {
	return "typesafe: invalid response data: " + e.Message
}

// NoSuchModelError is returned by the unsupported language, embedding, and
// image model factories.
type NoSuchModelError struct {
	ModelID   string
	ModelType string
}

func (e *NoSuchModelError) Error() string {
	return fmt.Sprintf("typesafe: no such %s: %s", e.ModelType, e.ModelID)
}

// retryableStatus follows the repo convention: 408, 409, 429, and 5xx.
func retryableStatus(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusConflict ||
		status == http.StatusTooManyRequests || status >= 500
}

// errorMessage extracts a message from a TypeSafe error body, most specific
// first: message, string error, error.message, string detail, JSON-encoded
// detail, error_type, then a generic fallback.
func errorMessage(body []byte) string {
	var env struct {
		Message   *string         `json:"message"`
		Detail    json.RawMessage `json:"detail"`
		Error     json.RawMessage `json:"error"`
		ErrorType *string         `json:"error_type"`
	}
	if err := json.Unmarshal(body, &env); err == nil {
		if env.Message != nil {
			return *env.Message
		}
		if len(env.Error) > 0 && !bytes.Equal(env.Error, []byte("null")) {
			var s string
			if err := json.Unmarshal(env.Error, &s); err == nil {
				return s
			}
			var obj struct {
				Message *string `json:"message"`
			}
			if err := json.Unmarshal(env.Error, &obj); err == nil && obj.Message != nil {
				return *obj.Message
			}
		}
		if len(env.Detail) > 0 && !bytes.Equal(env.Detail, []byte("null")) {
			var s string
			if err := json.Unmarshal(env.Detail, &s); err == nil {
				return s
			}
			var compact bytes.Buffer
			if err := json.Compact(&compact, env.Detail); err == nil {
				return compact.String()
			}
		}
		if env.ErrorType != nil {
			return *env.ErrorType
		}
	}
	return "TypeSafe request failed"
}
