package anthropic

import (
	"fmt"
	"net/http"
)

type preSendTransportError struct{ cause error }

func (e *preSendTransportError) Error() string { return e.cause.Error() }
func (e *preSendTransportError) Unwrap() error { return e.cause }

// NewPreSendTransportError marks a Fetcher failure as having occurred before
// any request bytes were sent. Only failures carrying this package-owned mark
// are eligible for transport-error retries on replay-safe requests.
func NewPreSendTransportError(cause error) error {
	if cause == nil {
		return nil
	}
	return &preSendTransportError{cause: cause}
}

func isPreSendTransportError(err error) bool {
	type unwrapper interface{ Unwrap() error }
	for err != nil {
		if _, ok := err.(*preSendTransportError); ok {
			return true
		}
		wrapped, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = wrapped.Unwrap()
	}
	return false
}

type RequestConstructionError struct {
	ToolIndex int
	ToolID    string
	Message   string
}

type ThinkingTokenLimitError struct {
	ModelID         string
	BudgetTokens    int
	OutputTokens    int
	MaxOutputTokens int
}

type InvalidBaseURLError struct {
	URL   string
	Cause error
}

func (e *InvalidBaseURLError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("anthropic: invalid base URL %q: %v", e.URL, e.Cause)
}

func (e *InvalidBaseURLError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

func (e *RequestConstructionError) Error() string {
	if e == nil {
		return ""
	}
	if e.ToolIndex < 0 {
		return "anthropic: invalid request: " + e.Message
	}
	if e.ToolID != "" {
		return fmt.Sprintf("anthropic: invalid tool at index %d (%s): %s", e.ToolIndex, e.ToolID, e.Message)
	}
	return fmt.Sprintf("anthropic: invalid tool at index %d: %s", e.ToolIndex, e.Message)
}

func (e *ThinkingTokenLimitError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("anthropic: thinking budget %d plus output tokens %d exceeds %s %d max output tokens; thinking budget must be less than the model limit and leave at least one output token", e.BudgetTokens, e.OutputTokens, e.ModelID, e.MaxOutputTokens)
}

type APIError struct {
	Type    string
	Message string
}

func (e APIError) Error() string {
	if e.Type == "" {
		return e.Message
	}
	return e.Type + ": " + e.Message
}

type APICallError struct {
	Message string
	Type    string
	Status  int
	// SyntheticStatus reports whether Status was inferred from a provider error
	// type rather than received as an HTTP response status. Adding exported
	// fields can break external unkeyed APICallError literals; keyed literals
	// remain source compatible.
	SyntheticStatus bool
	Retryable       bool
	Headers         http.Header
	RequestID       string
	Body            []byte
	Truncated       bool
	Cause           error
}

func (e *APICallError) Error() string {
	if e == nil {
		return ""
	}
	if e.Status != 0 {
		return fmt.Sprintf("anthropic: API call failed with status %d: %s", e.Status, e.Message)
	}
	return "anthropic: API call failed: " + e.Message
}

func (e *APICallError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

type Logger interface {
	Debug(msg string, keysAndValues ...any)
	Info(msg string, keysAndValues ...any)
	Warn(msg string, keysAndValues ...any)
	Error(msg string, keysAndValues ...any)
}

type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
