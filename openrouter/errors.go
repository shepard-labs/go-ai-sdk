package openrouter

import (
	"errors"
	"fmt"
)

var (
	ErrMissingModelID = errors.New("openrouter: missing model id")
)

type APICallError struct {
	Message      string
	StatusCode   int
	Code         any
	Type         string
	Param        string
	Raw          string
	ProviderName string
	UserID       string
	Retryable    bool
	Body         []byte
	Cause        error
}

func (e *APICallError) Error() string {
	if e == nil {
		return ""
	}
	if e.StatusCode > 0 {
		return fmt.Sprintf("openrouter api error: status %d: %s", e.StatusCode, e.Message)
	}
	return "openrouter api error: " + e.Message
}

func (e *APICallError) Unwrap() error { return e.Cause }

type UnsupportedFunctionalityError struct{ Message string }

func (e UnsupportedFunctionalityError) Error() string {
	return "unsupported functionality: " + e.Message
}

type InvalidArgumentError struct{ Message string }

func (e InvalidArgumentError) Error() string { return "invalid argument: " + e.Message }

type InvalidPromptError struct{ Message string }

func (e InvalidPromptError) Error() string { return "invalid prompt: " + e.Message }

type InvalidResponseDataError struct{ Message string }

func (e InvalidResponseDataError) Error() string { return "invalid response data: " + e.Message }

type NoContentGeneratedError struct{ Message string }

func (e NoContentGeneratedError) Error() string {
	if e.Message == "" {
		return "no content generated"
	}
	return "no content generated: " + e.Message
}
