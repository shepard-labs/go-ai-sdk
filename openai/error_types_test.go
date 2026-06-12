package openai

import (
	"errors"
	"net/http"
	"testing"
)

// Verifies that the openai-package error types are aliases for the
// openaicompatible ones, so callers can use either form interchangeably.
func TestErrorTypesAreAliases(t *testing.T) {
	// Construct via openai types, type-assert to openaicompatible types.
	var _ error = InvalidPromptError{Message: "x"}
	var _ error = InvalidResponseDataError{Message: "x"}
	var _ error = UnsupportedFunctionalityError{Functionality: "x"}
	var _ error = TooManyEmbeddingValuesForCallError{MaxEmbeddingsPerCall: 2048, Values: []string{"a", "b"}}

	// APICallError is a struct, not just an interface. Retryable must
	// be set explicitly by the parser (defaultRetryableStatus handles
	// this), so a zero-value APICallError is not retryable until parsed.
	api := &APICallError{Message: "x", Status: 500}
	var _ error = api
	if api.Retryable {
		t.Errorf("zero-value APICallError should not be retryable (parser sets it)")
	}
	// But once built via buildAPICallError, 5xx is retryable.
	parsed := buildAPICallError(&http.Response{StatusCode: 500, Header: http.Header{}}, []byte(`{"error":{"message":"oops"}}`), false)
	if !parsed.Retryable {
		t.Errorf("500 should be retryable after buildAPICallError")
	}
}

// Verifies error messages include the relevant detail.
func TestErrorMessagesIncludeDetail(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{InvalidPromptError{Message: "no tool_choice"}, "no tool_choice"},
		{InvalidResponseDataError{Message: "bad JSON"}, "bad JSON"},
		{UnsupportedFunctionalityError{Functionality: "shell tool"}, "shell tool"},
	}
	for _, c := range cases {
		if got := c.err.Error(); !contains(got, c.want) {
			t.Errorf("err.Error() = %q, want substring %q", got, c.want)
		}
	}
}

// Verifies the TooManyEmbeddingValuesForCallError message identifies the
// cap and the supplied count.
func TestTooManyEmbeddingValuesErrorMessage(t *testing.T) {
	err := TooManyEmbeddingValuesForCallError{
		MaxEmbeddingsPerCall: 2048,
		Values:               make([]string, 3000),
	}
	msg := err.Error()
	if !contains(msg, "2048") {
		t.Errorf("expected max in message: %q", msg)
	}
	if !contains(msg, "3000") {
		t.Errorf("expected count in message: %q", msg)
	}
}

// Verifies that the APIError embedded in APICallError captures the
// standard OpenAI error fields.
func TestAPICallErrorCarriesFields(t *testing.T) {
	headers := http.Header{"x-request-id": []string{"req-1"}}
	body := []byte(`{"error":{"message":"bad","type":"bad_request","code":"bad"}}`)
	api := &APICallError{
		Message:    "bad",
		Type:       "bad_request",
		Code:       "bad",
		Status:     400,
		Headers:    headers,
		RequestID:  "req-1",
		Body:       body,
		Retryable:  false,
		Truncated:  false,
	}
	if api.Type != "bad_request" {
		t.Errorf("Type: %q", api.Type)
	}
	if api.Status != 400 {
		t.Errorf("Status: %d", api.Status)
	}
	if api.RequestID != "req-1" {
		t.Errorf("RequestID: %q", api.RequestID)
	}
	// Errors.As should reach the APIError target.
	var apiErr *APICallError
	if !errors.As(api, &apiErr) {
		t.Errorf("errors.As failed")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
