package openaicompatible

import (
	"errors"
	"net/http"
	"strings"
	"testing"
)

func TestErrorParserDefaultAndAPICallErrorFields(t *testing.T) {
	resp := response(400, ``)
	resp.Header.Set("x-request-id", "req-123")
	err := buildAPICallError(resp, []byte(`{"error":{"message":"bad","type":"invalid","param":"p","code":7}}`), false, ProviderErrorStructure{})
	if err.Message != "bad" || err.Type != "invalid" || err.Param != "p" || err.Code.(float64) != 7 || err.Status != 400 || err.RequestID != "req-123" || err.Truncated || err.Retryable {
		t.Fatalf("unexpected error fields: %#v", err)
	}
	if !strings.Contains(err.Error(), "openaicompatible") || !strings.Contains(err.Error(), "status 400") || !strings.Contains(err.Error(), "bad") {
		t.Fatalf("Error() = %q", err.Error())
	}
}

func TestErrorParserCustomHooks(t *testing.T) {
	parsedSentinel := &struct{ Message string }{Message: "custom"}
	structure := ProviderErrorStructure{
		Parse: func(input ErrorParseInput) (any, error) {
			if input.Status != 418 || string(input.Body) != "body" || !input.Truncated {
				t.Fatalf("parse input = %#v", input)
			}
			return parsedSentinel, nil
		},
		ErrorToMessage: func(parsed any) string {
			if parsed != parsedSentinel {
				t.Fatalf("parsed = %#v", parsed)
			}
			return "custom message"
		},
		IsRetryable: func(resp *http.Response, parsed any) bool { return true },
	}
	err := buildAPICallError(response(418, ``), []byte("body"), true, structure)
	if err.Message != "custom message" || !err.Retryable || !err.Truncated {
		t.Fatalf("error = %#v", err)
	}
}

func TestErrorParserFallbacks(t *testing.T) {
	structure := ProviderErrorStructure{Parse: func(ErrorParseInput) (any, error) { return nil, errors.New("custom failed") }}
	err := buildAPICallError(response(500, ``), []byte(`{"error":{"message":"default"}}`), false, structure)
	if err.Message != "default" || !err.Retryable {
		t.Fatalf("default fallback error = %#v", err)
	}
	err = buildAPICallError(response(400, ``), []byte(" plain text \n"), false, structure)
	if err.Message != "plain text" {
		t.Fatalf("raw fallback message = %q", err.Message)
	}
	err = buildAPICallError(response(401, ``), []byte{0xff, 0xfe}, false, structure)
	if err.Message != "API call failed with status 401" {
		t.Fatalf("non-text fallback message = %q", err.Message)
	}
}

func TestAPICallErrorUnwrap(t *testing.T) {
	cause := errors.New("cause")
	err := &APICallError{Message: "m", Cause: cause}
	if !errors.Is(err, cause) {
		t.Fatalf("Unwrap did not expose cause")
	}
}
