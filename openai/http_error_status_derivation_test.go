package openai

import (
	"net/http"
	"testing"
)

// TestOpenAIErrorTypeStatusCode verifies the type→status mapping for
// known OpenAI error.type strings. This is used when the server returns
// a non-error HTTP status but the body is an error.
func TestOpenAIErrorTypeStatusCode(t *testing.T) {
	tests := []struct {
		typ   string
		want  int
		found bool
	}{
		{"insufficient_quota", http.StatusTooManyRequests, true},
		{"rate_limit_exceeded", http.StatusTooManyRequests, true},
		{"invalid_request_error", http.StatusBadRequest, true},
		{"invalid_api_key", http.StatusBadRequest, true},
		{"authentication_error", http.StatusUnauthorized, true},
		{"permission_error", http.StatusForbidden, true},
		{"forbidden", http.StatusForbidden, true},
		{"not_found", http.StatusNotFound, true},
		{"conflict", http.StatusConflict, true},
		{"unprocessable_entity", http.StatusUnprocessableEntity, true},
		{"server_error", http.StatusInternalServerError, true},
		{"overloaded", http.StatusServiceUnavailable, true},
		{"unknown_thing", 0, false},
		{"", 0, false},
	}
	for _, tt := range tests {
		got, found := openAIErrorTypeStatusCode(tt.typ)
		if got != tt.want || found != tt.found {
			t.Errorf("openAIErrorTypeStatusCode(%q) = (%d, %v), want (%d, %v)",
				tt.typ, got, found, tt.want, tt.found)
		}
	}
}

// TestBuildAPICallErrorDerivesStatusFromType verifies that when the HTTP
// status is 200 but the body has an error.type, the status is upgraded
// to the derived code.
func TestBuildAPICallErrorDerivesStatusFromType(t *testing.T) {
	headers := http.Header{}
	headers.Set("x-request-id", "req_abc")
	resp := &http.Response{
		StatusCode: 200,
		Header:     headers,
	}
	body := []byte(`{"error":{"message":"You exceeded your quota","type":"insufficient_quota","code":"insufficient_quota"}}`)
	apiErr := buildAPICallError(resp, body, false)
	if apiErr.Status != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", apiErr.Status)
	}
	if apiErr.Type != "insufficient_quota" {
		t.Errorf("type = %q", apiErr.Type)
	}
	if !apiErr.Retryable {
		t.Errorf("429 should be retryable")
	}
	if apiErr.RequestID != "req_abc" {
		t.Errorf("request id = %q", apiErr.RequestID)
	}
}

// TestBuildAPICallErrorKeepsHTTPStatus verifies that when the HTTP
// status is already >= 400, no derivation is applied.
func TestBuildAPICallErrorKeepsHTTPStatus(t *testing.T) {
	resp := &http.Response{StatusCode: 500, Header: http.Header{}}
	body := []byte(`{"error":{"message":"oh no","type":"insufficient_quota"}}`)
	apiErr := buildAPICallError(resp, body, false)
	if apiErr.Status != 500 {
		t.Errorf("status = %d, want 500 (preserved)", apiErr.Status)
	}
}

// TestBuildAPICallErrorTruncatedFlag verifies the Truncated field is
// propagated to the APICallError.
func TestBuildAPICallErrorTruncatedFlag(t *testing.T) {
	resp := &http.Response{StatusCode: 503, Header: http.Header{}}
	body := []byte("body bytes")
	apiErr := buildAPICallError(resp, body, true)
	if !apiErr.Truncated {
		t.Errorf("Truncated should be true")
	}
}
