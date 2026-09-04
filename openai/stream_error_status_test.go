package openai

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

// TestChatStreamErrorSurfacesAPICallError verifies that a chat-completion
// stream whose first chunk carries an error object (delivered with HTTP
// 200) surfaces a *APICallError with a status code derived from the
// error type, per spec.
func TestChatStreamErrorSurfacesAPICallError(t *testing.T) {
	sseBody := "data: {\"id\":\"r\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[],\"error\":{\"message\":\"insufficient credits\",\"type\":\"insufficient_quota\",\"code\":\"insufficient_quota\"}}\n\n"
	sresp := &StreamResponse{}
	parts := make(chan StreamPart, 8)
	m := &openaiChatLanguageModel{modelID: "gpt-4o"}
	go func() {
		m.runChatStream(context.Background(), makeStreamResp(sseBody), parts, sresp, nil, StreamOptions{})
	}()
	var got StreamError
	for p := range parts {
		if se, ok := p.(StreamError); ok {
			got = se
			break
		}
	}
	if got.Err == nil {
		t.Fatal("expected StreamError, got none")
	}
	apiErr, ok := got.Err.(*APICallError)
	if !ok {
		t.Fatalf("expected *APICallError, got %T: %v", got.Err, got.Err)
	}
	if apiErr.Status != 429 {
		t.Errorf("Status: %d, want 429 (derived from insufficient_quota)", apiErr.Status)
	}
	if !strings.Contains(apiErr.Message, "insufficient") {
		t.Errorf("Message: %q", apiErr.Message)
	}
	if apiErr.Type != "insufficient_quota" {
		t.Errorf("Type: %q", apiErr.Type)
	}
}

// TestResponsesStreamErrorSurfacesAPICallError verifies that a
// responses stream whose first event is a wire "error" surfaces a
// *StreamProviderError wrapping a *APICallError with a status code
// derived from the error type.
func TestResponsesStreamErrorSurfacesAPICallError(t *testing.T) {
	sseBody := "data: {\"type\":\"error\",\"error\":{\"message\":\"forbidden\",\"type\":\"permission_error\",\"code\":\"forbidden\"}}\n\n"
	parts := make(chan StreamPart, 8)
	m := &openaiResponsesModel{modelID: "gpt-4o"}
	go func() {
		m.runResponsesStream(context.Background(), makeStreamResp(sseBody), parts, &StreamResponse{}, nil, ResponsesStreamOptions{})
	}()
	var got StreamError
	for p := range parts {
		if se, ok := p.(StreamError); ok {
			got = se
			break
		}
	}
	if got.Err == nil {
		t.Fatal("expected StreamError, got none")
	}
	spe, ok := got.Err.(*StreamProviderError)
	if !ok {
		t.Fatalf("expected *StreamProviderError, got %T: %v", got.Err, got.Err)
	}
	if spe.Provider != "openai" {
		t.Errorf("Provider: %q, want openai", spe.Provider)
	}
	apiErr, ok := spe.Unwrap().(*APICallError)
	if !ok {
		t.Fatalf("expected wrapped *APICallError, got %T", spe.Unwrap())
	}
	if apiErr.Status != 403 {
		t.Errorf("Status: %d, want 403 (derived from permission_error)", apiErr.Status)
	}
}

// makeStreamResp builds an httpStreamResponse for stream tests.
func makeStreamResp(sseBody string) *httpStreamResponse {
	return &httpStreamResponse{
		Headers: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:    io.NopCloser(strings.NewReader(sseBody)),
	}
}
