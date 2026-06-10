package google

import (
	"net/http"
	"time"
)

// responseMetadata builds a ResponseMetadata value from the HTTP response
// fields and optional raw body bytes. modelID and id are read from the
// response body by the caller and passed in.
func responseMetadata(headers http.Header, body []byte, id, modelID string) ResponseMetadata {
	ts := time.Now()
	return ResponseMetadata{
		ID:        id,
		ModelID:   modelID,
		Timestamp: &ts,
		Headers:   cloneHeader(headers),
		Body:      body,
	}
}

// streamResponseFromHeaders builds a StreamResponse from the initial HTTP
// response headers of a streaming call.
func streamResponseFromHeaders(headers http.Header) *StreamResponse {
	ts := time.Now()
	return &StreamResponse{
		Timestamp: &ts,
		Headers:   cloneHeader(headers),
	}
}
