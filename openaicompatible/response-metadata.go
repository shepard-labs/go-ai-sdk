package openaicompatible

import (
	"net/http"
	"time"
)

func responseMetadata(id, modelID string, created *int64, headers http.Header, body []byte) ResponseMetadata {
	var ts *time.Time
	if created != nil {
		value := time.Unix(*created, 0)
		ts = &value
	}
	return ResponseMetadata{
		ID:        id,
		ModelID:   modelID,
		Timestamp: ts,
		Headers:   cloneHeader(headers),
		Body:      append([]byte(nil), body...),
	}
}
