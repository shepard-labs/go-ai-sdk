package internal

import (
	"encoding/json"
	"net/http"

	anthropic "github.com/shepard-labs/go-ai-sdk/anthropic"
)

type StreamEvent struct {
	Type string
	Data []byte
}

func ParseStreamEvent(event StreamEvent) (any, error) {
	if event.Type == "ping" {
		return nil, nil
	}
	if event.Type == "error" {
		var payload struct {
			Error anthropic.APIError `json:"error"`
		}
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			return nil, err
		}
		status := http.StatusInternalServerError
		if payload.Error.Type == "overloaded_error" {
			status = 529
		}
		return anthropic.StreamError{Err: &anthropic.APICallError{Status: status, Message: payload.Error.Message, Retryable: true}}, nil
	}
	return event, nil
}
