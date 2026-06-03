package internal

import (
	"testing"

	anthropic "github.com/shepard-labs/go-ai-sdk-anthropic/anthropic"
)

func TestSSEParsePing(t *testing.T) {
	got, err := ParseStreamEvent(StreamEvent{Type: "ping", Data: []byte(`{}`)})
	if err != nil || got != nil {
		t.Fatalf("got = %#v err = %v", got, err)
	}
}

func TestSSEParseError(t *testing.T) {
	got, err := ParseStreamEvent(StreamEvent{Type: "error", Data: []byte(`{"error":{"type":"api_error","message":"failed"}}`)})
	if err != nil {
		t.Fatal(err)
	}
	streamErr := got.(anthropic.StreamError)
	apiErr := streamErr.Err.(*anthropic.APICallError)
	if apiErr.Status != 500 || apiErr.Message != "failed" {
		t.Fatalf("err = %#v", apiErr)
	}
}

func TestSSEParseOverloadedError(t *testing.T) {
	got, err := ParseStreamEvent(StreamEvent{Type: "error", Data: []byte(`{"error":{"type":"overloaded_error","message":"busy"}}`)})
	if err != nil {
		t.Fatal(err)
	}
	streamErr := got.(anthropic.StreamError)
	apiErr := streamErr.Err.(*anthropic.APICallError)
	if apiErr.Status != 529 || apiErr.Message != "busy" || !apiErr.Retryable {
		t.Fatalf("err = %#v", apiErr)
	}
}
