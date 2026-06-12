package openai

import (
	"context"
	"net/http"
	"testing"
)

// Verifies that the responses API forwards seed, temperature, top_p, and
// max_output_tokens for non-reasoning models.
func TestResponsesForwardsSampling(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	temp := 0.7
	topp := 0.9
	maxTokens := 256
	seed := 42
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Temperature:     &temp,
		TopP:            &topp,
		MaxOutputTokens: &maxTokens,
		Seed:            &seed,
		StopSequences:   []string{"STOP"},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if v, ok := body["temperature"].(float64); !ok || v != 0.7 {
		t.Errorf("temperature: %v", body["temperature"])
	}
	if v, ok := body["top_p"].(float64); !ok || v != 0.9 {
		t.Errorf("top_p: %v", body["top_p"])
	}
	if v, ok := body["max_output_tokens"].(float64); !ok || v != 256 {
		t.Errorf("max_output_tokens: %v", body["max_output_tokens"])
	}
	if v, ok := body["seed"].(float64); !ok || v != 42 {
		t.Errorf("seed: %v", body["seed"])
	}
	stop, ok := body["stop"].([]any)
	if !ok || len(stop) != 1 || stop[0] != "STOP" {
		t.Errorf("stop: %v", body["stop"])
	}
}

// Verifies that temperature/top_p are stripped on reasoning models that
// don't support non-reasoning parameters.
func TestResponsesStripsSamplingOnReasoning(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"o1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	temp := 0.7
	topp := 0.9
	result, err := p.Responses("o1").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:    []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Temperature: &temp,
		TopP:        &topp,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, has := body["temperature"]; has {
		t.Errorf("temperature should be stripped on o1: %v", body["temperature"])
	}
	if _, has := body["top_p"]; has {
		t.Errorf("top_p should be stripped on o1: %v", body["top_p"])
	}
}

// Verifies that the responses API forwards reasoning effort and summary
// when opts.Reasoning is supplied.
func TestResponsesForwardsReasoning(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-5","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	effort := "high"
	summary := "auto"
	result, err := p.Responses("gpt-5").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:  []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		Reasoning: &ReasoningConfig{Effort: &effort, Summary: &summary},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	r, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning: %v", body["reasoning"])
	}
	if r["effort"] != "high" {
		t.Errorf("effort: %v", r["effort"])
	}
	if r["summary"] != "auto" {
		t.Errorf("summary: %v", r["summary"])
	}
}

// Verifies that the responses API includes previous_response_id when
// supplied.
func TestResponsesForwardsPreviousResponseID(t *testing.T) {
	respBody := `{"id":"r","created_at":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	prevID := "resp_abc"
	result, err := p.Responses("gpt-4o").DoGenerate(context.Background(), ResponsesGenerateOptions{
		Messages:           []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
		PreviousResponseID: &prevID,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["previous_response_id"] != "resp_abc" {
		t.Errorf("previous_response_id: %v", body["previous_response_id"])
	}
}
