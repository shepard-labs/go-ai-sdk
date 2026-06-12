package openai

import (
	"context"
	"net/http"
	"testing"
)

// TestCompletionGenerateBasic verifies a basic completion call.
func TestCompletionGenerateBasic(t *testing.T) {
	respBody := `{"id":"cmpl-1","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"hi there","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("content len = %d", len(result.Content))
	}
	if tc, ok := result.Content[0].(TextContent); !ok || tc.Text != "hi there" {
		t.Errorf("text: %#v", result.Content[0])
	}
	if result.FinishReason.Raw != "stop" {
		t.Errorf("finish: %+v", result.FinishReason)
	}
}

// TestCompletionRequestBody verifies the request body shape.
func TestCompletionRequestBody(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	maxTokens := 64
	seed := 42
	temp := 0.5
	topp := 0.9
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages:        []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
		MaxOutputTokens: &maxTokens,
		Seed:            &seed,
		Temperature:     &temp,
		TopP:            &topp,
		StopSequences:   []string{"STOP"},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["model"] != "gpt-3.5-turbo-instruct" {
		t.Errorf("model: %v", body["model"])
	}
	if v, ok := body["max_tokens"].(float64); !ok || v != 64 {
		t.Errorf("max_tokens: %v", body["max_tokens"])
	}
	if v, ok := body["seed"].(float64); !ok || v != 42 {
		t.Errorf("seed: %v", body["seed"])
	}
	if v, ok := body["temperature"].(float64); !ok || v != 0.5 {
		t.Errorf("temperature: %v", body["temperature"])
	}
	if v, ok := body["top_p"].(float64); !ok || v != 0.9 {
		t.Errorf("top_p: %v", body["top_p"])
	}
	stop, ok := body["stop"].([]any)
	if !ok || len(stop) != 2 || stop[0] != "\n<user>:" || stop[1] != "STOP" {
		t.Errorf("stop: %v", body["stop"])
	}
	if _, has := body["prompt"]; !has {
		t.Errorf("missing prompt: %v", body)
	}
}

// TestCompletionRequestWithPenalties verifies frequency/presence penalty
// fields are forwarded.
func TestCompletionRequestWithPenalties(t *testing.T) {
	respBody := `{"id":"x","choices":[],"usage":{}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	fp := 0.5
	pp := 0.25
	result, err := p.Completion("gpt-3.5-turbo-instruct").DoGenerate(context.Background(), GenerateOptions{
		Messages:          []Message{UserMessage{Content: []UserContent{TextContent{Text: "x"}}}},
		FrequencyPenalty:  &fp,
		PresencePenalty:   &pp,
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if v, ok := body["frequency_penalty"].(float64); !ok || v != 0.5 {
		t.Errorf("frequency_penalty: %v", body["frequency_penalty"])
	}
	if v, ok := body["presence_penalty"].(float64); !ok || v != 0.25 {
		t.Errorf("presence_penalty: %v", body["presence_penalty"])
	}
}

// TestCompletionStreamSimpleText verifies the completion stream emits
// text deltas and a final StreamFinish.
func TestCompletionStreamSimpleText(t *testing.T) {
	stream := "data: {\"id\":\"x\",\"created\":1,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"text\":\"hi\",\"index\":0}]}\n\n" +
		"data: {\"id\":\"x\",\"created\":1,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"text\":\" there\",\"index\":0}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n" +
		"data: [DONE]\n\n"
	fetcher := &streamingFetcher{body: stream}
	p := newOpenAIForTest(fetcher, "https://example.test/v1")
	res, err := p.Completion("gpt-3.5-turbo-instruct").DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	var collected []StreamPart
	for p := range res.Parts {
		collected = append(collected, p)
	}
	if len(collected) == 0 {
		t.Fatal("no parts")
	}
	if _, ok := collected[0].(StreamStart); !ok {
		t.Errorf("first part: %T", collected[0])
	}
	foundHi := false
	for _, p := range collected {
		if d, ok := p.(StreamTextDelta); ok {
			if d.Text == "hi" || d.Text == " there" {
				foundHi = true
			}
		}
	}
	if !foundHi {
		t.Errorf("no text deltas found")
	}
	hasFinish := false
	for _, p := range collected {
		if _, ok := p.(StreamFinish); ok {
			hasFinish = true
		}
	}
	if !hasFinish {
		t.Errorf("no StreamFinish")
	}
}
