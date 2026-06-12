package openai

import (
	"context"
	"testing"
)

// TestResponsesStreamProviderMetadataOnTextStart verifies that
// StreamTextStart carries itemId + phase and StreamTextEnd carries
// annotations in ProviderMetadata["openai"] per the spec.
func TestResponsesStreamProviderMetadataOnTextStart(t *testing.T) {
	sse := "" +
		"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"created_at\":1}}\n\n" +
		"data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg-7\",\"phase\":\"final_answer\"}}\n\n" +
		"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg-7\",\"delta\":\"hi\"}\n\n" +
		"data: {\"type\":\"response.output_text.annotation.added\",\"item_id\":\"msg-7\",\"annotation\":{\"type\":\"url_citation\",\"url\":\"https://e.com\",\"title\":\"E\"}}\n\n" +
		"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg-7\"}}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"service_tier\":\"default\"}}\n\n" +
		"data: [DONE]\n\n"
	parts := make(chan StreamPart, 16)
	m := &openaiResponsesModel{modelID: "gpt-5"}
	sresp := &StreamResponse{}
	go func() {
		m.runResponsesStream(context.Background(), makeStreamResp(sse), parts, sresp, nil, ResponsesStreamOptions{})
	}()
	var start StreamTextStart
	var end StreamTextEnd
	var finish StreamFinish
	for p := range parts {
		switch v := p.(type) {
		case StreamTextStart:
			start = v
		case StreamTextEnd:
			end = v
		case StreamFinish:
			finish = v
		}
	}
	om, _ := start.ProviderMetadata["openai"].(map[string]any)
	if om["itemId"] != "msg-7" {
		t.Errorf("start.itemId = %v", om["itemId"])
	}
	if om["phase"] != "final_answer" {
		t.Errorf("start.phase = %v", om["phase"])
	}
	em, _ := end.ProviderMetadata["openai"].(map[string]any)
	if em == nil {
		t.Fatalf("expected end.ProviderMetadata[openai], got %v", end.ProviderMetadata)
	}
	if _, has := em["annotations"]; !has {
		t.Errorf("expected annotations on end, got: %v", em)
	}
	fm, _ := finish.ProviderMetadata["openai"].(map[string]any)
	if fm["responseId"] != "resp_1" {
		t.Errorf("finish.responseId = %v", fm["responseId"])
	}
	if fm["serviceTier"] != "default" {
		t.Errorf("finish.serviceTier = %v", fm["serviceTier"])
	}
}

// TestResponsesStreamProviderMetadataOnReasoningStart verifies that
// StreamReasoningStart carries itemId + reasoningEncryptedContent in
// ProviderMetadata["openai"] per the spec.
func TestResponsesStreamProviderMetadataOnReasoningStart(t *testing.T) {
	sse := "" +
		"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"created_at\":1}}\n\n" +
		"data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"rsn-1\",\"encrypted_content\":\"ENC\"}}\n\n" +
		"data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rsn-1\",\"delta\":\"hm\"}\n\n" +
		"data: {\"type\":\"response.reasoning_summary_part.done\",\"item_id\":\"rsn-1\",\"summary_index\":0}\n\n" +
		"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"rsn-1\"}}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n" +
		"data: [DONE]\n\n"
	parts := make(chan StreamPart, 16)
	m := &openaiResponsesModel{modelID: "gpt-5", store: boolPtr(false)}
	sresp := &StreamResponse{}
	go func() {
		m.runResponsesStream(context.Background(), makeStreamResp(sse), parts, sresp, nil, ResponsesStreamOptions{})
	}()
	var start StreamReasoningStart
	var end StreamReasoningEnd
	for p := range parts {
		switch v := p.(type) {
		case StreamReasoningStart:
			start = v
		case StreamReasoningEnd:
			end = v
		}
	}
	om, _ := start.ProviderMetadata["openai"].(map[string]any)
	if om["itemId"] != "rsn-1" {
		t.Errorf("start.itemId = %v", om["itemId"])
	}
	if om["reasoningEncryptedContent"] != "ENC" {
		t.Errorf("start.reasoningEncryptedContent = %v", om["reasoningEncryptedContent"])
	}
	em, _ := end.ProviderMetadata["openai"].(map[string]any)
	if em == nil {
		t.Fatalf("expected end.ProviderMetadata[openai], got %v", end.ProviderMetadata)
	}
	if em["itemId"] != "rsn-1" {
		t.Errorf("end.itemId = %v", em["itemId"])
	}
	if em["reasoningEncryptedContent"] != "ENC" {
		t.Errorf("end.reasoningEncryptedContent = %v", em["reasoningEncryptedContent"])
	}
}
