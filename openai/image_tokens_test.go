package openai

import (
	"context"
	"net/http"
	"reflect"
	"testing"
)

func TestDistributeTokenDetailsEvenSplit(t *testing.T) {
	img, txt := distributeTokenDetails(100, 50, 4)
	wantImg := []int{25, 25, 25, 25}
	wantTxt := []int{12, 12, 12, 14} // 50/4 = 12 base, 2 remainder to last
	if !reflect.DeepEqual(img, wantImg) {
		t.Errorf("image: got %v, want %v", img, wantImg)
	}
	if !reflect.DeepEqual(txt, wantTxt) {
		t.Errorf("text: got %v, want %v", txt, wantTxt)
	}
}

func TestDistributeTokenDetailsRemainderToLast(t *testing.T) {
	img, txt := distributeTokenDetails(7, 3, 3)
	wantImg := []int{2, 2, 3}
	wantTxt := []int{1, 1, 1}
	if !reflect.DeepEqual(img, wantImg) {
		t.Errorf("image: got %v, want %v", img, wantImg)
	}
	if !reflect.DeepEqual(txt, wantTxt) {
		t.Errorf("text: got %v, want %v", txt, wantTxt)
	}
}

func TestDistributeTokenDetailsSingleImage(t *testing.T) {
	img, txt := distributeTokenDetails(123, 45, 1)
	if !reflect.DeepEqual(img, []int{123}) {
		t.Errorf("image: got %v, want [123]", img)
	}
	if !reflect.DeepEqual(txt, []int{45}) {
		t.Errorf("text: got %v, want [45]", txt)
	}
}

func TestDistributeTokenDetailsZeroImages(t *testing.T) {
	img, txt := distributeTokenDetails(10, 5, 0)
	if img != nil || txt != nil {
		t.Errorf("expected nil for n=0, got %v, %v", img, txt)
	}
}

func TestImageResponseDistributesTokensToImageMeta(t *testing.T) {
	respBody := `{"data":[{"b64_json":"aGVsbG8="},{"b64_json":"d29ybGQ="},{"b64_json":"Zm9vYmFy"}],"usage":{"input_tokens":30,"output_tokens":100,"total_tokens":130,"input_tokens_details":{"image_tokens":75,"text_tokens":12}}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "3 cats",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata should be set")
	}
	openaiMeta, ok := res.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("openai meta: %v", res.ProviderMetadata["openai"])
	}
	images, ok := openaiMeta["images"].([]map[string]any)
	if !ok {
		t.Fatalf("images meta: %v", openaiMeta["images"])
	}
	if len(images) != 3 {
		t.Fatalf("images: %d", len(images))
	}
	// 75 / 3 = 25 per image (no remainder); 12 / 3 = 4 per image.
	for i, entry := range images {
		if entry["imageTokens"] != 25 {
			t.Errorf("image[%d].imageTokens: %v", i, entry["imageTokens"])
		}
		if entry["textTokens"] != 4 {
			t.Errorf("image[%d].textTokens: %v", i, entry["textTokens"])
		}
	}
}

func TestImageResponseDistributesTokensWithRemainder(t *testing.T) {
	// 7 image_tokens, 3 text_tokens, 3 images → 2/2/3 image, 1/1/1 text.
	respBody := `{"data":[{"b64_json":"aA=="},{"b64_json":"Yg=="},{"b64_json":"Yw=="}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"image_tokens":7,"text_tokens":3}}}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "x",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	openaiMeta := res.ProviderMetadata["openai"].(map[string]any)
	images := openaiMeta["images"].([]map[string]any)
	wantImg := []int{2, 2, 3}
	wantTxt := []int{1, 1, 1}
	for i, entry := range images {
		if entry["imageTokens"] != wantImg[i] {
			t.Errorf("image[%d].imageTokens: %v, want %d", i, entry["imageTokens"], wantImg[i])
		}
		if entry["textTokens"] != wantTxt[i] {
			t.Errorf("image[%d].textTokens: %v, want %d", i, entry["textTokens"], wantTxt[i])
		}
	}
}

// Verifies that the dall-e-3 revised_prompt is surfaced per-image in
// the provider metadata. Per the spec, dall-e-3 returns a revised_prompt
// field that the SDK should preserve.
func TestImageResponseRevisedPromptPerImage(t *testing.T) {
	respBody := `{"data":[{"b64_json":"aGVsbG8=","revised_prompt":"A revised prompt"},{"b64_json":"d29ybGQ=","revised_prompt":"Another revised prompt"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "a cat",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	openaiMeta, ok := res.ProviderMetadata["openai"].(map[string]any)
	if !ok {
		t.Fatalf("openai meta: %v", res.ProviderMetadata["openai"])
	}
	images, ok := openaiMeta["images"].([]map[string]any)
	if !ok {
		t.Fatalf("images meta: %v", openaiMeta["images"])
	}
	if len(images) != 2 {
		t.Fatalf("images: %d", len(images))
	}
	if images[0]["revisedPrompt"] != "A revised prompt" {
		t.Errorf("image[0].revisedPrompt: %v", images[0]["revisedPrompt"])
	}
	if images[1]["revisedPrompt"] != "Another revised prompt" {
		t.Errorf("image[1].revisedPrompt: %v", images[1]["revisedPrompt"])
	}
}
