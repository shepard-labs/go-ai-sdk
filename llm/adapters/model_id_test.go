package adapters

import (
	"context"
	"errors"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/google"
	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/openai"
	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

func TestOpenAIPerRequestModelID(t *testing.T) {
	defaultModel := &fakeOpenAIModel{modelID: "default", result: &openai.GenerateResult{FinishReason: openai.FinishReason{Unified: "stop"}}}
	requestedModel := &fakeOpenAIModel{modelID: "requested", result: &openai.GenerateResult{FinishReason: openai.FinishReason{Unified: "stop"}}, stream: emptyOpenAIStream()}
	adapter := &OpenAIAdapter{model: defaultModel, defaultModelID: "default", newModel: func(modelID string) openai.LanguageModel {
		if modelID != "requested" {
			t.Fatalf("newModel modelID = %q, want requested", modelID)
		}
		return requestedModel
	}}
	assertFactoryModelID(t, adapter, requestedModel, "requested")
	assertDirectModelID(t, NewOpenAIAdapter(&fakeOpenAIModel{modelID: "default", result: defaultModel.result}), "default")
}

func TestAnthropicPerRequestModelID(t *testing.T) {
	defaultModel := &fakeAnthropicModel{modelID: "default", result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}}
	requestedModel := &fakeAnthropicModel{modelID: "requested", result: &anthropic.GenerateResult{FinishReason: anthropic.FinishReasonStop}, stream: emptyAnthropicStream()}
	adapter := &AnthropicAdapter{model: defaultModel, defaultModelID: "default", newModel: func(modelID string) anthropic.LanguageModel {
		if modelID != "requested" {
			t.Fatalf("newModel modelID = %q, want requested", modelID)
		}
		return requestedModel
	}}
	assertFactoryModelID(t, adapter, requestedModel, "requested")
	assertDirectModelID(t, NewAnthropicAdapter(&fakeAnthropicModel{modelID: "default", result: defaultModel.result}), "default")
}

func TestGooglePerRequestModelID(t *testing.T) {
	defaultModel := &fakeGoogleModel{modelID: "default", result: &google.GenerateResult{FinishReason: google.FinishReason{Unified: "stop"}}}
	requestedModel := &fakeGoogleModel{modelID: "requested", result: &google.GenerateResult{FinishReason: google.FinishReason{Unified: "stop"}}, stream: emptyGoogleStream()}
	adapter := &GoogleAdapter{model: defaultModel, defaultModelID: "default", newModel: func(modelID string) google.LanguageModel {
		if modelID != "requested" {
			t.Fatalf("newModel modelID = %q, want requested", modelID)
		}
		return requestedModel
	}}
	assertFactoryModelID(t, adapter, requestedModel, "requested")
	assertDirectModelID(t, NewGoogleAdapter(&fakeGoogleModel{modelID: "default", result: defaultModel.result}), "default")
}

func TestOpenAICompatiblePerRequestModelID(t *testing.T) {
	defaultModel := &fakeOpenAICompatibleModel{modelID: "default", result: &openaicompatible.GenerateResult{FinishReason: openaicompatible.FinishReason{Unified: "stop"}}}
	requestedModel := &fakeOpenAICompatibleModel{modelID: "requested", result: &openaicompatible.GenerateResult{FinishReason: openaicompatible.FinishReason{Unified: "stop"}}, stream: emptyOpenAICompatibleStream()}
	adapter := &OpenAICompatibleAdapter{model: defaultModel, defaultModelID: "default", newModel: func(modelID string) openaicompatible.LanguageModel {
		if modelID != "requested" {
			t.Fatalf("newModel modelID = %q, want requested", modelID)
		}
		return requestedModel
	}}
	assertFactoryModelID(t, adapter, requestedModel, "requested")
	assertDirectModelID(t, NewOpenAICompatibleAdapter(&fakeOpenAICompatibleModel{modelID: "default", result: defaultModel.result}), "default")
}

func TestOpenRouterPerRequestModelID(t *testing.T) {
	defaultModel := &fakeOpenRouterModel{modelID: "default", result: &openrouter.GenerateResult{FinishReason: openrouter.FinishReasonStop}}
	requestedModel := &fakeOpenRouterModel{modelID: "requested", result: &openrouter.GenerateResult{FinishReason: openrouter.FinishReasonStop}, stream: emptyOpenRouterStream()}
	adapter := &OpenRouterAdapter{model: defaultModel, defaultModelID: "default", newModel: func(modelID string) openrouter.LanguageModel {
		if modelID != "requested" {
			t.Fatalf("newModel modelID = %q, want requested", modelID)
		}
		return requestedModel
	}}
	assertFactoryModelID(t, adapter, requestedModel, "requested")
	assertDirectModelID(t, NewOpenRouterAdapter(&fakeOpenRouterModel{modelID: "default", result: defaultModel.result}), "default")
}

func assertFactoryModelID(t *testing.T, client llm.Client, requested any, modelID string) {
	t.Helper()
	result, err := client.Generate(context.Background(), GenerateOptions{ModelID: modelID})
	if err != nil {
		t.Fatalf("Generate requested model error = %v", err)
	}
	if result.Response.ModelID != modelID {
		t.Fatalf("response ModelID = %q, want %q", result.Response.ModelID, modelID)
	}
	if calls := generatedCalls(requested); calls != 1 {
		t.Fatalf("requested model calls = %d, want 1", calls)
	}
	ch, err := client.Stream(context.Background(), GenerateOptions{ModelID: modelID})
	if err != nil {
		t.Fatalf("Stream requested model error = %v", err)
	}
	if ch != nil {
		for range ch {
		}
	}
	if capabilities, ok := client.(llm.CapabilitiesProvider); ok {
		if got := capabilities.Capabilities().ModelID; got != "default" {
			t.Fatalf("Capabilities ModelID = %q, want default", got)
		}
	}
	if capabilities, ok := client.(llm.ModelCapabilities); ok {
		if got := capabilities.CapabilitiesForModel(modelID).ModelID; got != modelID {
			t.Fatalf("CapabilitiesForModel ModelID = %q, want %q", got, modelID)
		}
	}
}

func assertDirectModelID(t *testing.T, client llm.Client, defaultModelID string) {
	t.Helper()
	if _, err := client.Generate(context.Background(), GenerateOptions{ModelID: defaultModelID}); err != nil {
		t.Fatalf("direct same model Generate error = %v", err)
	}
	_, err := client.Generate(context.Background(), GenerateOptions{ModelID: "other"})
	var unsupported *llm.UnsupportedFeatureError
	if !errors.As(err, &unsupported) || unsupported.Feature != "model_id" {
		t.Fatalf("direct different model error = %v, want model_id UnsupportedFeatureError", err)
	}
}

func generatedCalls(model any) int {
	switch m := model.(type) {
	case *fakeOpenAIModel:
		return m.generateCalls
	case *fakeAnthropicModel:
		return m.generateCalls
	case *fakeGoogleModel:
		return m.generateCalls
	case *fakeOpenAICompatibleModel:
		return m.generateCalls
	case *fakeOpenRouterModel:
		return m.generateCalls
	}
	return 0
}

func emptyOpenAIStream() *openai.StreamResult {
	ch := make(chan openai.StreamPart)
	close(ch)
	return &openai.StreamResult{Parts: ch}
}

func emptyAnthropicStream() *anthropic.StreamResult {
	ch := make(chan anthropic.StreamPart)
	close(ch)
	return &anthropic.StreamResult{Parts: ch}
}

func emptyGoogleStream() *google.StreamResult {
	ch := make(chan google.StreamPart)
	close(ch)
	return &google.StreamResult{Parts: ch}
}

func emptyOpenAICompatibleStream() *openaicompatible.StreamResult {
	ch := make(chan openaicompatible.StreamPart)
	close(ch)
	return &openaicompatible.StreamResult{Parts: ch}
}

func emptyOpenRouterStream() *openrouter.StreamResult {
	ch := make(chan openrouter.StreamPart)
	close(ch)
	return &openrouter.StreamResult{Parts: ch}
}
