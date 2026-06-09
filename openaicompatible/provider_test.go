package openaicompatible

import (
	"context"
	"errors"
	"net/http"
	"os"
	"testing"
)

func TestProviderCreateValid(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test/v1", Name: "acme"})
	if p.Name() != "acme" {
		t.Fatalf("Name() = %q", p.Name())
	}
	if p.Err() != nil {
		t.Fatalf("Err() = %v", p.Err())
	}
}

func TestProviderMissingBaseURLAndName(t *testing.T) {
	t.Run("missing base URL", func(t *testing.T) {
		p := CreateOpenAICompatible(ProviderSettings{Name: "acme"})
		if !errors.Is(p.Err(), ErrMissingBaseURL) {
			t.Fatalf("Err() = %v, want ErrMissingBaseURL", p.Err())
		}
		_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{})
		if !errors.Is(err, ErrMissingBaseURL) {
			t.Fatalf("model error = %v, want ErrMissingBaseURL", err)
		}
	})

	t.Run("missing name", func(t *testing.T) {
		p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: " \t"})
		if !errors.Is(p.Err(), ErrMissingName) {
			t.Fatalf("Err() = %v, want ErrMissingName", p.Err())
		}
		_, err := p.Embedding("m").DoEmbed(context.Background(), EmbedOptions{})
		if !errors.Is(err, ErrMissingName) {
			t.Fatalf("model error = %v, want ErrMissingName", err)
		}
	})

	t.Run("joined", func(t *testing.T) {
		p := CreateOpenAICompatible(ProviderSettings{})
		if !errors.Is(p.Err(), ErrMissingBaseURL) || !errors.Is(p.Err(), ErrMissingName) {
			t.Fatalf("Err() = %v, want both missing errors", p.Err())
		}
	})
}

func TestProviderModelFactories(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test/", Name: "acme"})
	for name, model := range map[string]LanguageModel{
		"Model":         p.Model("free form/chat"),
		"LanguageModel": p.LanguageModel("free form/chat"),
		"ChatModel":     p.ChatModel("free form/chat"),
		"Chat":          p.Chat("free form/chat"),
	} {
		if model.Provider() != "acme.chat" {
			t.Fatalf("%s provider = %q", name, model.Provider())
		}
		if model.ModelID() != "free form/chat" {
			t.Fatalf("%s model ID = %q", name, model.ModelID())
		}
	}
	for name, model := range map[string]LanguageModel{
		"CompletionModel": p.CompletionModel("free form/completion"),
		"Completion":      p.Completion("free form/completion"),
	} {
		if model.Provider() != "acme.completion" {
			t.Fatalf("%s provider = %q", name, model.Provider())
		}
	}
	for name, model := range map[string]EmbeddingModel{
		"EmbeddingModel":     p.EmbeddingModel("free form/embedding"),
		"Embedding":          p.Embedding("free form/embedding"),
		"TextEmbeddingModel": p.TextEmbeddingModel("free form/embedding"),
	} {
		if model.Provider() != "acme.embedding" {
			t.Fatalf("%s provider = %q", name, model.Provider())
		}
	}
	for name, model := range map[string]ImageModel{
		"ImageModel": p.ImageModel("free form/image"),
		"Image":      p.Image("free form/image"),
	} {
		if model.Provider() != "acme.image" {
			t.Fatalf("%s provider = %q", name, model.Provider())
		}
	}
}

func TestProviderAliasesCompile(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme"})
	var _ ChatLanguageModel = p.Chat("chat")
	var _ CompletionLanguageModel = p.Completion("completion")
	var _ EmbeddingModel = p.Embedding("embedding")
	var _ ImageModel = p.Image("image")
	var _ *OpenAICompatibleChatLanguageModel = p.Chat("chat").(*openAICompatibleChatLanguageModel)
	var _ *OpenAICompatibleCompletionLanguageModel = p.Completion("completion").(*openAICompatibleCompletionLanguageModel)
	var _ *OpenAICompatibleEmbeddingModel = p.Embedding("embedding").(*openAICompatibleEmbeddingModel)
	var _ *OpenAICompatibleImageModel = p.Image("image").(*openAICompatibleImageModel)
}

func TestProviderDoesNotReadEnvironmentForAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "from-env")
	t.Setenv("ANTHROPIC_API_KEY", "from-env")
	f := &recordingFetcher{responses: []*http.Response{response(200, `{}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}}).(*openAICompatibleProvider)
	_, err := p.executeJSON(context.Background(), endpointChatCompletions, []byte(`{}`), nil)
	if err != nil {
		t.Fatalf("executeJSON error = %v", err)
	}
	if got := f.requests[0].Header.Get("Authorization"); got != "" {
		t.Fatalf("Authorization = %q, want empty", got)
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Fatal("test environment was not set")
	}
}
