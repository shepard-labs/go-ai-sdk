package openrouter_test

import (
	"context"
	"fmt"
	"net/http"

	"github.com/shepard-labs/go-ai-sdk/openrouter"
)

func ExampleCreateOpenRouter_chat() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
		APIKey:  "sk-or-example",
		AppName: "Example App",
		AppURL:  "https://example.test",
	})

	_, _ = provider.Chat("openai/gpt-4o-mini").DoGenerate(context.Background(), openrouter.GenerateOptions{
		Messages: []openrouter.Message{
			openrouter.UserMessage{Content: []openrouter.UserContent{openrouter.TextContent{Text: "Hello"}}},
		},
	})
}

func ExampleProvider_Embedding() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{APIKey: "sk-or-example"})
	_, _ = provider.Embedding("text-embedding-3-small").DoEmbed(context.Background(), openrouter.EmbedOptions{
		Values: []string{"first", "second"},
	})
}

func ExampleProvider_Image() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{APIKey: "sk-or-example"})
	_, _ = provider.Image("google/gemini-2.5-flash-image-preview").DoGenerate(context.Background(), openrouter.ImageGenerateOptions{
		Prompt:      "A small cabin under northern lights",
		AspectRatio: "16:9",
	})
}

func ExampleProvider_VideoModel() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{APIKey: "sk-or-example"})
	_, _ = provider.VideoModel("google/veo-3").DoGenerate(context.Background(), openrouter.VideoGenerateOptions{
		Prompt:     "A cinematic shot of waves at sunrise",
		Resolution: "720p",
		Duration:   "5s",
	})
}

func ExampleProviderSettings_headers() {
	provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
		Headers: http.Header{"Authorization": {"Bearer caller-managed-token"}},
	})
	fmt.Println(provider.Name())
	// Output: openrouter
}
