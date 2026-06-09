// Package openaicompatible implements a configurable, generic provider for any
// API shaped like OpenAI's API. It supports chat language models, legacy
// completion models, embedding models, and image generation/edit models from a
// single provider instance.
//
// # Creating a Provider
//
// Use [CreateOpenAICompatible] with a [ProviderSettings] value. Both BaseURL
// and Name are required; missing either causes [Err] to return an error and
// every model call to fail before making HTTP requests.
//
//	p := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
//	    BaseURL: "https://api.example.com/v1",
//	    Name:    "example",
//	})
//
// # Authentication
//
// Supply APIKey to include an Authorization: Bearer header on every request.
// The package does not read API keys from environment variables and does not
// support Anthropic-style auth tokens.
//
//	p := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
//	    BaseURL: "https://api.example.com/v1",
//	    Name:    "example",
//	    APIKey:  os.Getenv("MY_API_KEY"),
//	})
//
// # Chat Generation
//
// Obtain a chat model and call [LanguageModel.DoGenerate]:
//
//	model := p.Chat("gpt-4o")
//	result, err := model.DoGenerate(ctx, openaicompatible.GenerateOptions{
//	    Messages: []openaicompatible.Message{
//	        openaicompatible.UserMessage{Content: []openaicompatible.UserContent{
//	            openaicompatible.TextContent{Text: "Hello!"},
//	        }},
//	    },
//	})
//
// # Embeddings
//
// Obtain an embedding model and call [EmbeddingModel.DoEmbed]:
//
//	emb := p.Embedding("text-embedding-3-small")
//	result, err := emb.DoEmbed(ctx, openaicompatible.EmbedOptions{
//	    Values: []string{"hello world", "foo bar"},
//	})
//
// # Image Generation
//
// Obtain an image model and call [ImageModel.DoGenerate]:
//
//	img := p.Image("dall-e-3")
//	result, err := img.DoGenerate(ctx, openaicompatible.ImageGenerateOptions{
//	    Prompt: "A futuristic city at night",
//	    N:      1,
//	    Size:   "1024x1024",
//	})
//
// # Streaming with IncludeRawChunks
//
// Use [LanguageModel.DoStream] for streaming responses. Set IncludeRawChunks
// in [StreamOptions] to receive raw SSE payloads as [StreamRaw] parts before
// each parsed event:
//
//	result, err := model.DoStream(ctx, openaicompatible.StreamOptions{
//	    GenerateOptions:  openaicompatible.GenerateOptions{Messages: msgs},
//	    IncludeRawChunks: true,
//	})
//	for part := range result.Stream {
//	    switch p := part.(type) {
//	    case openaicompatible.StreamRaw:
//	        // p.Raw contains raw SSE JSON bytes; p.Decoded is the decoded map.
//	    case openaicompatible.StreamTextDelta:
//	        fmt.Print(p.Text)
//	    case openaicompatible.StreamFinish:
//	        fmt.Println("done:", p.FinishReason.Unified)
//	    }
//	}
//
// # Provider Options
//
// Provider-specific options are passed as [ProviderOptions] (a
// map[string]map[string]any). Use the provider name or its camelCase variant
// as the outer key. Chat and embedding models also accept the legacy
// "openai-compatible" and "openaiCompatible" compatibility keys (the former
// emits a deprecation warning):
//
//	opts := openaicompatible.ProviderOptions{
//	    "example": {"reasoning_effort": "high"},
//	}
//
// # Version
//
// The package exports a [Version] constant used in the User-Agent header sent
// with every request:
//
//	User-Agent: ai-sdk-go/openai-compatible/<Version>
package openaicompatible
