// Package openrouter provides a first-class OpenRouter provider for the Go AI SDK.
//
// The package is standalone and does not wrap another provider package. It
// implements OpenRouter-specific request options, provider routing, app
// attribution headers, BYOK headers, usage accounting, multimodal chat content,
// embeddings, image generation through chat completions, and experimental video
// generation/polling.
//
// CreateOpenRouter defaults to compatible mode, which omits strict upstream
// stream_options fields. The package-level OpenRouter value uses strict mode and
// includes stream_options.include_usage for chat and completion streams.
//
// Embedding MaxEmbeddingsPerCall returns a package default of 2048 because the
// Go interface requires an integer. Upstream leaves this undefined.
//
// Video is exposed through a package-local VideoModel abstraction because the
// repository has no shared video model interface.
package openrouter
