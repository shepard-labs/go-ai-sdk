// Package openai implements the OpenAI provider for the go-ai-sdk.
//
// It supports chat completions, the Responses API, legacy completions,
// embeddings, image generation and editing, speech synthesis, audio
// transcription, file upload, skill upload, and a Realtime client-secret
// helper. Provider-defined tools (web_search, file_search, code_interpreter,
// image_generation, local_shell, shell, apply_patch, MCP, custom, tool_search)
// are exposed via the [github.com/shepard-labs/go-ai-sdk/openai/tools]
// subpackage and are recognized by the chat and Responses models.
//
// # Creating a Provider
//
// Use [CreateOpenAI] with a [ProviderSettings] value. The provider reads
// $OPENAI_API_KEY and $OPENAI_BASE_URL when settings are not supplied.
//
//	p := openai.CreateOpenAI(openai.ProviderSettings{Organization: "org_..."})
//
// # Chat Generation
//
//	model := p.Chat("gpt-4o")
//	result, err := model.DoGenerate(ctx, openai.GenerateOptions{
//	    Messages: []openaicompatible.Message{...},
//	})
//
// The package also exposes [openai.OpenAI] for a default provider instance
// using environment variables.
package openai
