// Package google implements the Google Generative AI provider for the Go AI
// SDK. It supports language models (Gemini), embedding models, image generation
// (Imagen and Gemini image), video generation (Veo), speech synthesis, and the
// Files API.
//
// # Quick Start
//
// The default provider instance reads GOOGLE_GENERATIVE_AI_API_KEY from the
// environment automatically:
//
//	model := google.Google.LanguageModel("gemini-2.0-flash")
//	result, err := model.DoGenerate(ctx, google.GenerateOptions{
//	    Messages: []google.Message{
//	        google.UserMessage{Content: []google.UserContent{
//	            google.TextContent{Text: "Hello!"},
//	        }},
//	    },
//	})
//
// # Creating a Custom Provider
//
// Use [CreateGoogle] to pass an explicit API key or customize transport settings:
//
//	p := google.CreateGoogle(google.ProviderSettings{
//	    APIKey: os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"),
//	})
//
// # Authentication
//
// By default, CreateGoogle sends an x-goog-api-key header on every request.
// Set UseHeaderAuth to false to suppress that header (for proxy setups that
// inject auth separately). The API key is read from ProviderSettings.APIKey;
// if empty, the GOOGLE_GENERATIVE_AI_API_KEY environment variable is used.
// If neither is set, [Provider.Err] returns [ErrMissingAPIKey] and every model
// call fails before issuing HTTP requests.
//
// # Model Families
//
//   - Language models: [Provider.LanguageModel] / [Provider.Chat] / [Provider.Model]
//   - Embedding models: [Provider.EmbeddingModel] / [Provider.Embedding]
//   - Image models: [Provider.ImageModel] / [Provider.Image]
//   - Video models: [Provider.VideoModel] / [Provider.Video]
//   - Speech models: [Provider.SpeechModel] / [Provider.Speech]
//   - Files API: [Provider.Files]
//   - Provider tools: [Provider.Tools]
//
// # Base URL
//
// The default base URL is https://generativelanguage.googleapis.com/v1beta.
// Override via ProviderSettings.BaseURL, e.g. to point at a Vertex AI endpoint.
//
// # Version
//
// The package version is reported in the User-Agent header as
// ai-sdk-go/google/<Version>.
//
// # Provider-Specific Options
//
// Google-specific options are passed as [ProviderOptions] under the "google"
// key. Recognized options are described by [GoogleOptions]; any unrecognized
// keys are forwarded as-is in the request body.
package google
