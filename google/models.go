package google

// Model ID constants for well-known Google Generative AI models.
// Use these constants for autocomplete, documentation, and capability lookups
// via [ModelCapabilitiesForID].
//
// The list is non-exhaustive; any model ID string accepted by the Google API
// can be passed to the model-family factory methods.
const (
	// ---- Gemini 3 / 3.1 ----

	// ModelGemini3ProPreview is the Gemini 3 Pro preview.
	ModelGemini3ProPreview = "gemini-3-pro-preview"
	// ModelGemini3ProImage is the Gemini 3 Pro image preview.
	ModelGemini3ProImage = "gemini-3-pro-image"
	// ModelGemini3FlashPreview is the Gemini 3 Flash preview.
	ModelGemini3FlashPreview = "gemini-3-flash-preview"
	// ModelGemini31ProPreview is the Gemini 3.1 Pro preview.
	ModelGemini31ProPreview = "gemini-3.1-pro-preview"
	// ModelGemini31ProPreviewCustomTools is the Gemini 3.1 Pro preview with custom tools.
	ModelGemini31ProPreviewCustomTools = "gemini-3.1-pro-preview-customtools"
	// ModelGemini31FlashImage is the Gemini 3.1 Flash image preview.
	ModelGemini31FlashImage = "gemini-3.1-flash-image"
	// ModelGemini31FlashLitePreview is the Gemini 3.1 Flash Lite preview.
	ModelGemini31FlashLitePreview = "gemini-3.1-flash-lite-preview"
	// ModelGemini31FlashTTSPreview is the Gemini 3.1 Flash TTS preview.
	ModelGemini31FlashTTSPreview = "gemini-3.1-flash-tts-preview"
	// ModelGemini35Flash is the Gemini 3.5 Flash model.
	ModelGemini35Flash = "gemini-3.5-flash"

	// ---- Latest aliases ----

	// ModelGeminiProLatest is the latest Gemini Pro alias.
	ModelGeminiProLatest = "gemini-pro-latest"
	// ModelGeminiFlashLatest is the latest Gemini Flash alias.
	ModelGeminiFlashLatest = "gemini-flash-latest"
	// ModelGeminiFlashLiteLatest is the latest Gemini Flash Lite alias.
	ModelGeminiFlashLiteLatest = "gemini-flash-lite-latest"

	// ---- Deep Research ----

	// ModelDeepResearchProPreview122025 is the deep-research Pro preview from December 2025.
	ModelDeepResearchProPreview122025 = "deep-research-pro-preview-12-2025"
	// ModelDeepResearchMaxPreview042026 is the deep-research Max preview from April 2026.
	ModelDeepResearchMaxPreview042026 = "deep-research-max-preview-04-2026"
	// ModelDeepResearchPreview042026 is the deep-research preview from April 2026.
	ModelDeepResearchPreview042026 = "deep-research-preview-04-2026"

	// ModelAQA is the Attributed Question Answering model.
	ModelAQA = "aqa"

	// ---- Experimental ----

	// ModelGeminiRoboticsER15Preview is the Gemini Robotics ER 1.5 preview.
	ModelGeminiRoboticsER15Preview = "gemini-robotics-er-1.5-preview"

	// ---- Gemma ----

	// ModelGemma3_1BIt is the Gemma 3 1B instruction-tuned model.
	ModelGemma3_1BIt = "gemma-3-1b-it"
	// ModelGemma3_4BIt is the Gemma 3 4B instruction-tuned model.
	ModelGemma3_4BIt = "gemma-3-4b-it"
	// ModelGemma3nE4BIt is the Gemma 3n E4B instruction-tuned model.
	ModelGemma3nE4BIt = "gemma-3n-e4b-it"
	// ModelGemma3nE2BIt is the Gemma 3n E2B instruction-tuned model.
	ModelGemma3nE2BIt = "gemma-3n-e2b-it"
	// ModelGemma3_12BIt is the Gemma 3 12B instruction-tuned model.
	ModelGemma3_12BIt = "gemma-3-12b-it"
	// ModelGemma3_27BIt is the Gemma 3 27B instruction-tuned model.
	ModelGemma3_27BIt = "gemma-3-27b-it"

	// ---- Embeddings ----

	// ModelGeminiEmbedding001 is the Gemini Embedding 001 model.
	ModelGeminiEmbedding001 = "gemini-embedding-001"
	// ModelGeminiEmbedding2 is the Gemini Embedding 2 model.
	ModelGeminiEmbedding2 = "gemini-embedding-2"
	// ModelGeminiEmbedding2Preview is the Gemini Embedding 2 preview.
	ModelGeminiEmbedding2Preview = "gemini-embedding-2-preview"

	// ---- Video (Veo) ----

	// ModelVeo31FastGeneratePreview is the Veo 3.1 fast generate preview.
	ModelVeo31FastGeneratePreview = "veo-3.1-fast-generate-preview"
	// ModelVeo31GeneratePreview is the Veo 3.1 generate preview.
	ModelVeo31GeneratePreview = "veo-3.1-generate-preview"
	// ModelVeo31Generate is the Veo 3.1 generate model.
	ModelVeo31Generate = "veo-3.1-generate"
	// ModelVeo31LiteGeneratePreview is the Veo 3.1 lite generate preview.
	ModelVeo31LiteGeneratePreview = "veo-3.1-lite-generate-preview"
	// ModelVeo30Generate001 is the Veo 3.0 generate 001 model.
	ModelVeo30Generate001 = "veo-3.0-generate-001"
	// ModelVeo30FastGenerate001 is the Veo 3.0 fast generate 001 model.
	ModelVeo30FastGenerate001 = "veo-3.0-fast-generate-001"
)
