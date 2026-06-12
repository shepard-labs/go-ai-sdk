package openai

import "context"

// Call is the callable form: openai("gpt-5") returns a ResponsesModel.
func (p *openaiProvider) Call(modelID string) ResponsesModel {
	return p.Responses(modelID)
}

// Model is an alias for Responses.
func (p *openaiProvider) Model(modelID string) ResponsesModel {
	return p.Responses(modelID)
}

// LanguageModel is an alias for Responses.
func (p *openaiProvider) LanguageModel(modelID string) ResponsesModel {
	return p.Responses(modelID)
}

// Chat returns a chat language model wrapping the openaicompatible chat
// model with OpenAI-specific request building.
func (p *openaiProvider) Chat(modelID string) LanguageModel {
	return newChatLanguageModel(p, modelID)
}

// Responses returns a Responses API model.
func (p *openaiProvider) Responses(modelID string) ResponsesModel {
	return newResponsesModel(p, modelID)
}

// Completion returns a legacy completion language model wrapping the
// openaicompatible completion model.
func (p *openaiProvider) Completion(modelID string) LanguageModel {
	return newCompletionLanguageModel(p, modelID)
}

// EmbeddingModel returns an embedding model.
func (p *openaiProvider) EmbeddingModel(modelID string) EmbeddingModel {
	return newEmbeddingModel(p, modelID)
}

// Embedding is an alias for EmbeddingModel.
func (p *openaiProvider) Embedding(modelID string) EmbeddingModel {
	return p.EmbeddingModel(modelID)
}

// TextEmbeddingModel is a deprecated alias for EmbeddingModel.
func (p *openaiProvider) TextEmbeddingModel(modelID string) EmbeddingModel {
	return p.EmbeddingModel(modelID)
}

// ImageModel returns an image generation model.
func (p *openaiProvider) ImageModel(modelID string) ImageModel {
	return newImageModel(p, modelID)
}

// Image is an alias for ImageModel.
func (p *openaiProvider) Image(modelID string) ImageModel {
	return p.ImageModel(modelID)
}

// Speech returns a speech synthesis model.
func (p *openaiProvider) Speech(modelID string) SpeechModel {
	return newSpeechModel(p, modelID)
}

// SpeechModel is an alias for Speech.
func (p *openaiProvider) SpeechModel(modelID string) SpeechModel {
	return p.Speech(modelID)
}

// Transcription returns a transcription model.
func (p *openaiProvider) Transcription(modelID string) TranscriptionModel {
	return newTranscriptionModel(p, modelID)
}

// TranscriptionModel is an alias for Transcription.
func (p *openaiProvider) TranscriptionModel(modelID string) TranscriptionModel {
	return p.Transcription(modelID)
}

// Files returns the Files API surface.
func (p *openaiProvider) Files() Files { return p.files }

// Skills returns the Skills API surface.
func (p *openaiProvider) Skills() Skills { return p.skills }

// ExperimentalRealtime returns the Realtime factory.
func (p *openaiProvider) ExperimentalRealtime() ExperimentalRealtimeFactory {
	return p.realtime
}

// Tools returns the tool factory bag.
func (p *openaiProvider) Tools() Tools { return &p.tools }

// ApplyPatch implements Tools.
func (t *openaiTools) ApplyPatch() Tool { return ApplyPatch() }

// CustomTool implements Tools.
func (t *openaiTools) CustomTool(description string, format *CustomToolFormat) Tool {
	return CustomTool(description, format)
}

// CodeInterpreter implements Tools.
func (t *openaiTools) CodeInterpreter(container *CodeInterpreterContainer) Tool {
	return CodeInterpreter(container)
}

// FileSearch implements Tools.
func (t *openaiTools) FileSearch(args FileSearchArgs) Tool { return FileSearch(args) }

// ImageGeneration implements Tools.
func (t *openaiTools) ImageGeneration(args ImageGenerationArgs) Tool {
	return ImageGeneration(args)
}

// LocalShell implements Tools.
func (t *openaiTools) LocalShell() Tool { return LocalShell() }

// Shell implements Tools.
func (t *openaiTools) Shell(args ShellArgs) Tool { return Shell(args) }

// WebSearch implements Tools.
func (t *openaiTools) WebSearch(args WebSearchArgs) Tool { return WebSearch(args) }

// WebSearchPreview implements Tools.
func (t *openaiTools) WebSearchPreview(args WebSearchPreviewArgs) Tool {
	return WebSearchPreview(args)
}

// MCP implements Tools.
func (t *openaiTools) MCP(args MCPArgs) Tool { return MCP(args) }

// ToolSearch implements Tools.
func (t *openaiTools) ToolSearch(args ToolSearchArgs) Tool { return ToolSearch(args) }

// openaiTools is defined in tools.go; this is a placeholder reference.

// Ensure newFilesClient / newSkillsClient / newRealtimeFactory are referenced
// in at least one declaration so the package compiles even before their
// respective files are added.
var _ = context.Background
