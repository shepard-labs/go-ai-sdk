package openaicompatible

import "context"

func (m *openAICompatibleChatLanguageModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	return m.doGenerateChat(ctx, opts)
}
