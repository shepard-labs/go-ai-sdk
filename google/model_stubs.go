package google

// model_stubs.go defines the concrete model structs and wires them to the
// model interfaces. The language model struct and its methods live in
// chat.go (Milestone 4). The image model struct and its methods live in
// image.go (Milestone 3). The other model structs are stubs for later
// milestones; they return UnsupportedFunctionalityError until implemented.

import (
	"context"

	"github.com/shepard-labs/go-ai-sdk/google/tools"
)

// ---- Speech model stub ----

type googleSpeechModel struct {
	provider *googleProvider
	modelID  string
}

func (m *googleSpeechModel) ModelID() string  { return m.modelID }
func (m *googleSpeechModel) Provider() string { return m.provider.name + ".speech" }

func (m *googleSpeechModel) DoGenerate(ctx context.Context, opts SpeechGenerateOptions) (*SpeechGenerateResult, error) {
	return nil, UnsupportedFunctionalityError{Functionality: "DoGenerate (speech, not yet implemented — see Milestone 8)"}
}

// ---- Files stub ----

type googleFiles struct {
	provider *googleProvider
}

func (f *googleFiles) Upload(ctx context.Context, data []byte, opts FilesUploadOptions) (*FilesUploadResult, error) {
	return nil, UnsupportedFunctionalityError{Functionality: "Upload (not yet implemented — see Milestone 9)"}
}

// buildToolFactories returns the Google provider-tool factories.
// tools.Build() returns tools.ToolFactories (local types, avoids import cycle).
// This function adapts it to google.ToolFactories.
func buildToolFactories() ToolFactories {
	tf := tools.Tools{}
	return ToolFactories{
		GoogleSearch: func(args ...GoogleSearchArgs) Tool {
			return convertTool(tf.GoogleSearch(args...))
		},
		EnterpriseWebSearch: func() Tool {
			return convertTool(tf.EnterpriseWebSearch())
		},
		GoogleMaps: func() Tool {
			return convertTool(tf.GoogleMaps())
		},
		UrlContext: func() Tool {
			return convertTool(tf.UrlContext())
		},
		FileSearch: func(args FileSearchArgs) Tool {
			return convertTool(tf.FileSearch(args))
		},
		CodeExecution: func() Tool {
			return convertTool(tf.CodeExecution())
		},
		VertexRagStore: func(args VertexRagStoreArgs) Tool {
			return convertTool(tf.VertexRagStore(args))
		},
	}
}

// convertTool converts a tools.Tool to a google.Tool.
func convertTool(t tools.Tool) Tool {
	return Tool{
		ID:               t.ID,
		Name:             t.Name,
		Type:             t.Type,
		ArgsSchema:       t.ArgsSchema,
		InputSchema:      t.InputSchema,
		Strict:           t.Strict,
		ProviderExecuted: t.ProviderExecuted,
		Dynamic:          t.Dynamic,
	}
}
