package google

// model_stubs.go defines the concrete model structs and wires them to the
// model interfaces. The language model struct and its methods live in
// chat.go (Milestone 4). The image model struct and its methods live in
// image.go (Milestone 3). The other model structs are stubs for later
// milestones; they return UnsupportedFunctionalityError until implemented.

import "context"

// ---- Video model stub ----

type googleVideoModel struct {
	provider *googleProvider
	modelID  string
}

func (m *googleVideoModel) ModelID() string       { return m.modelID }
func (m *googleVideoModel) Provider() string      { return m.provider.name + ".video" }
func (m *googleVideoModel) MaxVideosPerCall() int { return defaultMaxVideosPerCall }

func (m *googleVideoModel) DoGenerate(ctx context.Context, opts VideoGenerateOptions) (*VideoGenerateResult, error) {
	return nil, UnsupportedFunctionalityError{Functionality: "DoGenerate (video, not yet implemented — see Milestone 7)"}
}

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

// ---- Tool factories stub ----

// buildToolFactories returns the set of provider-tool factory functions.
// Full implementations are added in Milestone 6; for now each factory is a
// minimal stub that returns the correct Tool shape.
func buildToolFactories() ToolFactories {
	return ToolFactories{
		GoogleSearch: func(args ...GoogleSearchArgs) Tool {
			var a any
			if len(args) > 0 {
				a = args[0]
			}
			return Tool{Type: "provider", ID: "google.google_search", Name: "google_search", ArgsSchema: a}
		},
		EnterpriseWebSearch: func() Tool {
			return Tool{Type: "provider", ID: "google.enterprise_web_search", Name: "enterprise_web_search"}
		},
		GoogleMaps: func() Tool {
			return Tool{Type: "provider", ID: "google.google_maps", Name: "google_maps"}
		},
		UrlContext: func() Tool {
			return Tool{Type: "provider", ID: "google.url_context", Name: "url_context"}
		},
		FileSearch: func(args FileSearchArgs) Tool {
			return Tool{Type: "provider", ID: "google.file_search", Name: "file_search", ArgsSchema: args}
		},
		CodeExecution: func() Tool {
			return Tool{Type: "provider", ID: "google.code_execution", Name: "code_execution"}
		},
		VertexRagStore: func(args VertexRagStoreArgs) Tool {
			return Tool{Type: "provider", ID: "google.vertex_rag_store", Name: "vertex_rag_store", ArgsSchema: args}
		},
	}
}
