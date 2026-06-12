package openai

// Constructor stubs for the additional model types referenced by
// provider_methods.go. Each is filled out in a later iteration. They are
// defined here so that the package compiles end-to-end.

func newFilesClient(p *openaiProvider) Files {
	return &openaiFilesClient{provider: p}
}

func newSkillsClient(p *openaiProvider) Skills {
	return &openaiSkillsClient{provider: p}
}

func newRealtimeFactory(p *openaiProvider) ExperimentalRealtimeFactory {
	return &openaiRealtimeFactory{provider: p}
}
