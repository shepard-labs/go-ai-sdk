package tools

import (
	"encoding/json"
	"testing"
)

func view(typ, id, name string) ToolView {
	return ToolView{Type: typ, ID: id, Name: name}
}

func TestPrepareTools_Empty(t *testing.T) {
	out, cfg, w, err := PrepareTools(nil, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	if out != nil || cfg != nil || w != nil {
		t.Errorf("empty input: got %v %v %v, want nil", out, cfg, w)
	}
}

func TestPrepareTools_FunctionTool_BasicShape(t *testing.T) {
	in := []ToolView{view("function", "", "get_weather")}
	in[0].Description = "Get the weather"
	in[0].InputSchema = map[string]any{
		"type": "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
	}
	out, _, _, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("out = %d, want 1", len(out))
	}
	body := out[0].Body
	if body == nil {
		t.Fatal("body is nil")
	}
	decls, ok := body["functionDeclarations"].([]map[string]any)
	if !ok || len(decls) != 1 {
		t.Fatalf("functionDeclarations: got %+v", body)
	}
	d := decls[0]
	if d["name"] != "get_weather" {
		t.Errorf("name = %v", d["name"])
	}
	if d["description"] != "Get the weather" {
		t.Errorf("description = %v", d["description"])
	}
	// Parameters should be an OpenAPI schema; the schema is an object here.
	params, ok := d["parameters"].(map[string]any)
	if !ok {
		t.Fatalf("parameters: got %T", d["parameters"])
	}
	if params["type"] != "object" {
		t.Errorf("parameters.type = %v, want object", params["type"])
	}
}

func TestPrepareTools_GoogleSearch_Empty(t *testing.T) {
	in := []ToolView{view("provider", "google.google_search", "google_search")}
	out, _, _, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Body["googleSearch"] == nil {
		t.Errorf("googleSearch key missing: %+v", out[0].Body)
	}
}

func TestPrepareTools_GoogleSearch_WithArgs(t *testing.T) {
	in := []ToolView{{
		Type: "provider", ID: "google.google_search", Name: "google_search",
		ArgsSchema: GoogleSearchArgs{
			SearchTypes: &GoogleSearchTypes{
				WebSearch:   map[string]any{"mode": "MODE_UNSPECIFIED"},
				ImageSearch: map[string]any{"mode": "MODE_UNSPECIFIED"},
			},
			TimeRangeFilter: &TimeRangeFilter{StartTime: "2024-01-01T00:00:00Z", EndTime: "2024-12-31T23:59:59Z"},
		},
	}}
	out, _, _, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	gs, ok := out[0].Body["googleSearch"].(map[string]any)
	if !ok {
		t.Fatalf("googleSearch: %+v", out[0].Body)
	}
	if _, ok := gs["searchTypes"]; !ok {
		t.Errorf("searchTypes missing")
	}
	if _, ok := gs["timeRangeFilter"]; !ok {
		t.Errorf("timeRangeFilter missing")
	}
}

func TestPrepareTools_MixedTools_Gemini3_HasIncludeServerSide(t *testing.T) {
	in := []ToolView{
		view("provider", "google.google_search", "google_search"),
		view("function", "", "get_weather"),
	}
	out, cfg, w, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-3-pro-preview"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Errorf("out = %d, want 2", len(out))
	}
	if cfg == nil || cfg.IncludeServerSideToolInvocations == nil || !*cfg.IncludeServerSideToolInvocations {
		t.Errorf("expected includeServerSideToolInvocations=true on Gemini 3 mixed, got %+v", cfg)
	}
	if w != nil {
		t.Errorf("expected no warnings on Gemini 3, got %+v", w)
	}
}

func TestPrepareTools_MixedTools_PreGemini3_Warning(t *testing.T) {
	in := []ToolView{
		view("provider", "google.google_search", "google_search"),
		view("function", "", "get_weather"),
	}
	out, cfg, w, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	// Pre-Gemini 3: function tools are dropped, only provider tool is emitted.
	if len(out) != 1 {
		t.Errorf("out = %d, want 1 (function tools dropped)", len(out))
	}
	if cfg != nil {
		t.Errorf("cfg: got %+v, want nil", cfg)
	}
	if len(w) == 0 || w[0].Type != "unsupported" {
		t.Errorf("expected unsupported warning, got %+v", w)
	}
}

func TestPrepareTools_ToolChoice_Mapping(t *testing.T) {
	cases := []struct {
		name    string
		choice  *ToolChoiceView
		anyStrict bool
		wantMode string
	}{
		{"nil choice, no strict", nil, false, ""},
		{"nil choice, strict", nil, true, "VALIDATED"},
		{"auto", &ToolChoiceView{Type: "auto"}, false, "AUTO"},
		{"auto+strict", &ToolChoiceView{Type: "auto"}, true, "VALIDATED"},
		{"none", &ToolChoiceView{Type: "none"}, false, "NONE"},
		{"required", &ToolChoiceView{Type: "required"}, false, "ANY"},
		{"tool", &ToolChoiceView{Type: "tool", ToolName: "f"}, false, "ANY"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mode, allowed := toolChoiceToMode(tc.choice, tc.anyStrict, false)
			if mode != tc.wantMode {
				t.Errorf("mode = %q, want %q", mode, tc.wantMode)
			}
			if tc.choice != nil && tc.choice.Type == "tool" && len(allowed) != 1 {
				t.Errorf("allowed = %v, want 1 entry", allowed)
			}
		})
	}
}

func TestPrepareTools_VertexRagStore_NonVertex_Warning(t *testing.T) {
	in := []ToolView{view("provider", "google.vertex_rag_store", "vertex_rag_store")}
	in[0].ArgsSchema = VertexRagStoreArgs{RagCorpus: "projects/p/locations/l/ragCorpora/c", TopK: intPtr(5)}
	_, _, w, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro", IsVertexProvider: false})
	if err != nil {
		t.Fatal(err)
	}
	if len(w) == 0 || w[0].Feature != "vertexRagStore" {
		t.Errorf("expected vertexRagStore warning, got %+v", w)
	}
}

func TestPrepareTools_VertexRagStore_Vertex_OK(t *testing.T) {
	in := []ToolView{view("provider", "google.vertex_rag_store", "vertex_rag_store")}
	in[0].ArgsSchema = VertexRagStoreArgs{RagCorpus: "projects/p/locations/l/ragCorpora/c", TopK: intPtr(5)}
	out, _, w, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro", IsVertexProvider: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 0 {
		t.Errorf("expected no warnings, got %+v", w)
	}
	retrieval, ok := out[0].Body["retrieval"].(map[string]any)
	if !ok {
		t.Fatalf("retrieval: %+v", out[0].Body)
	}
	vrs, ok := retrieval["vertex_rag_store"].(map[string]any)
	if !ok {
		t.Fatalf("vertex_rag_store: %+v", retrieval)
	}
	resources, ok := vrs["rag_resources"].(map[string]any)
	if !ok || resources["rag_corpus"] != "projects/p/locations/l/ragCorpora/c" {
		t.Errorf("rag_corpus: %+v", resources)
	}
	if vrs["similarity_top_k"].(int) != 5 {
		t.Errorf("similarity_top_k: %v", vrs["similarity_top_k"])
	}
}

func TestPrepareTools_StreamFunctionCallArguments_VertexStreaming(t *testing.T) {
	t1 := view("function", "", "f1")
	tt := true
	in := []ToolView{t1}
	_, cfg, _, err := PrepareTools(in, PrepareToolsOpts{
		ModelID: "gemini-2.5-pro", IsVertexProvider: true, IsStreaming: true,
		StreamFunctionCallArguments: &tt,
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg == nil || cfg.FunctionCallingConfig == nil || cfg.FunctionCallingConfig.StreamFunctionCallArguments == nil || !*cfg.FunctionCallingConfig.StreamFunctionCallArguments {
		t.Errorf("expected streamFunctionCallArguments=true, got %+v", cfg)
	}
}

func TestPrepareTools_StreamFunctionCallArguments_NonVertex_Warning(t *testing.T) {
	tt := true
	in := []ToolView{view("function", "", "f1")}
	_, _, w, err := PrepareTools(in, PrepareToolsOpts{
		ModelID: "gemini-2.5-pro", IsVertexProvider: false,
		StreamFunctionCallArguments: &tt,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(w) == 0 || w[0].Feature != "streamFunctionCallArguments" {
		t.Errorf("expected streamFunctionCallArguments warning, got %+v", w)
	}
}

func TestPrepareTools_FileSearch_WithArgs(t *testing.T) {
	in := []ToolView{view("provider", "google.file_search", "file_search")}
	in[0].ArgsSchema = FileSearchArgs{
		FileSearchStoreNames: []string{"storeA", "storeB"},
		TopK:                 intPtr(10),
		MetadataFilter:       "key=value",
	}
	out, _, _, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	fs, ok := out[0].Body["fileSearch"].(map[string]any)
	if !ok {
		t.Fatalf("fileSearch: %+v", out[0].Body)
	}
	stores, ok := fs["fileSearchStoreNames"].([]string)
	if !ok || len(stores) != 2 {
		t.Errorf("fileSearchStoreNames: %+v", fs["fileSearchStoreNames"])
	}
	if fs["topK"].(int) != 10 {
		t.Errorf("topK: %v", fs["topK"])
	}
	if fs["metadataFilter"] != "key=value" {
		t.Errorf("metadataFilter: %v", fs["metadataFilter"])
	}
}

func TestPrepareTools_UnknownProviderTool_Passthrough(t *testing.T) {
	in := []ToolView{{
		Type: "provider", ID: "google.custom_thing", Name: "custom_thing",
		ArgsSchema: map[string]any{"foo": "bar"},
	}}
	out, _, _, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Body["custom_thing"] == nil {
		t.Errorf("expected passthrough custom_thing, got %+v", out[0].Body)
	}
}

func TestPrepareTools_JSONMarshalStable(t *testing.T) {
	in := []ToolView{view("provider", "google.url_context", "url_context")}
	out, _, _, err := PrepareTools(in, PrepareToolsOpts{ModelID: "gemini-2.5-pro"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(out[0].Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{"urlContext":{}}` {
		t.Errorf("got %s", b)
	}
}

func intPtr(n int) *int { return &n }
