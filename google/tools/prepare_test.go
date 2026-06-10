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


// ---- Tool factory output tests ----

func TestTools_GoogleSearch_Default(t *testing.T) {
	got := tools.GoogleSearch()
	if got.ID != "google.google_search" {
		t.Errorf("ID = %q, want google.google_search", got.ID)
	}
	if got.Name != "google_search" {
		t.Errorf("Name = %q, want google_search", got.Name)
	}
	if got.Type != "provider" {
		t.Errorf("Type = %q, want provider", got.Type)
	}
	if got.ArgsSchema != nil {
		t.Errorf("ArgsSchema = %v, want nil", got.ArgsSchema)
	}
}

func TestTools_GoogleSearch_WithArgs(t *testing.T) {
	ts := &GoogleSearchTypes{WebSearch: map[string]any{"mode": "MODE_UNSPECIFIED"}}
	tr := &TimeRangeFilter{StartTime: "2024-01-01T00:00:00Z"}
	got := tools.GoogleSearch(GoogleSearchArgs{SearchTypes: ts, TimeRangeFilter: tr})
	if got.ID != "google.google_search" {
		t.Errorf("ID = %q", got.ID)
	}
	args, ok := got.ArgsSchema.(GoogleSearchArgs)
	if !ok {
		t.Fatalf("ArgsSchema type = %T, want GoogleSearchArgs", got.ArgsSchema)
	}
	if args.SearchTypes == nil || args.SearchTypes.WebSearch == nil {
		t.Errorf("SearchTypes missing")
	}
	if args.TimeRangeFilter == nil {
		t.Errorf("TimeRangeFilter missing")
	}
}

func TestTools_EnterpriseWebSearch(t *testing.T) {
	got := tools.EnterpriseWebSearch()
	if got.ID != "google.enterprise_web_search" {
		t.Errorf("ID = %q, want google.enterprise_web_search", got.ID)
	}
	if got.Name != "enterprise_web_search" {
		t.Errorf("Name = %q, want enterprise_web_search", got.Name)
	}
	if got.Type != "provider" {
		t.Errorf("Type = %q, want provider", got.Type)
	}
	if got.ArgsSchema != nil {
		t.Errorf("ArgsSchema = %v, want nil", got.ArgsSchema)
	}
}

func TestTools_GoogleMaps(t *testing.T) {
	got := tools.GoogleMaps()
	if got.ID != "google.google_maps" {
		t.Errorf("ID = %q, want google.google_maps", got.ID)
	}
	if got.Name != "google_maps" {
		t.Errorf("Name = %q, want google_maps", got.Name)
	}
	if got.Type != "provider" {
		t.Errorf("Type = %q, want provider", got.Type)
	}
}

func TestTools_UrlContext(t *testing.T) {
	got := tools.UrlContext()
	if got.ID != "google.url_context" {
		t.Errorf("ID = %q, want google.url_context", got.ID)
	}
	if got.Name != "url_context" {
		t.Errorf("Name = %q, want url_context", got.Name)
	}
	if got.Type != "provider" {
		t.Errorf("Type = %q, want provider", got.Type)
	}
}

func TestTools_FileSearch_Default(t *testing.T) {
	got := tools.FileSearch(FileSearchArgs{})
	if got.ID != "google.file_search" {
		t.Errorf("ID = %q, want google.file_search", got.ID)
	}
	if got.Name != "file_search" {
		t.Errorf("Name = %q, want file_search", got.Name)
	}
	if got.Type != "provider" {
		t.Errorf("Type = %q, want provider", got.Type)
	}
}

func TestTools_FileSearch_WithArgs(t *testing.T) {
	k := 10
	got := tools.FileSearch(FileSearchArgs{
		FileSearchStoreNames: []string{"store-a", "store-b"},
		TopK:                &k,
		MetadataFilter:      "key=value",
	})
	args, ok := got.ArgsSchema.(FileSearchArgs)
	if !ok {
		t.Fatalf("ArgsSchema type = %T, want FileSearchArgs", got.ArgsSchema)
	}
	if len(args.FileSearchStoreNames) != 2 {
		t.Errorf("FileSearchStoreNames = %v, want 2 entries", args.FileSearchStoreNames)
	}
	if args.TopK == nil || *args.TopK != 10 {
		t.Errorf("TopK = %v", args.TopK)
	}
	if args.MetadataFilter != "key=value" {
		t.Errorf("MetadataFilter = %q", args.MetadataFilter)
	}
}

func TestTools_CodeExecution(t *testing.T) {
	got := tools.CodeExecution()
	if got.ID != "google.code_execution" {
		t.Errorf("ID = %q, want google.code_execution", got.ID)
	}
	if got.Name != "code_execution" {
		t.Errorf("Name = %q, want code_execution", got.Name)
	}
	if got.Type != "provider" {
		t.Errorf("Type = %q, want provider", got.Type)
	}
}

func TestTools_VertexRagStore_Default(t *testing.T) {
	got := tools.VertexRagStore(VertexRagStoreArgs{})
	if got.ID != "google.vertex_rag_store" {
		t.Errorf("ID = %q, want google.vertex_rag_store", got.ID)
	}
	if got.Name != "vertex_rag_store" {
		t.Errorf("Name = %q, want vertex_rag_store", got.Name)
	}
	if got.Type != "provider" {
		t.Errorf("Type = %q, want provider", got.Type)
	}
}

func TestTools_VertexRagStore_WithArgs(t *testing.T) {
	k := 5
	got := tools.VertexRagStore(VertexRagStoreArgs{
		RagCorpus: "projects/p/locations/l/ragCorpora/c",
		TopK:      &k,
	})
	args, ok := got.ArgsSchema.(VertexRagStoreArgs)
	if !ok {
		t.Fatalf("ArgsSchema type = %T, want VertexRagStoreArgs", got.ArgsSchema)
	}
	if args.RagCorpus != "projects/p/locations/l/ragCorpora/c" {
		t.Errorf("RagCorpus = %q", args.RagCorpus)
	}
	if args.TopK == nil || *args.TopK != 5 {
		t.Errorf("TopK = %v", args.TopK)
	}
}

func TestTools_Build(t *testing.T) {
	tf := tools.Build()
	// Verify each factory is non-nil and returns the correct tool.
	if got := tf.GoogleSearch(); got.ID != "google.google_search" {
		t.Errorf("GoogleSearch: ID = %q", got.ID)
	}
	if got := tf.EnterpriseWebSearch(); got.ID != "google.enterprise_web_search" {
		t.Errorf("EnterpriseWebSearch: ID = %q", got.ID)
	}
	if got := tf.GoogleMaps(); got.ID != "google.google_maps" {
		t.Errorf("GoogleMaps: ID = %q", got.ID)
	}
	if got := tf.UrlContext(); got.ID != "google.url_context" {
		t.Errorf("UrlContext: ID = %q", got.ID)
	}
	if got := tf.FileSearch(FileSearchArgs{}); got.ID != "google.file_search" {
		t.Errorf("FileSearch: ID = %q", got.ID)
	}
	if got := tf.CodeExecution(); got.ID != "google.code_execution" {
		t.Errorf("CodeExecution: ID = %q", got.ID)
	}
	if got := tf.VertexRagStore(VertexRagStoreArgs{}); got.ID != "google.vertex_rag_store" {
		t.Errorf("VertexRagStore: ID = %q", got.ID)
	}
}
