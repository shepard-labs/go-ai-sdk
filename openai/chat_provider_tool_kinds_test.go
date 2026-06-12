package openai

import (
	"testing"
)

// --- coerceArgs ---

func TestCoerceArgsNil(t *testing.T) {
	got, err := coerceArgs(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got == nil {
		t.Errorf("expected non-nil empty map")
	}
}

func TestCoerceArgsMapStringAny(t *testing.T) {
	in := map[string]any{"a": 1.0}
	got, err := coerceArgs(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got["a"] != 1.0 {
		t.Errorf("a: %v", got["a"])
	}
}

func TestCoerceArgsMapStringString(t *testing.T) {
	in := map[string]string{"a": "b"}
	got, err := coerceArgs(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got["a"] != "b" {
		t.Errorf("a: %v", got["a"])
	}
}

func TestCoerceArgsStringJSON(t *testing.T) {
	got, err := coerceArgs(`{"a":1}`)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got["a"] != 1.0 {
		t.Errorf("a: %v", got["a"])
	}
}

func TestCoerceArgsStringInvalidJSON(t *testing.T) {
	_, err := coerceArgs("not-json")
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T", err)
	}
}

func TestCoerceArgsStruct(t *testing.T) {
	type myArgs struct {
		Field string `json:"field"`
	}
	got, err := coerceArgs(myArgs{Field: "value"})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got["field"] != "value" {
		t.Errorf("field: %v", got["field"])
	}
}

func TestCoerceArgsBadStruct(t *testing.T) {
	// channels are not marshalable to JSON
	_, err := coerceArgs(make(chan int))
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T", err)
	}
}

func TestCoerceArgsStructArray(t *testing.T) {
	// A slice of struct should still be coerced: it round-trips through JSON
	// into an array. coerceArgs then... let me check: a slice marshals to
	// an array, but the result is unmarshaled into map[string]any. That fails
	// and returns InvalidPromptError.
	_, err := coerceArgs([]int{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for slice")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T", err)
	}
}

// --- convertProviderTool: each kind ---

func TestConvertProviderToolApplyPatch(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.applyPatch"})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "apply_patch" {
		t.Errorf("type: %v", out["type"])
	}
}

func TestConvertProviderToolCodeInterpreter(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.codeInterpreter", Args: map[string]any{
		"container": map[string]any{"type": "auto"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "code_interpreter" {
		t.Errorf("type: %v", out["type"])
	}
	c, _ := out["container"].(map[string]any)
	if c["type"] != "auto" {
		t.Errorf("container: %v", out["container"])
	}
}

func TestConvertProviderToolFileSearch(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.fileSearch", Args: map[string]any{
		"vectorStoreIDs": []string{"vs_1", "vs_2"},
		"maxNumResults":  float64(5),
		"ranking":        map[string]any{"type": "rrf"},
		"filters":        map[string]any{"k": "v"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	vs, _ := out["vector_store_ids"].([]string)
	if len(vs) != 2 {
		t.Errorf("vector_store_ids: %v", out["vector_store_ids"])
	}
	if out["max_num_results"] != 5 {
		t.Errorf("max_num_results: %v", out["max_num_results"])
	}
}

func TestConvertProviderToolFileSearchAnySlice(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.fileSearch", Args: map[string]any{
		"vectorStoreIDs": []any{"vs_1", "vs_2"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	vs, _ := out["vector_store_ids"].([]string)
	if len(vs) != 2 {
		t.Errorf("vector_store_ids: %v", out["vector_store_ids"])
	}
}

func TestConvertProviderToolFileSearchMaxInt(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.fileSearch", Args: map[string]any{
		"maxNumResults": 7,
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["max_num_results"] != 7 {
		t.Errorf("max_num_results: %v", out["max_num_results"])
	}
}

func TestConvertProviderToolImageGeneration(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.imageGeneration", Args: map[string]any{
		"background":    "opaque",
		"inputImageMask": map[string]any{"image_url": "x"},
		"model":         "gpt-image-1",
		"moderation":    "auto",
		"n":             float64(2),
		"outputFormat":  "png",
		"partialImages": float64(3),
		"quality":       "high",
		"size":          "1024x1024",
		"user":          "u1",
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "image_generation" {
		t.Errorf("type: %v", out["type"])
	}
	for _, k := range []string{"background", "model", "moderation", "output_format", "quality", "size", "user"} {
		if _, has := out[k]; !has {
			t.Errorf("%s missing", k)
		}
	}
	if out["n"] != 2 {
		t.Errorf("n: %v", out["n"])
	}
	if out["partial_images"] != 3 {
		t.Errorf("partial_images: %v", out["partial_images"])
	}
	if _, has := out["input_image_mask"]; !has {
		t.Errorf("input_image_mask missing")
	}
}

func TestConvertProviderToolImageGenerationInt(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.imageGeneration", Args: map[string]any{
		"n":             4,
		"partialImages": 2,
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["n"] != 4 {
		t.Errorf("n: %v", out["n"])
	}
	if out["partial_images"] != 2 {
		t.Errorf("partial_images: %v", out["partial_images"])
	}
}

func TestConvertProviderToolLocalShell(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.localShell"})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "local_shell" {
		t.Errorf("type: %v", out["type"])
	}
}

func TestConvertProviderToolShell(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.shell", Args: map[string]any{
		"environment":   map[string]any{"type": "container"},
		"networkPolicy": map[string]any{"type": "open"},
		"skills":        []any{"x", "y"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "shell" {
		t.Errorf("type: %v", out["type"])
	}
	if _, has := out["environment"]; !has {
		t.Errorf("environment missing")
	}
	if _, has := out["network_policy"]; !has {
		t.Errorf("network_policy missing")
	}
	if _, has := out["skills"]; !has {
		t.Errorf("skills missing")
	}
}

func TestConvertProviderToolWebSearch(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.webSearch", Args: map[string]any{
		"filters":           map[string]any{"k": "v"},
		"searchContextSize": "high",
		"userLocation":      map[string]any{"country": "US"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "web_search" {
		t.Errorf("type: %v", out["type"])
	}
	if _, has := out["filters"]; !has {
		t.Errorf("filters missing")
	}
	if out["search_context_size"] != "high" {
		t.Errorf("search_context_size: %v", out["search_context_size"])
	}
	if _, has := out["user_location"]; !has {
		t.Errorf("user_location missing")
	}
}

func TestConvertProviderToolWebSearchPreview(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.webSearchPreview", Args: map[string]any{
		"searchContextSize": "medium",
		"userLocation":      map[string]any{"city": "NYC"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "web_search_preview" {
		t.Errorf("type: %v", out["type"])
	}
	if out["search_context_size"] != "medium" {
		t.Errorf("search_context_size: %v", out["search_context_size"])
	}
}

func TestConvertProviderToolMCP(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.mcp", Args: map[string]any{
		"serverLabel":     "my-server",
		"serverUrl":       "https://example.com",
		"headers":         map[string]any{"X-Key": "abc"},
		"allowedTools":    []string{"tool1"},
		"requireApproval": map[string]any{"always": []any{"tool1"}},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "mcp" {
		t.Errorf("type: %v", out["type"])
	}
	if out["server_label"] != "my-server" {
		t.Errorf("server_label: %v", out["server_label"])
	}
	if out["server_url"] != "https://example.com" {
		t.Errorf("server_url: %v", out["server_url"])
	}
	if _, has := out["headers"]; !has {
		t.Errorf("headers missing")
	}
	if _, has := out["allowed_tools"]; !has {
		t.Errorf("allowed_tools missing")
	}
	if _, has := out["require_approval"]; !has {
		t.Errorf("require_approval missing")
	}
}

func TestConvertProviderToolMCPAnySlice(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.mcp", Args: map[string]any{
		"allowedTools": []any{"a", "b"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, has := out["allowed_tools"]; !has {
		t.Errorf("allowed_tools missing")
	}
}

func TestConvertProviderToolToolSearch(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.toolSearch", Args: map[string]any{
		"searchContextSize": "low",
		"userLocation":      map[string]any{"country": "CA"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "tool_search" {
		t.Errorf("type: %v", out["type"])
	}
	if out["search_context_size"] != "low" {
		t.Errorf("search_context_size: %v", out["search_context_size"])
	}
}

func TestConvertProviderToolCustom(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.custom", Args: map[string]any{
		"description": "my tool",
		"format":      map[string]any{"type": "grammar"},
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "custom" {
		t.Errorf("type: %v", out["type"])
	}
	if out["description"] != "my tool" {
		t.Errorf("description: %v", out["description"])
	}
	if _, has := out["format"]; !has {
		t.Errorf("format missing")
	}
}

func TestConvertProviderToolArgsCoercionFromString(t *testing.T) {
	m := newTestChatModel()
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.webSearch", Args: `{"searchContextSize":"high"}`})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["search_context_size"] != "high" {
		t.Errorf("search_context_size: %v", out["search_context_size"])
	}
}

func TestConvertProviderToolInvalidJSONArgs(t *testing.T) {
	m := newTestChatModel()
	_, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.webSearch", Args: "not-json"})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T", err)
	}
}

func TestConvertProviderToolArgsCoercionFromStruct(t *testing.T) {
	m := newTestChatModel()
	// Use a typed args struct produced by the openai/tools subpackage.
	max := 3
	out, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai.fileSearch", Args: FileSearchArgs{
		VectorStoreIDs: []string{"vs_a"},
		MaxNumResults:  &max,
	}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out["type"] != "file_search" {
		t.Errorf("type: %v", out["type"])
	}
	vs, _ := out["vector_store_ids"].([]string)
	if len(vs) != 1 || vs[0] != "vs_a" {
		t.Errorf("vector_store_ids: %v", out["vector_store_ids"])
	}
}

func TestConvertProviderToolNoKind(t *testing.T) {
	m := newTestChatModel()
	// "openai." with no kind trips the prefix-length check first.
	_, _, err := m.convertProviderTool(Tool{Type: "provider", ID: "openai."})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T", err)
	}
}
