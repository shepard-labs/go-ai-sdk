package openai

import (
	"testing"
)

// TestApplyPatch verifies the apply_patch provider tool constructor.
func TestApplyPatch(t *testing.T) {
	got := ApplyPatch()
	if got.Type != "provider" {
		t.Errorf("type: %q", got.Type)
	}
	if got.ID != "openai.apply_patch" {
		t.Errorf("id: %q", got.ID)
	}
	if _, ok := got.Args.(ApplyPatchArgs); !ok {
		t.Errorf("args: %T", got.Args)
	}
}

// TestCustomTool verifies the custom provider tool constructor.
func TestCustomTool(t *testing.T) {
	format := &CustomToolFormat{Type: "grammar", Syntax: "..."}
	got := CustomTool("desc", format)
	if got.ID != "openai.custom" {
		t.Errorf("id: %q", got.ID)
	}
	args, ok := got.Args.(CustomToolArgs)
	if !ok {
		t.Fatalf("args type: %T", got.Args)
	}
	if args.Description != "desc" {
		t.Errorf("description: %q", args.Description)
	}
	if args.Format != format {
		t.Errorf("format pointer mismatch")
	}
}

// TestCodeInterpreter verifies the code_interpreter provider tool.
func TestCodeInterpreter(t *testing.T) {
	c := &CodeInterpreterContainer{Type: "auto"}
	got := CodeInterpreter(c)
	if got.ID != "openai.code_interpreter" {
		t.Errorf("id: %q", got.ID)
	}
	args, ok := got.Args.(CodeInterpreterArgs)
	if !ok {
		t.Fatalf("args type: %T", got.Args)
	}
	if args.Container != c {
		t.Errorf("container mismatch")
	}
}

// TestCodeInterpreterNil verifies the code_interpreter provider tool
// accepts a nil container.
func TestCodeInterpreterNil(t *testing.T) {
	got := CodeInterpreter(nil)
	if got.ID != "openai.code_interpreter" {
		t.Errorf("id: %q", got.ID)
	}
}

// TestFileSearch verifies the file_search provider tool.
func TestFileSearch(t *testing.T) {
	got := FileSearch(FileSearchArgs{VectorStoreIDs: []string{"vs_1"}})
	if got.ID != "openai.file_search" {
		t.Errorf("id: %q", got.ID)
	}
	args, ok := got.Args.(FileSearchArgs)
	if !ok {
		t.Fatalf("args type: %T", got.Args)
	}
	if len(args.VectorStoreIDs) != 1 || args.VectorStoreIDs[0] != "vs_1" {
		t.Errorf("vectorStoreIDs: %v", args.VectorStoreIDs)
	}
}

// TestImageGeneration verifies the image_generation provider tool.
func TestImageGeneration(t *testing.T) {
	model := "gpt-image-1"
	got := ImageGeneration(ImageGenerationArgs{Model: &model})
	if got.ID != "openai.image_generation" {
		t.Errorf("id: %q", got.ID)
	}
	args, _ := got.Args.(ImageGenerationArgs)
	if args.Model == nil || *args.Model != "gpt-image-1" {
		t.Errorf("model: %v", args.Model)
	}
}

// TestLocalShell verifies the local_shell provider tool.
func TestLocalShell(t *testing.T) {
	got := LocalShell()
	if got.ID != "openai.local_shell" {
		t.Errorf("id: %q", got.ID)
	}
	if _, ok := got.Args.(LocalShellArgs); !ok {
		t.Errorf("args type: %T", got.Args)
	}
}

// TestShell verifies the shell provider tool.
func TestShell(t *testing.T) {
	got := Shell(ShellArgs{})
	if got.ID != "openai.shell" {
		t.Errorf("id: %q", got.ID)
	}
	if _, ok := got.Args.(ShellArgs); !ok {
		t.Errorf("args type: %T", got.Args)
	}
}

// TestWebSearch verifies the web_search provider tool.
func TestWebSearch(t *testing.T) {
	got := WebSearch(WebSearchArgs{})
	if got.ID != "openai.web_search" {
		t.Errorf("id: %q", got.ID)
	}
	if _, ok := got.Args.(WebSearchArgs); !ok {
		t.Errorf("args type: %T", got.Args)
	}
}

// TestWebSearchPreview verifies the web_search_preview provider tool.
func TestWebSearchPreview(t *testing.T) {
	got := WebSearchPreview(WebSearchPreviewArgs{})
	if got.ID != "openai.web_search_preview" {
		t.Errorf("id: %q", got.ID)
	}
	if _, ok := got.Args.(WebSearchPreviewArgs); !ok {
		t.Errorf("args type: %T", got.Args)
	}
}

// TestMCP verifies the MCP provider tool.
func TestMCP(t *testing.T) {
	got := MCP(MCPArgs{ServerLabel: "label"})
	if got.ID != "openai.mcp" {
		t.Errorf("id: %q", got.ID)
	}
	args, _ := got.Args.(MCPArgs)
	if args.ServerLabel != "label" {
		t.Errorf("serverLabel: %q", args.ServerLabel)
	}
}

// TestToolSearch verifies the tool_search provider tool.
func TestToolSearch(t *testing.T) {
	got := ToolSearch(ToolSearchArgs{})
	if got.ID != "openai.tool_search" {
		t.Errorf("id: %q", got.ID)
	}
	if _, ok := got.Args.(ToolSearchArgs); !ok {
		t.Errorf("args type: %T", got.Args)
	}
}

// TestProviderToolsInterface verifies the openaiTools struct methods
// delegate to the package-level constructor functions.
func TestProviderToolsInterface(t *testing.T) {
	tl := openaiTools{}
	ap := tl.ApplyPatch()
	if ap.ID != "openai.apply_patch" {
		t.Errorf("ApplyPatch id: %q", ap.ID)
	}
	if got := tl.FileSearch(FileSearchArgs{}); got.ID != "openai.file_search" {
		t.Errorf("FileSearch id: %q", got.ID)
	}
	if got := tl.LocalShell(); got.ID != "openai.local_shell" {
		t.Errorf("LocalShell id: %q", got.ID)
	}
	if got := tl.Shell(ShellArgs{}); got.ID != "openai.shell" {
		t.Errorf("Shell id: %q", got.ID)
	}
	if got := tl.WebSearch(WebSearchArgs{}); got.ID != "openai.web_search" {
		t.Errorf("WebSearch id: %q", got.ID)
	}
	if got := tl.WebSearchPreview(WebSearchPreviewArgs{}); got.ID != "openai.web_search_preview" {
		t.Errorf("WebSearchPreview id: %q", got.ID)
	}
	if got := tl.MCP(MCPArgs{}); got.ID != "openai.mcp" {
		t.Errorf("MCP id: %q", got.ID)
	}
	if got := tl.ToolSearch(ToolSearchArgs{}); got.ID != "openai.tool_search" {
		t.Errorf("ToolSearch id: %q", got.ID)
	}
	if got := tl.CodeInterpreter(nil); got.ID != "openai.code_interpreter" {
		t.Errorf("CodeInterpreter id: %q", got.ID)
	}
	if got := tl.ImageGeneration(ImageGenerationArgs{}); got.ID != "openai.image_generation" {
		t.Errorf("ImageGeneration id: %q", got.ID)
	}
	if got := tl.CustomTool("d", nil); got.ID != "openai.custom" {
		t.Errorf("CustomTool id: %q", got.ID)
	}
}
