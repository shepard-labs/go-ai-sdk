package tools

import (
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

func strPtr(s string) *string { return &s }

func TestAllFactoriesReturnValidTools(t *testing.T) {
	tests := []struct {
		name string
		tool openai.Tool
		want string
	}{
		{"apply_patch", ApplyPatch(), "openai.apply_patch"},
		{"code_interpreter", CodeInterpreter(&openai.CodeInterpreterContainer{Type: "auto"}), "openai.code_interpreter"},
		{"file_search", FileSearch(openai.FileSearchArgs{VectorStoreIDs: []string{"vs_1"}}), "openai.file_search"},
		{"image_generation", ImageGeneration(openai.ImageGenerationArgs{}), "openai.image_generation"},
		{"local_shell", LocalShell(), "openai.local_shell"},
		{"shell", Shell(openai.ShellArgs{}), "openai.shell"},
		{"web_search", WebSearch(openai.WebSearchArgs{}), "openai.web_search"},
		{"web_search_preview", WebSearchPreview(openai.WebSearchPreviewArgs{}), "openai.web_search_preview"},
		{"mcp", MCP(openai.MCPArgs{ServerLabel: "x", ServerURL: strPtr("https://example.com")}), "openai.mcp"},
		{"tool_search", ToolSearch(openai.ToolSearchArgs{}), "openai.tool_search"},
		{"custom", CustomTool("hello", nil), "openai.custom"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.tool.Type != "provider" {
				t.Errorf("tool type = %q, want %q", tt.tool.Type, "provider")
			}
			if tt.tool.ID != tt.want {
				t.Errorf("tool ID = %q, want %q", tt.tool.ID, tt.want)
			}
			if tt.tool.Args == nil {
				t.Errorf("tool args is nil")
			}
		})
	}
}
