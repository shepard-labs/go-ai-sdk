// Package tools is the public re-export of the OpenAI provider tool
// factories. Each function returns an [openai.Tool] value with
// `Type: "provider"` and `ID: "openai.<kind>"`. The chat and responses
// models recognize these IDs and emit the correct wire-format tool entry.
//
// Usage:
//
//	import "github.com/shepard-labs/go-ai-sdk/openai/tools"
//	ts := []openai.Tool{
//	    tools.WebSearch(openai.WebSearchArgs{SearchContextSize: ptr.String("medium")}),
//	    tools.FileSearch(openai.FileSearchArgs{VectorStoreIDs: []string{"vs_abc"}}),
//	}
package tools

import "github.com/shepard-labs/go-ai-sdk/openai"

// ApplyPatch returns the apply_patch provider tool.
func ApplyPatch() openai.Tool { return openai.ApplyPatch() }

// CustomTool returns a custom provider tool.
func CustomTool(description string, format *openai.CustomToolFormat) openai.Tool {
	return openai.CustomTool(description, format)
}

// CodeInterpreter returns the code_interpreter provider tool.
func CodeInterpreter(container *openai.CodeInterpreterContainer) openai.Tool {
	return openai.CodeInterpreter(container)
}

// FileSearch returns the file_search provider tool.
func FileSearch(args openai.FileSearchArgs) openai.Tool { return openai.FileSearch(args) }

// ImageGeneration returns the image_generation provider tool.
func ImageGeneration(args openai.ImageGenerationArgs) openai.Tool {
	return openai.ImageGeneration(args)
}

// LocalShell returns the local_shell provider tool.
func LocalShell() openai.Tool { return openai.LocalShell() }

// Shell returns the shell provider tool.
func Shell(args openai.ShellArgs) openai.Tool { return openai.Shell(args) }

// WebSearch returns the web_search provider tool.
func WebSearch(args openai.WebSearchArgs) openai.Tool { return openai.WebSearch(args) }

// WebSearchPreview returns the web_search_preview provider tool.
func WebSearchPreview(args openai.WebSearchPreviewArgs) openai.Tool {
	return openai.WebSearchPreview(args)
}

// MCP returns the MCP provider tool.
func MCP(args openai.MCPArgs) openai.Tool { return openai.MCP(args) }

// ToolSearch returns the tool_search provider tool.
func ToolSearch(args openai.ToolSearchArgs) openai.Tool { return openai.ToolSearch(args) }
