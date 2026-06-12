package openai

// openaiTools implements Tools.
type openaiTools struct{}

// ApplyPatch returns the apply_patch provider tool.
func ApplyPatch() Tool {
	return Tool{Type: "provider", ID: "openai.apply_patch", Args: ApplyPatchArgs{}}
}

// CustomTool returns a custom provider tool.
func CustomTool(description string, format *CustomToolFormat) Tool {
	return Tool{Type: "provider", ID: "openai.custom", Args: CustomToolArgs{Description: description, Format: format}}
}

// CodeInterpreter returns the code_interpreter provider tool.
func CodeInterpreter(container *CodeInterpreterContainer) Tool {
	return Tool{Type: "provider", ID: "openai.code_interpreter", Args: CodeInterpreterArgs{Container: container}}
}

// FileSearch returns the file_search provider tool.
func FileSearch(args FileSearchArgs) Tool {
	return Tool{Type: "provider", ID: "openai.file_search", Args: args}
}

// ImageGeneration returns the image_generation provider tool.
func ImageGeneration(args ImageGenerationArgs) Tool {
	return Tool{Type: "provider", ID: "openai.image_generation", Args: args}
}

// LocalShell returns the local_shell provider tool.
func LocalShell() Tool {
	return Tool{Type: "provider", ID: "openai.local_shell", Args: LocalShellArgs{}}
}

// Shell returns the shell provider tool.
func Shell(args ShellArgs) Tool { return Tool{Type: "provider", ID: "openai.shell", Args: args} }

// WebSearch returns the web_search provider tool.
func WebSearch(args WebSearchArgs) Tool {
	return Tool{Type: "provider", ID: "openai.web_search", Args: args}
}

// WebSearchPreview returns the web_search_preview provider tool.
func WebSearchPreview(args WebSearchPreviewArgs) Tool {
	return Tool{Type: "provider", ID: "openai.web_search_preview", Args: args}
}

// MCP returns the MCP provider tool.
func MCP(args MCPArgs) Tool { return Tool{Type: "provider", ID: "openai.mcp", Args: args} }

// ToolSearch returns the tool_search provider tool.
func ToolSearch(args ToolSearchArgs) Tool {
	return Tool{Type: "provider", ID: "openai.tool_search", Args: args}
}
