package tools

import anthropic "github.com/shepard-labs/go-ai-sdk/anthropic"

func PrepareTools(input []anthropic.Tool, opts anthropic.ToolOptions) []anthropic.Tool {
	prepared := make([]anthropic.Tool, len(input))
	for i, tool := range input {
		if opts.DeferLoading != nil {
			tool.DeferLoading = opts.DeferLoading
		}
		if opts.AllowedCallers != nil {
			tool.AllowedCallers = append([]anthropic.ToolCallCaller(nil), opts.AllowedCallers...)
		}
		if opts.EagerInputStreaming != nil {
			tool.EagerInputStreaming = opts.EagerInputStreaming
		}
		prepared[i] = tool
	}
	return prepared
}

var ToolNameMapping = anthropic.ToolNameMapping{
	"anthropic.code_execution_20250522":    "code_execution",
	"anthropic.code_execution_20250825":    "code_execution",
	"anthropic.code_execution_20260120":    "code_execution",
	"anthropic.computer_20241022":          "computer",
	"anthropic.computer_20250124":          "computer",
	"anthropic.computer_20251124":          "computer",
	"anthropic.text_editor_20241022":       "str_replace_editor",
	"anthropic.text_editor_20250124":       "str_replace_editor",
	"anthropic.text_editor_20250429":       "str_replace_based_edit_tool",
	"anthropic.text_editor_20250728":       "str_replace_based_edit_tool",
	"anthropic.bash_20241022":              "bash",
	"anthropic.bash_20250124":              "bash",
	"anthropic.memory_20250818":            "memory",
	"anthropic.web_search_20250305":        "web_search",
	"anthropic.web_search_20260209":        "web_search",
	"anthropic.web_fetch_20250910":         "web_fetch",
	"anthropic.web_fetch_20260209":         "web_fetch",
	"anthropic.tool_search_regex_20251119": "tool_search_tool_regex",
	"anthropic.tool_search_bm25_20251119":  "tool_search_tool_bm25",
	"anthropic.advisor_20260301":           "advisor",
}

func Bash_20241022() anthropic.Tool { return providerTool("anthropic.bash_20241022", "bash_20241022") }
func Bash_20250124() anthropic.Tool { return providerTool("anthropic.bash_20250124", "bash_20250124") }

func CodeExecution_20250522() anthropic.Tool {
	return providerTool("anthropic.code_execution_20250522", "code_execution_20250522")
}
func CodeExecution_20250825() anthropic.Tool {
	return providerTool("anthropic.code_execution_20250825", "code_execution_20250825")
}
func CodeExecution_20260120() anthropic.Tool {
	return providerTool("anthropic.code_execution_20260120", "code_execution_20260120")
}

func Computer_20241022(displayWidthPx, displayHeightPx, displayNumber int) anthropic.Tool {
	tool := providerTool("anthropic.computer_20241022", "computer_20241022")
	tool.DisplayWidthPx = &displayWidthPx
	tool.DisplayHeightPx = &displayHeightPx
	tool.DisplayNumber = &displayNumber
	return tool
}

func Computer_20250124(displayWidthPx, displayHeightPx, displayNumber int) anthropic.Tool {
	tool := providerTool("anthropic.computer_20250124", "computer_20250124")
	tool.DisplayWidthPx = &displayWidthPx
	tool.DisplayHeightPx = &displayHeightPx
	tool.DisplayNumber = &displayNumber
	return tool
}

func Computer_20251124(displayWidthPx, displayHeightPx, displayNumber int, enableZoom bool) anthropic.Tool {
	tool := providerTool("anthropic.computer_20251124", "computer_20251124")
	tool.DisplayWidthPx = &displayWidthPx
	tool.DisplayHeightPx = &displayHeightPx
	tool.DisplayNumber = &displayNumber
	tool.EnableZoom = &enableZoom
	return tool
}

func TextEditor_20241022() anthropic.Tool {
	return providerTool("anthropic.text_editor_20241022", "text_editor_20241022")
}
func TextEditor_20250124() anthropic.Tool {
	return providerTool("anthropic.text_editor_20250124", "text_editor_20250124")
}
func TextEditor_20250429() anthropic.Tool {
	return providerTool("anthropic.text_editor_20250429", "text_editor_20250429")
}

func TextEditor_20250728(maxCharacters *int) anthropic.Tool {
	tool := providerTool("anthropic.text_editor_20250728", "text_editor_20250728")
	tool.MaxCharacters = maxCharacters
	return tool
}

func Memory_20250818() anthropic.Tool {
	return providerTool("anthropic.memory_20250818", "memory_20250818")
}

func WebFetch_20250910(maxUses *int, allowedDomains, blockedDomains []string, citations *anthropic.CitationsConfig, maxContentTokens *int) anthropic.Tool {
	tool := providerTool("anthropic.web_fetch_20250910", "web_fetch_20250910")
	tool.MaxUses = maxUses
	tool.AllowedDomains = allowedDomains
	tool.BlockedDomains = blockedDomains
	tool.Citations = citations
	tool.MaxContentTokens = maxContentTokens
	return tool
}

func WebFetch_20260209(maxUses *int, allowedDomains, blockedDomains []string, citations *anthropic.CitationsConfig, maxContentTokens *int) anthropic.Tool {
	tool := WebFetch_20250910(maxUses, allowedDomains, blockedDomains, citations, maxContentTokens)
	tool.ID = "anthropic.web_fetch_20260209"
	tool.Name = "web_fetch_20260209"
	return tool
}

func WebSearch_20250305(maxUses *int, allowedDomains, blockedDomains []string, userLocation *anthropic.UserLocation) anthropic.Tool {
	tool := providerTool("anthropic.web_search_20250305", "web_search_20250305")
	tool.MaxUses = maxUses
	tool.AllowedDomains = allowedDomains
	tool.BlockedDomains = blockedDomains
	tool.UserLocation = userLocation
	return tool
}

func WebSearch_20260209(maxUses *int, allowedDomains, blockedDomains []string, userLocation *anthropic.UserLocation) anthropic.Tool {
	tool := WebSearch_20250305(maxUses, allowedDomains, blockedDomains, userLocation)
	tool.ID = "anthropic.web_search_20260209"
	tool.Name = "web_search_20260209"
	return tool
}

func ToolSearchRegex_20251119() anthropic.Tool {
	return providerTool("anthropic.tool_search_regex_20251119", "tool_search_regex_20251119")
}
func ToolSearchBm25_20251119() anthropic.Tool {
	return providerTool("anthropic.tool_search_bm25_20251119", "tool_search_bm25_20251119")
}

func Advisor_20260301(model string, maxUses *int, caching *anthropic.CachingConfig) anthropic.Tool {
	tool := providerTool("anthropic.advisor_20260301", "advisor_20260301")
	tool.AdvisorModel = model
	tool.AdvisorMaxUses = maxUses
	tool.AdvisorCaching = caching
	return tool
}

func providerTool(id, name string) anthropic.Tool {
	return anthropic.Tool{ID: id, Name: name, Type: id, ProviderExecuted: true}
}
