// Package tools holds the Google provider-tool factory set and the
// prepareTools dispatcher used by the language model to convert a slice of
// provider-tool descriptors into the wire [internal.APITool] entries plus
// an optional [internal.APIToolConfig].
//
// The ToolView struct decouples this subpackage from the parent google
// package: callers convert their google.Tool values into ToolView values
// before calling [PrepareTools]. This avoids the import cycle that would
// otherwise arise from google importing tools.
package tools

import (
	"encoding/json"
	"sort"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// ToolView is a plain-data view of a [google.Tool] used by [PrepareTools].
type ToolView struct {
	Type             string
	ID               string
	Name             string
	Description      string
	InputSchema      any
	ArgsSchema       any
	Strict           *bool
	ProviderExecuted bool
	Dynamic          bool
}

// ToolChoiceView mirrors google.ToolChoice for this subpackage.
type ToolChoiceView struct {
	Type     string
	ToolName string
}

// ProviderOptionsView is the per-tool provider options relevant to this
// dispatcher (used for the streamFunctionCallArguments signal).
type ProviderOptionsView map[string]map[string]any

// PrepareToolsOpts is the input to [PrepareTools].
type PrepareToolsOpts struct {
	// ModelID is used to detect Gemini 3+ for the includeServerSideToolInvocations
	// mixed-tool behavior.
	ModelID string
	// IsVertexProvider, when true, enables Vertex-only behaviors.
	IsVertexProvider bool
	// IsStreaming, when true, allows emitting streamFunctionCallArguments
	// in toolConfig (Vertex only).
	IsStreaming bool
	// ToolChoice, when non-nil, maps to functionCallingConfig.mode.
	ToolChoice *ToolChoiceView
	// StreamFunctionCallArguments, when non-nil and on Vertex streaming,
	// sets toolConfig.functionCallingConfig.streamFunctionCallArguments.
	StreamFunctionCallArguments *bool
}

// Warning mirrors google.Warning for this subpackage.
type Warning struct {
	Type    string
	Feature string
	Details string
	Message string
}

// PrepareTools converts a slice of [ToolView] into the wire tools[] entries
// plus an optional toolConfig.
func PrepareTools(input []ToolView, opts PrepareToolsOpts) ([]internal.APITool, *internal.APIToolConfig, []Warning, error) {
	if len(input) == 0 {
		return nil, nil, nil, nil
	}

	var providerTools []ToolView
	var functionTools []ToolView
	var warnings []Warning
	for _, t := range input {
		if isProviderTool(t) {
			providerTools = append(providerTools, t)
		} else {
			functionTools = append(functionTools, t)
		}
	}

	gemini3 := isGemini3(opts.ModelID)

	// Mixed-tools check.
	if len(providerTools) > 0 && len(functionTools) > 0 && !gemini3 {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "mixed function and provider tools on pre-Gemini-3 models",
			Message: "mixed tools require Gemini 3+; the function tools will be ignored",
		})
		// Per spec, drop function tools; emit provider tools only.
		functionTools = nil
	}

	out := make([]internal.APITool, 0, len(input))
	for _, t := range providerTools {
		wire, ws, err := providerToolWire(t, opts)
		if err != nil {
			return nil, nil, append(warnings, ws...), err
		}
		warnings = append(warnings, ws...)
		out = append(out, wire)
	}
	if len(functionTools) > 0 {
		decls := make([]map[string]any, 0, len(functionTools))
		for _, t := range functionTools {
			decls = append(decls, functionDeclaration(t))
		}
		out = append(out, internal.APITool{Body: map[string]any{"functionDeclarations": decls}})
	}

	cfg, ws, err := buildToolConfig(functionTools, providerTools, opts)
	if err != nil {
		return nil, nil, append(warnings, ws...), err
	}
	warnings = append(warnings, ws...)
	return out, cfg, warnings, nil
}

// isGemini3 reports whether the model is Gemini 3 or newer. Mirrors
// google.IsGemini3ForTools but lives here to avoid the import cycle.
func isGemini3(modelID string) bool {
	// Use a simple case-insensitive prefix match; we only check for the
	// "gemini-3" family prefix here. The full predicate is in the parent
	// package for richer semantics.
	if len(modelID) < 8 {
		return false
	}
	prefix := modelID
	if len(prefix) > 8 {
		prefix = prefix[:8]
	}
	// Cheap case-fold for the common ASCII case.
	if prefix[7] >= 'A' && prefix[7] <= 'Z' {
		prefix = prefix[:7] + string(prefix[7]+32)
	}
	return prefix == "gemini-3" || hasGemini3DotPrefix(modelID)
}

func hasGemini3DotPrefix(modelID string) bool {
	if len(modelID) < 9 {
		return false
	}
	return modelID[:9] == "gemini-3." || modelID[:9] == "GEMINI-3." ||
		(modelID[0] == 'g' && modelID[:9] == "gemini-3.")
}

func isProviderTool(t ToolView) bool {
	if t.Type == "provider" {
		return true
	}
	// Heuristic: a Tool with an ID like "google.<name>".
	if len(t.ID) > 7 && t.ID[:7] == "google." {
		return true
	}
	return false
}

func providerToolWire(t ToolView, opts PrepareToolsOpts) (internal.APITool, []Warning, error) {
	var warnings []Warning
	id := t.ID
	if id == "" {
		id = t.Type
	}
	switch id {
	case "google.google_search":
		body := map[string]any{"googleSearch": map[string]any{}}
		if args, ok := t.ArgsSchema.(GoogleSearchArgs); ok {
			filled := fillGoogleSearchArgs(args)
			if len(filled) > 0 {
				body["googleSearch"] = filled
			}
		}
		return internal.APITool{Body: body}, warnings, nil
	case "google.enterprise_web_search":
		return internal.APITool{Body: map[string]any{"enterpriseWebSearch": map[string]any{}}}, warnings, nil
	case "google.url_context":
		return internal.APITool{Body: map[string]any{"urlContext": map[string]any{}}}, warnings, nil
	case "google.code_execution":
		return internal.APITool{Body: map[string]any{"codeExecution": map[string]any{}}}, warnings, nil
	case "google.file_search":
		body := map[string]any{"fileSearch": map[string]any{}}
		if args, ok := t.ArgsSchema.(FileSearchArgs); ok {
			filled := fillFileSearchArgs(args)
			if len(filled) > 0 {
				body["fileSearch"] = filled
			}
		}
		return internal.APITool{Body: body}, warnings, nil
	case "google.google_maps":
		return internal.APITool{Body: map[string]any{"googleMaps": map[string]any{}}}, warnings, nil
	case "google.vertex_rag_store":
		if !opts.IsVertexProvider {
			warnings = append(warnings, Warning{
				Type:    "other",
				Feature: "vertexRagStore",
				Message: "vertexRagStore is Vertex-only and will be ignored on the public Gemini API",
			})
			return internal.APITool{Body: map[string]any{"retrieval": map[string]any{}}}, warnings, nil
		}
		body := map[string]any{
			"retrieval": map[string]any{
				"vertex_rag_store": map[string]any{
					"rag_resources": map[string]any{"rag_corpus": ""},
				},
			},
		}
		if args, ok := t.ArgsSchema.(VertexRagStoreArgs); ok {
			rag := body["retrieval"].(map[string]any)["vertex_rag_store"].(map[string]any)
			rag["rag_resources"] = map[string]any{"rag_corpus": args.RagCorpus}
			if args.TopK != nil {
				rag["similarity_top_k"] = *args.TopK
			}
		}
		return internal.APITool{Body: body}, warnings, nil
	default:
		// Unknown provider tool id: fall through to passthrough wrapping.
		if t.ArgsSchema != nil {
			b, _ := json.Marshal(t.ArgsSchema)
			var inner any
			_ = json.Unmarshal(b, &inner)
			return internal.APITool{Body: map[string]any{t.Name: inner}}, warnings, nil
		}
		return internal.APITool{Body: map[string]any{t.Name: map[string]any{}}}, warnings, nil
	}
}

func functionDeclaration(t ToolView) map[string]any {
	decl := map[string]any{"name": t.Name}
	if t.Description != "" {
		decl["description"] = t.Description
	}
	if t.InputSchema != nil {
		decl["parameters"] = internal.ConvertJSONSchemaToOpenAPISchema(t.InputSchema)
	}
	return decl
}

func buildToolConfig(functionTools, providerTools []ToolView, opts PrepareToolsOpts) (*internal.APIToolConfig, []Warning, error) {
	var warnings []Warning
	if len(functionTools) == 0 && len(providerTools) == 0 {
		return nil, warnings, nil
	}
	var cfg internal.APIToolConfig
	gemini3 := isGemini3(opts.ModelID)
	mixed := len(functionTools) > 0 && len(providerTools) > 0

	if len(functionTools) > 0 {
		anyStrict := false
		for _, t := range functionTools {
			if t.Strict != nil && *t.Strict {
				anyStrict = true
				break
			}
		}
		mode, allowed := toolChoiceToMode(opts.ToolChoice, anyStrict, mixed && gemini3)
		if mode != "" {
			cfg.FunctionCallingConfig = &internal.APIFunctionCallingConfig{Mode: mode, AllowedFunctionNames: allowed}
		} else if len(allowed) > 0 {
			cfg.FunctionCallingConfig = &internal.APIFunctionCallingConfig{AllowedFunctionNames: allowed}
		}
		if opts.IsStreaming && opts.IsVertexProvider && opts.StreamFunctionCallArguments != nil && *opts.StreamFunctionCallArguments {
			if cfg.FunctionCallingConfig == nil {
				cfg.FunctionCallingConfig = &internal.APIFunctionCallingConfig{}
			}
			b := true
			cfg.FunctionCallingConfig.StreamFunctionCallArguments = &b
		} else if !opts.IsVertexProvider && opts.StreamFunctionCallArguments != nil {
			warnings = append(warnings, Warning{
				Type:    "other",
				Feature: "streamFunctionCallArguments",
				Message: "streamFunctionCallArguments is Vertex-only and will be ignored on the public Gemini API",
			})
		}
	}

	if mixed && gemini3 && !opts.IsVertexProvider {
		b := true
		cfg.IncludeServerSideToolInvocations = &b
	}

	if cfg.FunctionCallingConfig == nil && cfg.IncludeServerSideToolInvocations == nil {
		return nil, warnings, nil
	}
	return &cfg, warnings, nil
}

func toolChoiceToMode(choice *ToolChoiceView, anyStrict, mixedGemini3 bool) (string, []string) {
	if choice == nil {
		if mixedGemini3 {
			return "VALIDATED", nil
		}
		if anyStrict {
			return "VALIDATED", nil
		}
		return "", nil
	}
	switch choice.Type {
	case "auto":
		if anyStrict {
			return "VALIDATED", nil
		}
		return "AUTO", nil
	case "none":
		return "NONE", nil
	case "required":
		if anyStrict {
			return "VALIDATED", nil
		}
		return "ANY", nil
	case "tool":
		if anyStrict {
			return "VALIDATED", []string{choice.ToolName}
		}
		return "ANY", []string{choice.ToolName}
	}
	return "", nil
}

func fillGoogleSearchArgs(args GoogleSearchArgs) map[string]any {
	if args.SearchTypes == nil && args.TimeRangeFilter == nil {
		return nil
	}
	out := map[string]any{}
	if args.SearchTypes != nil {
		types := map[string]any{}
		if args.SearchTypes.WebSearch != nil {
			types["web_search"] = args.SearchTypes.WebSearch
		}
		if args.SearchTypes.ImageSearch != nil {
			types["image_search"] = args.SearchTypes.ImageSearch
		}
		if len(types) > 0 {
			out["searchTypes"] = types
		}
	}
	if args.TimeRangeFilter != nil {
		out["timeRangeFilter"] = map[string]any{
			"startTime": args.TimeRangeFilter.StartTime,
			"endTime":   args.TimeRangeFilter.EndTime,
		}
	}
	return out
}

func fillFileSearchArgs(args FileSearchArgs) map[string]any {
	if len(args.FileSearchStoreNames) == 0 && args.TopK == nil && args.MetadataFilter == "" {
		return nil
	}
	out := map[string]any{}
	if len(args.FileSearchStoreNames) > 0 {
		sort.Strings(args.FileSearchStoreNames)
		out["fileSearchStoreNames"] = append([]string(nil), args.FileSearchStoreNames...)
	}
	if args.TopK != nil {
		out["topK"] = *args.TopK
	}
	if args.MetadataFilter != "" {
		out["metadataFilter"] = args.MetadataFilter
	}
	return out
}

// ---- Tool factory set ----

// Tool mirrors [google.Tool] for use within this subpackage.
// Defined locally to avoid an import cycle: google imports tools, so tools
// cannot import google directly.
type Tool struct {
	ID               string
	Name             string
	Type             string
	ArgsSchema       any
	InputSchema      any
	Strict           *bool
	ProviderExecuted bool
	Dynamic          bool
}

// ToolFactories mirrors [google.ToolFactories] for use within this subpackage.
// Defined locally to avoid an import cycle.
type ToolFactories struct {
	GoogleSearch        func(args ...GoogleSearchArgs) Tool
	EnterpriseWebSearch func() Tool
	GoogleMaps          func() Tool
	UrlContext          func() Tool
	FileSearch          func(args FileSearchArgs) Tool
	CodeExecution       func() Tool
	VertexRagStore      func(args VertexRagStoreArgs) Tool
}

// Tools holds the Google provider-tool factory set.
type Tools struct{}

// GoogleSearch returns a google_search tool, optionally configured with args.
func (Tools) GoogleSearch(args ...GoogleSearchArgs) Tool {
	a := any(nil)
	if len(args) > 0 {
		a = args[0]
	}
	return buildTool("google.google_search", "google_search", a)
}

// EnterpriseWebSearch returns an enterprise_web_search tool.
func (Tools) EnterpriseWebSearch() Tool {
	return buildTool("google.enterprise_web_search", "enterprise_web_search", nil)
}

// GoogleMaps returns a google_maps tool.
func (Tools) GoogleMaps() Tool {
	return buildTool("google.google_maps", "google_maps", nil)
}

// UrlContext returns a url_context tool.
func (Tools) UrlContext() Tool {
	return buildTool("google.url_context", "url_context", nil)
}

// FileSearch returns a file_search tool, optionally configured with args.
func (Tools) FileSearch(args FileSearchArgs) Tool {
	return buildTool("google.file_search", "file_search", args)
}

// CodeExecution returns a code_execution tool.
func (Tools) CodeExecution() Tool {
	return buildTool("google.code_execution", "code_execution", nil)
}

// VertexRagStore returns a vertex_rag_store tool, optionally configured with args.
func (Tools) VertexRagStore(args VertexRagStoreArgs) Tool {
	return buildTool("google.vertex_rag_store", "vertex_rag_store", args)
}

// Build returns the full ToolFactories from the default Tools instance.
func (Tools) Build() ToolFactories {
	return ToolFactories{
		GoogleSearch:        func(args ...GoogleSearchArgs) Tool { return tools.GoogleSearch(args...) },
		EnterpriseWebSearch: func() Tool { return tools.EnterpriseWebSearch() },
		GoogleMaps:          func() Tool { return tools.GoogleMaps() },
		UrlContext:          func() Tool { return tools.UrlContext() },
		FileSearch:          func(args FileSearchArgs) Tool { return tools.FileSearch(args) },
		CodeExecution:       func() Tool { return tools.CodeExecution() },
		VertexRagStore:      func(args VertexRagStoreArgs) Tool { return tools.VertexRagStore(args) },
	}
}

// tools is the default Tools instance.
var tools = Tools{}

func buildTool(id, name string, argsSchema any) Tool {
	return Tool{ID: id, Name: name, Type: "provider", ArgsSchema: argsSchema}
}
