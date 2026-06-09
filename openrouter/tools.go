package openrouter

import "encoding/json"

type Tool struct {
	Type            string
	ID              string
	Name            string
	Description     string
	InputSchema     any
	Strict          *bool
	ProviderOptions ProviderMetadata
}

type Tools struct{}

func (Tools) WebSearch(args WebSearchToolArgs) Tool {
	return Tool{Type: "provider-defined", ID: "openrouter.web_search", Name: "web_search", InputSchema: args}
}

type WebSearchToolArgs struct {
	MaxResults   *int
	SearchPrompt string
	Engine       string
}

type Plugin interface{ pluginMarker() }

type Engine string
type PDFEngine string

const (
	EngineNative Engine = "native"
	EngineExa    Engine = "exa"
	EngineAuto   Engine = "auto"

	PDFEngineMistralOCR PDFEngine = "mistral-ocr"
	PDFEnginePDFText    PDFEngine = "pdf-text"
	PDFEngineNative     PDFEngine = "native"
)

type WebPlugin struct {
	MaxResults   *int   `json:"max_results,omitempty"`
	SearchPrompt string `json:"search_prompt,omitempty"`
	Engine       Engine `json:"engine,omitempty"`
}

func (WebPlugin) pluginMarker() {}

type FileParserPlugin struct {
	MaxFiles *int        `json:"max_files,omitempty"`
	PDF      *PDFOptions `json:"pdf,omitempty"`
}

func (FileParserPlugin) pluginMarker() {}

type PDFOptions struct {
	Engine PDFEngine `json:"engine,omitempty"`
}

type ModerationPlugin struct{}

func (ModerationPlugin) pluginMarker() {}

type ResponseHealingPlugin struct{}

func (ResponseHealingPlugin) pluginMarker() {}

type AutoRouterPlugin struct {
	AllowedModels []string `json:"allowed_models,omitempty"`
}

func (AutoRouterPlugin) pluginMarker() {}

type WebSearchOptions struct {
	MaxResults   *int   `json:"max_results,omitempty"`
	SearchPrompt string `json:"search_prompt,omitempty"`
	Engine       Engine `json:"engine,omitempty"`
}

func marshalPlugin(p Plugin) (map[string]any, error) {
	var id string
	switch p.(type) {
	case WebPlugin:
		id = "web"
	case FileParserPlugin:
		id = "file-parser"
	case ModerationPlugin:
		id = "moderation"
	case ResponseHealingPlugin:
		id = "response-healing"
	case AutoRouterPlugin:
		id = "auto-router"
	default:
		return nil, InvalidArgumentError{Message: "unknown plugin type"}
	}
	b, err := json.Marshal(p)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	out["id"] = id
	return out, nil
}

func mapTool(t Tool) (map[string]any, error) {
	if t.ID == "openrouter.web_search" || t.Type == "openrouter.web_search" {
		args, _ := t.InputSchema.(WebSearchToolArgs)
		body := map[string]any{"type": "openrouter:web_search"}
		if args.MaxResults != nil {
			body["max_results"] = *args.MaxResults
		}
		if args.SearchPrompt != "" {
			body["search_prompt"] = args.SearchPrompt
		}
		if args.Engine != "" {
			body["engine"] = args.Engine
		}
		return body, nil
	}
	fn := map[string]any{"name": t.Name}
	if t.Description != "" {
		fn["description"] = t.Description
	}
	if t.InputSchema != nil {
		fn["parameters"] = t.InputSchema
	} else {
		fn["parameters"] = map[string]any{}
	}
	tool := map[string]any{"type": "function", "function": fn}
	if opts, ok := t.ProviderOptions["openrouter"].(map[string]any); ok {
		if v, ok := opts["eager_input_streaming"]; ok {
			tool["eager_input_streaming"] = v
		}
	}
	return tool, nil
}

func mapToolChoice(choice ToolChoice) (any, error) {
	switch choice.Type {
	case "", "auto", "none", "required":
		if choice.Type == "" {
			return nil, nil
		}
		return choice.Type, nil
	case "tool", "function":
		if choice.ToolName == "" {
			return nil, InvalidArgumentError{Message: "tool choice requires tool name"}
		}
		return map[string]any{"type": "function", "function": map[string]any{"name": choice.ToolName}}, nil
	default:
		return nil, InvalidArgumentError{Message: "invalid tool choice " + choice.Type}
	}
}
