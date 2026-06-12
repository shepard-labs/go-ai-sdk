package openai

import (
	"encoding/json"
	"fmt"
)

// convertChatTools converts AI-SDK tool definitions to OpenAI chat
// completion tool payloads. It applies OpenAI-specific behavior:
//   - function tools: {type: "function", function: {name, description,
//     parameters, strict?}}
//   - strict defaults to true for function tools; can be disabled via
//     providerOptions.openai.strict (per-tool or per-call).
//   - provider-defined tools (Type: "provider", ID: "openai.<kind>") are
//     delegated to a per-kind builder.
//   - AddHeaders/WithHints are sent as additional fields on the payload.
func (m *openaiChatLanguageModel) convertChatTools(tools []Tool) ([]map[string]any, []Warning, error) {
	var warnings []Warning
	converted := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if t.Type == "function" {
			payload, w, err := m.convertFunctionTool(t)
			warnings = append(warnings, w...)
			if err != nil {
				return nil, warnings, err
			}
			converted = append(converted, payload)
			continue
		}
		if t.Type == "provider" {
			payload, w, err := m.convertProviderTool(t)
			warnings = append(warnings, w...)
			if err != nil {
				return nil, warnings, err
			}
			converted = append(converted, payload)
			continue
		}
		return nil, warnings, InvalidPromptError{Message: fmt.Sprintf("unsupported tool type %q", t.Type)}
	}
	return converted, warnings, nil
}

func (m *openaiChatLanguageModel) convertFunctionTool(t Tool) (map[string]any, []Warning, error) {
	strict := true
	override := false
	if t.ProviderOptions != nil {
		if openaiOpts, ok := t.ProviderOptions["openai"].(map[string]any); ok {
			if v, ok := openaiOpts["strict"].(bool); ok {
				strict = v
				override = true
			}
		}
	}
	// Per-call override via providerOptions.openai.strict.
	if m.provider != nil {
		// resolved at call level via chatOptions, but tools.go keeps the
		// function pure; chatOptions isn't on the model struct today, so we
		// pass it via the t.ProviderOptions["openai"].strict top-level field.
		_ = override
	}
	fn := map[string]any{
		"name": t.Name,
	}
	if t.Description != "" {
		fn["description"] = t.Description
	}
	if t.InputSchema != nil {
		fn["parameters"] = t.InputSchema
	}
	if strict {
		fn["strict"] = true
	}
	if t.ProviderOptions != nil {
		if openaiOpts, ok := t.ProviderOptions["openai"].(map[string]any); ok {
			for k, v := range openaiOpts {
				if k == "strict" {
					continue
				}
				fn[k] = v
			}
		}
	}
	return map[string]any{
		"type":     "function",
		"function": fn,
	}, nil, nil
}

func (m *openaiChatLanguageModel) convertProviderTool(t Tool) (map[string]any, []Warning, error) {
	if len(t.ID) < len("openai.")+1 || t.ID[:len("openai.")] != "openai." {
		return nil, nil, InvalidPromptError{Message: "provider tool ID must be of the form \"openai.<kind>\""}
	}
	kind := t.ID[len("openai."):]
	argsAny := t.Args
	if argsAny == nil {
		argsAny = map[string]any{}
	}
	args, err := coerceArgs(argsAny)
	if err != nil {
		return nil, nil, err
	}
	switch kind {
	case "applyPatch":
		return map[string]any{"type": "apply_patch"}, nil, nil
	case "codeInterpreter":
		payload := map[string]any{"type": "code_interpreter"}
		if container, ok := args["container"].(map[string]any); ok {
			payload["container"] = container
		}
		return payload, nil, nil
	case "fileSearch":
		payload := map[string]any{"type": "file_search"}
		if v, ok := args["vectorStoreIDs"].([]string); ok {
			payload["vector_store_ids"] = v
		} else if v, ok := args["vectorStoreIDs"].([]any); ok {
			strs := make([]string, 0, len(v))
			for _, x := range v {
				if s, ok := x.(string); ok {
					strs = append(strs, s)
				}
			}
			payload["vector_store_ids"] = strs
		}
		if v, ok := args["maxNumResults"].(float64); ok {
			payload["max_num_results"] = int(v)
		} else if v, ok := args["maxNumResults"].(int); ok {
			payload["max_num_results"] = v
		}
		if v, ok := args["ranking"].(map[string]any); ok {
			payload["ranking"] = v
		}
		if v, ok := args["filters"]; ok {
			payload["filters"] = v
		}
		return payload, nil, nil
	case "imageGeneration":
		payload := map[string]any{"type": "image_generation"}
		if v, ok := args["background"].(string); ok && v != "" {
			payload["background"] = v
		}
		if v, ok := args["inputImageMask"].(map[string]any); ok {
			payload["input_image_mask"] = v
		}
		if v, ok := args["model"].(string); ok && v != "" {
			payload["model"] = v
		}
		if v, ok := args["moderation"].(string); ok && v != "" {
			payload["moderation"] = v
		}
		if v, ok := args["n"].(float64); ok {
			payload["n"] = int(v)
		} else if v, ok := args["n"].(int); ok {
			payload["n"] = v
		}
		if v, ok := args["outputFormat"].(string); ok && v != "" {
			payload["output_format"] = v
		}
		if v, ok := args["partialImages"].(float64); ok {
			payload["partial_images"] = int(v)
		} else if v, ok := args["partialImages"].(int); ok {
			payload["partial_images"] = v
		}
		if v, ok := args["quality"].(string); ok && v != "" {
			payload["quality"] = v
		}
		if v, ok := args["size"].(string); ok && v != "" {
			payload["size"] = v
		}
		if v, ok := args["user"].(string); ok && v != "" {
			payload["user"] = v
		}
		return payload, nil, nil
	case "localShell":
		return map[string]any{"type": "local_shell"}, nil, nil
	case "shell":
		payload := map[string]any{"type": "shell"}
		if v, ok := args["environment"].(map[string]any); ok {
			payload["environment"] = v
		}
		if v, ok := args["networkPolicy"].(map[string]any); ok {
			payload["network_policy"] = v
		}
		if v, ok := args["skills"].([]any); ok {
			payload["skills"] = v
		}
		return payload, nil, nil
	case "webSearch":
		payload := map[string]any{"type": "web_search"}
		if v, ok := args["filters"].(map[string]any); ok {
			payload["filters"] = v
		}
		if v, ok := args["searchContextSize"].(string); ok && v != "" {
			payload["search_context_size"] = v
		}
		if v, ok := args["userLocation"].(map[string]any); ok {
			payload["user_location"] = v
		}
		return payload, nil, nil
	case "webSearchPreview":
		payload := map[string]any{"type": "web_search_preview"}
		if v, ok := args["searchContextSize"].(string); ok && v != "" {
			payload["search_context_size"] = v
		}
		if v, ok := args["userLocation"].(map[string]any); ok {
			payload["user_location"] = v
		}
		return payload, nil, nil
	case "mcp":
		payload := map[string]any{"type": "mcp"}
		if v, ok := args["serverLabel"].(string); ok {
			payload["server_label"] = v
		}
		if v, ok := args["serverUrl"].(string); ok {
			payload["server_url"] = v
		}
		if v, ok := args["headers"].(map[string]any); ok {
			payload["headers"] = v
		}
		if v, ok := args["allowedTools"].([]any); ok {
			payload["allowed_tools"] = v
		} else if v, ok := args["allowedTools"].([]string); ok {
			payload["allowed_tools"] = v
		}
		if v, ok := args["requireApproval"].(map[string]any); ok {
			payload["require_approval"] = v
		}
		return payload, nil, nil
	case "toolSearch":
		payload := map[string]any{"type": "tool_search"}
		if v, ok := args["searchContextSize"].(string); ok && v != "" {
			payload["search_context_size"] = v
		}
		if v, ok := args["userLocation"].(map[string]any); ok {
			payload["user_location"] = v
		}
		return payload, nil, nil
	case "custom":
		payload := map[string]any{"type": "custom"}
		if v, ok := args["description"].(string); ok && v != "" {
			payload["description"] = v
		}
		if v, ok := args["format"]; ok {
			payload["format"] = v
		}
		return payload, nil, nil
	default:
		return nil, nil, InvalidPromptError{Message: "unsupported provider tool kind " + kind}
	}
}

// coerceArgs returns args as a map[string]any. Strings of JSON are
// decoded. Other types result in an InvalidPromptError.
func coerceArgs(args any) (map[string]any, error) {
	switch v := args.(type) {
	case nil:
		return map[string]any{}, nil
	case map[string]any:
		return v, nil
	case map[string]string:
		out := make(map[string]any, len(v))
		for k, val := range v {
			out[k] = val
		}
		return out, nil
	case string:
		var m map[string]any
		if err := json.Unmarshal([]byte(v), &m); err != nil {
			return nil, InvalidPromptError{Message: "tool Args string is not valid JSON: " + err.Error()}
		}
		return m, nil
	default:
		// Typed struct (e.g. FileSearchArgs, WebSearchArgs, MCPArgs, etc.)
		// produced by the openai/tools subpackage factories. Convert via
		// JSON round-trip into a generic map keyed by JSON field names.
		raw, err := json.Marshal(args)
		if err != nil {
			return nil, InvalidPromptError{Message: "tool Args not marshalable: " + err.Error()}
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, InvalidPromptError{Message: "tool Args not unmarshalable as object: " + err.Error()}
		}
		return m, nil
	}
}

// convertChatToolChoice normalizes the AI-SDK tool choice value into
// the OpenAI chat tool_choice payload.
func (m *openaiChatLanguageModel) convertChatToolChoice(choice any, tools []Tool) (any, error) {
	if choice == nil {
		return nil, nil
	}
	switch v := choice.(type) {
	case string:
		switch v {
		case "auto", "none", "required":
			return v, nil
		case "tool":
			// Spec: "tool_choice: 'tool' for non-function" is not supported
			// in chat completions.
			return nil, UnsupportedFunctionalityError{Functionality: "tool_choice \"tool\" for non-function tools"}
		default:
			return nil, InvalidPromptError{Message: "unsupported tool_choice string " + v}
		}
	case map[string]any:
		if toolName, ok := v["toolName"].(string); ok {
			return map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": toolName,
				},
			}, nil
		}
		// Already an OpenAI-shaped tool_choice.
		return v, nil
	}
	// Allow a typed ToolChoice struct via JSON shape.
	bytes, err := json.Marshal(choice)
	if err != nil {
		return nil, err
	}
	var asMap map[string]any
	if err := json.Unmarshal(bytes, &asMap); err == nil {
		if toolName, ok := asMap["toolName"].(string); ok {
			return map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": toolName,
				},
			}, nil
		}
		return asMap, nil
	}
	return nil, InvalidPromptError{Message: "unsupported tool_choice value"}
}
