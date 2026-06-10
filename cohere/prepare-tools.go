package cohere

func prepareTools(tools []Tool, choice *ToolChoice) ([]map[string]any, *string, []Warning, error) {
	if len(tools) == 0 {
		return nil, nil, nil, nil
	}
	converted := make([]map[string]any, 0, len(tools))
	var warnings []Warning
	for _, tool := range tools {
		if tool.Type == "provider" {
			warnings = append(warnings, Warning{Type: "unsupported", Feature: "provider-defined tool " + tool.ID})
			continue
		}
		fn := map[string]any{"name": tool.Name, "parameters": tool.InputSchema}
		if tool.Description != "" {
			fn["description"] = tool.Description
		}
		converted = append(converted, map[string]any{"type": "function", "function": fn})
	}
	if choice == nil || choice.Type == "" || choice.Type == "auto" {
		return converted, nil, warnings, nil
	}
	switch choice.Type {
	case "none":
		v := "NONE"
		return converted, &v, warnings, nil
	case "required":
		v := "REQUIRED"
		return converted, &v, warnings, nil
	case "tool":
		v := "REQUIRED"
		filtered := []map[string]any{}
		for _, tool := range converted {
			fn, _ := tool["function"].(map[string]any)
			if fn["name"] == choice.ToolName {
				filtered = append(filtered, tool)
			}
		}
		return filtered, &v, warnings, nil
	default:
		return nil, nil, warnings, UnsupportedFunctionalityError{Functionality: "tool choice type: " + choice.Type}
	}
}
