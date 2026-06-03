package anthropic

func SanitizeSchema(schema any) any {
	return sanitizeSchemaValue(schema, 0)
}

func sanitizeSchemaValue(value any, depth int) any {
	if depth > 128 {
		return nil
	}
	switch v := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, child := range v {
			if unsupportedSchemaKeyword(key) {
				continue
			}
			out[key] = sanitizeSchemaValue(child, depth+1)
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, child := range v {
			out[i] = sanitizeSchemaValue(child, depth+1)
		}
		return out
	default:
		return value
	}
}

func unsupportedSchemaKeyword(key string) bool {
	switch key {
	case "$schema", "$id", "$defs", "definitions", "examples", "default", "readOnly", "writeOnly", "deprecated", "unevaluatedProperties", "patternProperties":
		return true
	default:
		return false
	}
}
