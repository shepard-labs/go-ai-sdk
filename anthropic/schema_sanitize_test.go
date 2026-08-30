package anthropic

import "testing"

func TestSanitizeSchema(t *testing.T) {
	schema := map[string]any{"$schema": "x", "type": "object", "properties": map[string]any{"x": map[string]any{"type": "string", "default": "bad", "examples": []any{"bad"}}, "nested": map[string]any{"type": "object"}}}
	sanitized := SanitizeSchema(schema).(map[string]any)
	if _, ok := sanitized["$schema"]; ok {
		t.Fatalf("schema = %#v", sanitized)
	}
	if sanitized["additionalProperties"] != false {
		t.Fatalf("schema = %#v", sanitized)
	}
	property := sanitized["properties"].(map[string]any)["x"].(map[string]any)
	if _, ok := property["default"]; ok {
		t.Fatalf("property = %#v", property)
	}
	nested := sanitized["properties"].(map[string]any)["nested"].(map[string]any)
	if nested["additionalProperties"] != false {
		t.Fatalf("nested = %#v", nested)
	}
}
