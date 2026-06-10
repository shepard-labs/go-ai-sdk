package internal

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestJSONSchemaToOpenAPI_EmptyRoot(t *testing.T) {
	got := convertJSONSchemaToOpenAPISchema(map[string]any{})
	if got != nil {
		t.Errorf("empty root: got %v, want nil", got)
	}
}

func TestJSONSchemaToOpenAPI_BooleanSchema(t *testing.T) {
	gotTrue := convertJSONSchemaToOpenAPISchema(true)
	if !reflect.DeepEqual(gotTrue, map[string]any{}) {
		t.Errorf("true schema: got %v, want {}", gotTrue)
	}
	gotFalse := convertJSONSchemaToOpenAPISchema(false)
	wantFalse := map[string]any{"not": map[string]any{}}
	if !reflect.DeepEqual(gotFalse, wantFalse) {
		t.Errorf("false schema: got %v, want %v", gotFalse, wantFalse)
	}
}

func TestJSONSchemaToOpenAPI_SimpleObject(t *testing.T) {
	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		"required": []any{"name"},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	if got["type"] != "object" {
		t.Errorf("type = %v, want object", got["type"])
	}
	props := got["properties"].(map[string]any)
	if props["name"].(map[string]any)["type"] != "string" {
		t.Errorf("name type = %v, want string", props["name"])
	}
	if got["required"].([]any)[0] != "name" {
		t.Errorf("required[0] = %v, want name", got["required"])
	}
}

func TestJSONSchemaToOpenAPI_NestedObjects(t *testing.T) {
	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"address": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	addr := got["properties"].(map[string]any)["address"].(map[string]any)
	if addr["type"] != "object" {
		t.Errorf("address.type = %v, want object", addr["type"])
	}
	city := addr["properties"].(map[string]any)["city"].(map[string]any)
	if city["type"] != "string" {
		t.Errorf("address.city.type = %v, want string", city["type"])
	}
}

func TestJSONSchemaToOpenAPI_Array(t *testing.T) {
	in := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "string",
		},
		"minItems": 1,
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	if got["type"] != "array" {
		t.Errorf("type = %v, want array", got["type"])
	}
	items := got["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("items.type = %v, want string", items["type"])
	}
	// minItems is passed through verbatim (the wire decoder produces float64;
	// here we use a Go int literal).
	if _, ok := got["minItems"]; !ok {
		t.Errorf("minItems missing from output")
	}
}

func TestJSONSchemaToOpenAPI_Enum(t *testing.T) {
	in := map[string]any{
		"type": "string",
		"enum": []any{"a", "b", "c"},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	if got["type"] != "string" {
		t.Errorf("type = %v, want string", got["type"])
	}
	enum := got["enum"].([]any)
	if len(enum) != 3 || enum[0].(string) != "a" {
		t.Errorf("enum = %v, want [a b c]", enum)
	}
}

func TestJSONSchemaToOpenAPI_Const(t *testing.T) {
	in := map[string]any{
		"const": "fixed",
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	enum, ok := got["enum"].([]any)
	if !ok || len(enum) != 1 || enum[0].(string) != "fixed" {
		t.Errorf("const: got %v, want enum: [fixed]", got)
	}
}

func TestJSONSchemaToOpenAPI_AnyOfWithNull(t *testing.T) {
	in := map[string]any{
		"anyOf": []any{
			map[string]any{"type": "string"},
			map[string]any{"type": "null"},
		},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	if got["type"] != "string" {
		t.Errorf("type = %v, want string", got["type"])
	}
	if got["nullable"] != true {
		t.Errorf("nullable = %v, want true", got["nullable"])
	}
}

func TestJSONSchemaToOpenAPI_AnyOfWithNull_Multiple(t *testing.T) {
	in := map[string]any{
		"anyOf": []any{
			map[string]any{"type": "string"},
			map[string]any{"type": "null"},
			map[string]any{"type": "integer"},
		},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	if got["nullable"] != true {
		t.Errorf("nullable = %v, want true", got["nullable"])
	}
	anyOf, ok := got["anyOf"].([]any)
	if !ok {
		t.Fatalf("anyOf: got %T, want []any", got["anyOf"])
	}
	if len(anyOf) != 2 {
		t.Errorf("anyOf length = %d, want 2", len(anyOf))
	}
}

func TestJSONSchemaToOpenAPI_TypeArrayWithNull(t *testing.T) {
	in := map[string]any{
		"type": []any{"string", "null"},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	if got["type"] != "string" {
		t.Errorf("type = %v, want string", got["type"])
	}
	if got["nullable"] != true {
		t.Errorf("nullable = %v, want true", got["nullable"])
	}
}

func TestJSONSchemaToOpenAPI_OneOf(t *testing.T) {
	in := map[string]any{
		"oneOf": []any{
			map[string]any{"type": "string"},
			map[string]any{"type": "integer"},
		},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	oneOf, ok := got["oneOf"].([]any)
	if !ok {
		t.Fatalf("oneOf: got %T, want []any", got["oneOf"])
	}
	if len(oneOf) != 2 {
		t.Errorf("oneOf length = %d, want 2", len(oneOf))
	}
}

func TestJSONSchemaToOpenAPI_AllOf(t *testing.T) {
	in := map[string]any{
		"allOf": []any{
			map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}},
			map[string]any{"type": "object", "properties": map[string]any{"b": map[string]any{"type": "integer"}}},
		},
	}
	got := convertJSONSchemaToOpenAPISchema(in).(map[string]any)
	allOf, ok := got["allOf"].([]any)
	if !ok || len(allOf) != 2 {
		t.Fatalf("allOf: got %v, want 2 entries", got["allOf"])
	}
}

func TestJSONSchemaToOpenAPI_RoundTrip(t *testing.T) {
	in := map[string]any{
		"type":     "object",
		"required": []any{"a"},
		"properties": map[string]any{
			"a": map[string]any{"type": "string"},
		},
	}
	got := convertJSONSchemaToOpenAPISchema(in)
	enc, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back map[string]any
	if err := json.Unmarshal(enc, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back["type"] != "object" {
		t.Errorf("roundtrip type = %v, want object", back["type"])
	}
}

func TestJSONSchemaToOpenAPI_Nil(t *testing.T) {
	if got := convertJSONSchemaToOpenAPISchema(nil); got != nil {
		t.Errorf("nil: got %v, want nil", got)
	}
}
