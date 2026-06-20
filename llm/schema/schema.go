// Package schema generates llm.Tool JSON Schemas from Go struct definitions
// using reflection, replacing hand-authored json.RawMessage schemas.
package schema

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// Tool builds an llm.Tool from a struct value, deriving the JSON Schema for the
// tool's input from the struct's fields and tags. The value may be a struct or
// a pointer to a struct. It returns an error for non-struct values, unsupported
// field types, or malformed numeric/bound tags.
func Tool(name, description string, input any) (llm.Tool, error) {
	t := reflect.TypeOf(input)
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil || t.Kind() != reflect.Struct {
		return llm.Tool{}, fmt.Errorf("schema: input must be a struct, got %T", input)
	}
	object, err := objectSchema(t)
	if err != nil {
		return llm.Tool{}, err
	}
	raw, err := json.Marshal(object)
	if err != nil {
		return llm.Tool{}, fmt.Errorf("schema: marshal: %w", err)
	}
	return llm.Tool{Name: name, Description: description, InputSchema: raw}, nil
}

// objectSchema builds an "object" schema for a struct type.
func objectSchema(t reflect.Type) (map[string]any, error) {
	properties := make(map[string]any)
	var required []string
	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		propName, ok := jsonName(field)
		if !ok {
			continue
		}
		prop, err := fieldSchema(field)
		if err != nil {
			return nil, fmt.Errorf("schema: field %s: %w", field.Name, err)
		}
		properties[propName] = prop
		if fieldRequired(field) {
			required = append(required, propName)
		}
	}
	object := map[string]any{"type": "object", "properties": properties}
	if len(required) > 0 {
		object["required"] = required
	}
	return object, nil
}

// fieldSchema builds the schema for one struct field, applying its tags.
func fieldSchema(field reflect.StructField) (map[string]any, error) {
	schema, err := typeSchema(field.Type)
	if err != nil {
		return nil, err
	}
	if desc := field.Tag.Get("description"); desc != "" {
		schema["description"] = desc
	}
	if raw, ok := field.Tag.Lookup("minimum"); ok {
		value, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return nil, fmt.Errorf("malformed minimum %q: %w", raw, err)
		}
		schema["minimum"] = value
	}
	if raw, ok := field.Tag.Lookup("maximum"); ok {
		value, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return nil, fmt.Errorf("malformed maximum %q: %w", raw, err)
		}
		schema["maximum"] = value
	}
	if raw, ok := field.Tag.Lookup("enum"); ok {
		parts := strings.Split(raw, ",")
		enum := make([]any, len(parts))
		for i, part := range parts {
			enum[i] = strings.TrimSpace(part)
		}
		schema["enum"] = enum
	}
	return schema, nil
}

// typeSchema maps a Go type to its JSON Schema fragment, recursing into structs,
// slices, and pointers.
func typeSchema(t reflect.Type) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Slice, reflect.Array:
		items, err := typeSchema(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Struct:
		return objectSchema(t)
	default:
		return nil, fmt.Errorf("unsupported type %s", t.Kind())
	}
}

// jsonName returns the schema property name for a field and whether the field is
// included (false if tagged json:"-").
func jsonName(field reflect.StructField) (string, bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false
	}
	name := strings.Split(tag, ",")[0]
	if name == "" {
		name = field.Name
	}
	return name, true
}

// fieldRequired reports whether a field is required: non-pointer fields are
// required by default, pointer fields are optional unless tagged required:"true".
func fieldRequired(field reflect.StructField) bool {
	if field.Tag.Get("required") == "true" {
		return true
	}
	return field.Type.Kind() != reflect.Pointer
}
