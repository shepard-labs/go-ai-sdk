// Package internal holds typed JSON request/response envelopes shared across
// the google package's chat, embedding, image, video, speech, and files
// implementations. These types are not part of the public API.
package internal

// ConvertJSONSchemaToOpenAPISchema is the public re-export of the
// package-private conversion function. The tools subpackage and the
// language model call this for responseSchema and tool parameter conversion.
func ConvertJSONSchemaToOpenAPISchema(schema any) any { return convertJSONSchemaToOpenAPISchema(schema) }

// convertJSONSchemaToOpenAPISchema converts a JSON Schema 7 (subset) value to
// the OpenAPI 3.0 schema shape that Google's responseSchema and tool
// functionDeclarations[].parameters fields expect. The output is a
// map[string]any (or another JSON value) suitable for json.Marshal.
//
// Conversion rules (per spec §"JSON Schema → OpenAPI Conversion"):
//
//   - empty object schema at root → nil
//   - boolean schema → { type: "boolean", properties: {} }
//   - type array containing "null" → anyOf: nonNullTypes, nullable: true
//   - const → enum: [const]
//   - enum passes through (as-is)
//   - anyOf containing { type: "null" } unwraps to nullable single-type
//   - oneOf, allOf mapped 1:1
//   - JSON-Schema-only fields are stripped; OpenAPI-only fields are added
//     (notably "nullable").
func convertJSONSchemaToOpenAPISchema(schema any) any {
	switch v := schema.(type) {
	case nil:
		return nil
	case bool:
		if v {
			return map[string]any{}
		}
		return map[string]any{"not": map[string]any{}}
	}
	m, ok := schema.(map[string]any)
	if !ok {
		// Unknown shape — return as-is. The caller will json.Marshal it.
		return schema
	}
	result := convertSchemaObject(m)
	if result == nil {
		return nil
	}
	return result
}

func convertSchemaObject(m map[string]any) map[string]any {
	if len(m) == 0 {
		return nil
	}
	out := map[string]any{}

	// const / enum precedence: const → enum: [const].
	if c, ok := m["const"]; ok {
		out["enum"] = []any{c}
		// Drop type/properties/etc.; const is exclusive.
		return out
	}
	if e, ok := m["enum"]; ok {
		out["enum"] = e
	}

	// OneOf / AllOf: passthrough after recursive conversion of subschemas.
	if v, ok := m["oneOf"]; ok {
		out["oneOf"] = convertSubschemas(v)
	}
	if v, ok := m["allOf"]; ok {
		out["allOf"] = convertSubschemas(v)
	}

	// anyOf: special case when the only / a member is { type: "null" } — the
	// anyOf unwraps to a nullable single-type schema.
	if v, ok := m["anyOf"]; ok {
		if unwrapped, nullable, handled := convertAnyOf(v); handled {
			for k, val := range unwrapped {
				out[k] = val
			}
			if nullable {
				out["nullable"] = true
			}
		} else {
			out["anyOf"] = convertSubschemas(v)
		}
	}

	// Type array containing "null" → anyOf: nonNullTypes, nullable: true.
	if t, ok := m["type"]; ok {
		switch tt := t.(type) {
		case []any:
			nonNull, hasNull := partitionNonNullTypes(tt)
			if hasNull && len(nonNull) > 0 {
				if len(nonNull) == 1 {
					out["type"] = nonNull[0]
				} else {
					out["type"] = nonNull
				}
				out["nullable"] = true
			} else {
				out["type"] = tt
			}
		case string:
			out["type"] = tt
		default:
			out["type"] = tt
		}
	}

	// Pass through description, format, default, example, title, items,
	// properties, required, additionalProperties, minItems, maxItems, minimum,
	// maximum, exclusiveMinimum, exclusiveMaximum, minLength, maxLength,
	// pattern, minProperties, maxProperties.
	for _, key := range []string{
		"description", "format", "default", "example", "title",
		"items", "properties", "required", "additionalProperties",
		"minItems", "maxItems", "minimum", "maximum",
		"exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
		"pattern", "minProperties", "maxProperties",
	} {
		if v, ok := m[key]; ok {
			out[key] = convertValue(v, key)
		}
	}

	return out
}

func convertValue(v any, key string) any {
	switch key {
	case "properties":
		m, ok := v.(map[string]any)
		if !ok {
			return v
		}
		out := make(map[string]any, len(m))
		for k, sub := range m {
			out[k] = convertJSONSchemaToOpenAPISchema(sub)
		}
		return out
	case "items":
		return convertJSONSchemaToOpenAPISchema(v)
	case "additionalProperties":
		if b, ok := v.(bool); ok {
			return b
		}
		return convertJSONSchemaToOpenAPISchema(v)
	}
	return v
}

func convertSubschemas(v any) []any {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]any, 0, len(arr))
	for _, item := range arr {
		out = append(out, convertJSONSchemaToOpenAPISchema(item))
	}
	return out
}

func convertAnyOf(v any) (map[string]any, bool, bool) {
	arr, ok := v.([]any)
	if !ok {
		return nil, false, false
	}
	// Look for a { type: "null" } member; if present, unwrap.
	hasNull := false
	remaining := make([]any, 0, len(arr))
	for _, item := range arr {
		if m, ok := item.(map[string]any); ok {
			if t, ok := m["type"]; ok {
				if s, ok := t.(string); ok && s == "null" {
					hasNull = true
					continue
				}
			}
		}
		remaining = append(remaining, item)
	}
	if !hasNull {
		return nil, false, false
	}
	// Unwrap: a single remaining schema is promoted; multiple are joined under
	// anyOf.
	if len(remaining) == 1 {
		converted := convertJSONSchemaToOpenAPISchema(remaining[0])
		m, ok := converted.(map[string]any)
		if !ok {
			// Degenerate; return the original anyOf.
			return nil, false, false
		}
		return m, true, true
	}
	out := make(map[string]any, 1)
	out["anyOf"] = convertSubschemas(remaining)
	return out, true, true
}

func partitionNonNullTypes(types []any) ([]any, bool) {
	nonNull := make([]any, 0, len(types))
	hasNull := false
	for _, t := range types {
		if s, ok := t.(string); ok && s == "null" {
			hasNull = true
			continue
		}
		nonNull = append(nonNull, t)
	}
	return nonNull, hasNull
}
