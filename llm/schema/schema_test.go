package schema

import (
	"encoding/json"
	"testing"
)

func decode(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	return m
}

func TestToolBasicTypes(t *testing.T) {
	type Input struct {
		Query   string  `json:"query" description:"search query"`
		Limit   int     `json:"limit" description:"max results" minimum:"1" maximum:"100"`
		Ratio   float64 `json:"ratio"`
		Enabled bool    `json:"enabled"`
	}
	tool, err := Tool("search", "Search the web", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	if tool.Name != "search" || tool.Description != "Search the web" {
		t.Fatalf("tool meta = %+v", tool)
	}
	m := decode(t, tool.InputSchema)
	if m["type"] != "object" {
		t.Fatalf("type = %v, want object", m["type"])
	}
	props := m["properties"].(map[string]any)
	query := props["query"].(map[string]any)
	if query["type"] != "string" || query["description"] != "search query" {
		t.Fatalf("query = %v", query)
	}
	limit := props["limit"].(map[string]any)
	if limit["type"] != "number" {
		t.Fatalf("limit type = %v, want number", limit["type"])
	}
	if limit["minimum"] != float64(1) || limit["maximum"] != float64(100) {
		t.Fatalf("limit bounds = %v", limit)
	}
	if props["ratio"].(map[string]any)["type"] != "number" {
		t.Fatalf("ratio = %v", props["ratio"])
	}
	if props["enabled"].(map[string]any)["type"] != "boolean" {
		t.Fatalf("enabled = %v", props["enabled"])
	}
}

func TestToolRequiredAndOptional(t *testing.T) {
	type Input struct {
		Required string  `json:"required"`
		Optional *string `json:"optional"`
		Forced   *int    `json:"forced" required:"true"`
	}
	tool, err := Tool("t", "d", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	m := decode(t, tool.InputSchema)
	required := toStringSet(m["required"])
	if !required["required"] {
		t.Fatal("required field missing from required list")
	}
	if required["optional"] {
		t.Fatal("pointer field should be optional")
	}
	if !required["forced"] {
		t.Fatal("required:\"true\" pointer should be required")
	}
}

func TestToolEnum(t *testing.T) {
	type Input struct {
		Color string `json:"color" enum:"red,green,blue"`
	}
	tool, err := Tool("t", "d", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	m := decode(t, tool.InputSchema)
	color := m["properties"].(map[string]any)["color"].(map[string]any)
	enum, ok := color["enum"].([]any)
	if !ok || len(enum) != 3 || enum[0] != "red" || enum[2] != "blue" {
		t.Fatalf("enum = %v", color["enum"])
	}
}

func TestToolSlice(t *testing.T) {
	type Input struct {
		Tags []string `json:"tags"`
	}
	tool, err := Tool("t", "d", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	m := decode(t, tool.InputSchema)
	tags := m["properties"].(map[string]any)["tags"].(map[string]any)
	if tags["type"] != "array" {
		t.Fatalf("tags type = %v, want array", tags["type"])
	}
	items := tags["items"].(map[string]any)
	if items["type"] != "string" {
		t.Fatalf("items type = %v, want string", items["type"])
	}
}

func TestToolNestedStruct(t *testing.T) {
	type Addr struct {
		City string `json:"city"`
		Zip  string `json:"zip"`
	}
	type Input struct {
		Name string `json:"name"`
		Addr Addr   `json:"addr" description:"home address"`
	}
	tool, err := Tool("t", "d", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	m := decode(t, tool.InputSchema)
	addr := m["properties"].(map[string]any)["addr"].(map[string]any)
	if addr["type"] != "object" || addr["description"] != "home address" {
		t.Fatalf("addr = %v", addr)
	}
	nested := addr["properties"].(map[string]any)
	if nested["city"].(map[string]any)["type"] != "string" {
		t.Fatalf("nested city = %v", nested["city"])
	}
	req := toStringSet(addr["required"])
	if !req["city"] || !req["zip"] {
		t.Fatalf("nested required = %v", addr["required"])
	}
}

func TestToolPointerToStruct(t *testing.T) {
	type Addr struct {
		City string `json:"city"`
	}
	type Input struct {
		Addr *Addr `json:"addr"`
	}
	tool, err := Tool("t", "d", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	m := decode(t, tool.InputSchema)
	addr := m["properties"].(map[string]any)["addr"].(map[string]any)
	if addr["type"] != "object" {
		t.Fatalf("addr type = %v, want object", addr["type"])
	}
	if req := m["required"]; req != nil {
		if toStringSet(req)["addr"] {
			t.Fatal("pointer struct should be optional")
		}
	}
}

func TestToolSkipsDashTag(t *testing.T) {
	type Input struct {
		Keep string `json:"keep"`
		Skip string `json:"-"`
	}
	tool, err := Tool("t", "d", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	props := decode(t, tool.InputSchema)["properties"].(map[string]any)
	if _, ok := props["skip"]; ok {
		t.Fatal("json:\"-\" field should be skipped")
	}
	if _, ok := props["keep"]; !ok {
		t.Fatal("keep field missing")
	}
}

func TestToolUsesFieldNameWithoutTag(t *testing.T) {
	type Input struct {
		Plain string
	}
	tool, err := Tool("t", "d", Input{})
	if err != nil {
		t.Fatalf("Tool error = %v", err)
	}
	props := decode(t, tool.InputSchema)["properties"].(map[string]any)
	if _, ok := props["Plain"]; !ok {
		t.Fatalf("expected field name key, got %v", props)
	}
}

func TestToolAcceptsPointerValue(t *testing.T) {
	type Input struct {
		Q string `json:"q"`
	}
	if _, err := Tool("t", "d", &Input{}); err != nil {
		t.Fatalf("pointer value: %v", err)
	}
}

func TestToolErrors(t *testing.T) {
	type unsupported struct {
		Ch chan int `json:"ch"`
	}
	type badMin struct {
		N int `json:"n" minimum:"abc"`
	}
	cases := []struct {
		name string
		v    any
	}{
		{"non-struct", 42},
		{"unsupported field", unsupported{}},
		{"malformed minimum", badMin{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Tool("t", "d", tc.v); err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func toStringSet(v any) map[string]bool {
	set := map[string]bool{}
	if v == nil {
		return set
	}
	for _, item := range v.([]any) {
		set[item.(string)] = true
	}
	return set
}
