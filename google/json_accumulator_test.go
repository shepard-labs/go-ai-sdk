package google

import (
	"reflect"
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

func TestJSONAccumulator_ParsePath(t *testing.T) {
	tests := []struct {
		in   string
		want []any
	}{
		{"name", []any{"name"}},
		{"recipe.ingredients[0].name", []any{"recipe", "ingredients", 0, "name"}},
		{"items[12]", []any{"items", 12}},
		{"a.b.c[0][1][2]", []any{"a", "b", "c", 0, 1, 2}},
		{"", nil},
		{`"weird key".x`, []any{"weird key", "x"}},
	}
	a := &GoogleJSONAccumulator{}
	for _, tt := range tests {
		got, err := a.parsePath(tt.in)
		if err != nil {
			t.Errorf("%q: err = %v", tt.in, err)
			continue
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%q = %v, want %v", tt.in, got, tt.want)
		}
	}
}

func ptrString(s string) *string      { return &s }
func ptrFloat(f float64) *float64     { return &f }
func ptrBool(b bool) *bool            { return &b }
func ptrInt(i int) *int               { return &i }
func ptrAPIPartial(s string) internal.APIPartialArg {
	return internal.APIPartialArg{JSONPath: "", StringValue: ptrString(s)}
}

var _ = ptrAPIPartial // keep helper available for later tasks

func TestJSONAccumulator_NestedObject(t *testing.T) {
	var a GoogleJSONAccumulator
	out, _ := a.Push(internal.APIPartialArg{JSONPath: "user.name", StringValue: ptrString("alice")})
	if out != `{"user":{"name":"alice"` {
		t.Errorf("step1 got %q, want %q", out, `{"user":{"name":"alice"`)
	}
	out, _ = a.Push(internal.APIPartialArg{JSONPath: "user.age", NumberValue: ptrFloat(30)})
	if out != `,"age":30` {
		t.Errorf("step2 got %q, want %q", out, `,"age":30`)
	}
	closing, _ := a.Finalize()
	if closing != `}}` {
		t.Errorf("finalize got %q, want %q", closing, `}}`)
	}
}

func TestJSONAccumulator_ArrayOfObjects(t *testing.T) {
	var a GoogleJSONAccumulator
	a.Push(internal.APIPartialArg{JSONPath: "items[0].name", StringValue: ptrString("a")})
	a.Push(internal.APIPartialArg{JSONPath: "items[0].qty", NumberValue: ptrFloat(1)})
	out, _ := a.Push(internal.APIPartialArg{JSONPath: "items[1].name", StringValue: ptrString("b")})
	if !strings.Contains(out, `,{"name":"b"`) {
		t.Errorf("step3 got %q, want contains %q", out, `,{"name":"b"`)
	}
	closing, _ := a.Finalize()
	if closing != `}]}` {
		t.Errorf("finalize got %q, want %q", closing, `}]}`)
	}
}

func TestJSONAccumulator_PushObjectSimple(t *testing.T) {
	var a GoogleJSONAccumulator
	out, err := a.Push(internal.APIPartialArg{
		JSONPath:    "name",
		StringValue: ptrString("hello"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if out != `{"name":"hello"` {
		t.Errorf("push got %q, want %q", out, `{"name":"hello"`)
	}
	closing, err := a.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	// String was closed inline by the Push, so Finalize just closes the
	// outer object.
	if closing != `}` {
		t.Errorf("finalize got %q, want %q", closing, `}`)
	}
}
