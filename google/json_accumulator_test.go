package google

import (
	"reflect"
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
