package google

// GoogleJSONAccumulator assembles JSON for a single streaming function-call
// input from a sequence of APIPartialArg updates. The accumulator is scoped
// to a single function-call id and is reset between calls.
//
// The accumulator mirrors the upstream google-json-accumulator.ts behavior:
// it maintains a stack of container states (object/array) and a stringOpen
// flag for mid-stream strings, emitting JSON text fragments that the caller
// concatenates in order.

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// GoogleJSONAccumulator is a single-call scoped JSON builder. Zero value is
// ready to use.
type GoogleJSONAccumulator struct {
	stack      []stackEntry
	stringOpen bool
	closed     bool
}

type stackEntry struct {
	segment    string
	isArray    bool
	childCount int
}

// parsePath parses a JSON-path string like "recipe.ingredients[0].name"
// into a list of path segments. Object keys are strings; array indices are
// ints. Returns nil for an empty path.
func (a *GoogleJSONAccumulator) parsePath(path string) ([]any, error) {
	if path == "" {
		return nil, nil
	}
	var out []any
	i := 0
	for i < len(path) {
		if path[i] == '"' {
			// Quoted key.
			j := i + 1
			var sb strings.Builder
			for j < len(path) && path[j] != '"' {
				if path[j] == '\\' && j+1 < len(path) {
					sb.WriteByte(path[j+1])
					j += 2
					continue
				}
				sb.WriteByte(path[j])
				j++
			}
			if j >= len(path) {
				return nil, fmt.Errorf("google: unterminated quoted key in %q", path)
			}
			out = append(out, sb.String())
			i = j + 1
		} else if path[i] == '[' {
			// Array index.
			j := i + 1
			start := j
			for j < len(path) && path[j] >= '0' && path[j] <= '9' {
				j++
			}
			if j == start || j >= len(path) || path[j] != ']' {
				return nil, fmt.Errorf("google: malformed array index in %q", path)
			}
			n, err := strconv.Atoi(path[start:j])
			if err != nil {
				return nil, fmt.Errorf("google: bad array index in %q: %w", path, err)
			}
			out = append(out, n)
			i = j + 1
		} else {
			// Bareword key.
			j := i
			for j < len(path) && path[j] != '.' && path[j] != '[' {
				j++
			}
			out = append(out, path[i:j])
			i = j
		}
		if i < len(path) && path[i] == '.' {
			i++
		}
	}
	return out, nil
}

// _ = internal.APIPartialArg{}  // ensure the import is used once Push lands below.
var _ = internal.APIPartialArg{}
