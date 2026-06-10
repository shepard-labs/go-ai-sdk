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

// Push consumes a single partial-args entry and returns the JSON text

// Push consumes a single partial-args entry and returns the JSON text
// fragment to append to the call's input. Callers should concatenate each
// returned fragment in order. Once WillContinue=false has been received
// for a string (or no string was ever opened), Finalize must be called to
// emit the closing characters.
func (a *GoogleJSONAccumulator) Push(arg internal.APIPartialArg) (string, error) {
	if a.closed {
		return "", fmt.Errorf("google: Push after Finalize")
	}
	path, err := a.parsePath(arg.JSONPath)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	if err := a.openPath(path, &b); err != nil {
		return "", err
	}
	if err := a.emitValue(arg, &b); err != nil {
		return "", err
	}
	return b.String(), nil
}

// openPath walks the existing stack and pushes new entries for any path
// segments that exceed the current depth. Each new entry emits its opening
// character and (for objects) its key+colon prefix. Existing entries must
// be consistent with the incoming path.
func (a *GoogleJSONAccumulator) openPath(path []any, b *strings.Builder) error {
	for i, seg := range path {
		if i < len(a.stack) {
			existing := a.stack[i]
			switch v := seg.(type) {
			case string:
				if existing.isArray {
					return fmt.Errorf("google: expected object key, got array index at depth %d", i)
				}
				if existing.segment != v {
					return fmt.Errorf("google: path mismatch at depth %d (have %q, want %q)", i, existing.segment, v)
				}
			case int:
				if !existing.isArray {
					return fmt.Errorf("google: expected object, got array index at depth %d", i)
				}
				_ = v
			}
			continue
		}
		// New container.
		isArray := false
		var segment string
		switch v := seg.(type) {
		case string:
			segment = v
		case int:
			isArray = true
		default:
			return fmt.Errorf("google: unsupported path segment type %T", v)
		}
		a.stack = append(a.stack, stackEntry{segment: segment, isArray: isArray})
		if isArray {
			b.WriteByte('[')
		} else {
			b.WriteByte('{')
			if segment != "" {
				b.WriteByte('"')
				b.WriteString(escapeJSONString(segment))
				b.WriteByte('"')
				b.WriteByte(':')
			}
		}
	}
	return nil
}

// emitValue writes the value portion of a partialArg to b and updates the
// current container's child count.
func (a *GoogleJSONAccumulator) emitValue(arg internal.APIPartialArg, b *strings.Builder) error {
	switch {
	case arg.StringValue != nil:
		if a.stringOpen {
			return fmt.Errorf("google: nested string open")
		}
		if sep := a.childSeparator(); sep != "" {
			b.WriteString(sep)
		}
		b.WriteByte('"')
		b.WriteString(escapeJSONString(*arg.StringValue))
		if arg.WillContinue {
			a.stringOpen = true
			// Don't close the string yet.
		} else {
			b.WriteByte('"')
		}
		a.bumpChild()
	case arg.NumberValue != nil:
		if sep := a.childSeparator(); sep != "" {
			b.WriteString(sep)
		}
		b.WriteString(strconv.FormatFloat(*arg.NumberValue, 'f', -1, 64))
		a.bumpChild()
	case arg.BoolValue != nil:
		if sep := a.childSeparator(); sep != "" {
			b.WriteString(sep)
		}
		if *arg.BoolValue {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
		a.bumpChild()
	case arg.NullValue != nil:
		if sep := a.childSeparator(); sep != "" {
			b.WriteString(sep)
		}
		b.WriteString("null")
		a.bumpChild()
	default:
		return fmt.Errorf("google: partialArg has no value field")
	}
	return nil
}

func (a *GoogleJSONAccumulator) childSeparator() string {
	if len(a.stack) == 0 {
		return ""
	}
	top := &a.stack[len(a.stack)-1]
	if top.childCount == 0 {
		return ""
	}
	return ","
}

func (a *GoogleJSONAccumulator) bumpChild() {
	if len(a.stack) == 0 {
		return
	}
	a.stack[len(a.stack)-1].childCount++
}

// Finalize emits any remaining closing characters. If WillContinue=true was
// never seen, Finalize returns "}" so the call's input is at least "{}".
// After Finalize the accumulator must not be used again.
func (a *GoogleJSONAccumulator) Finalize() (string, error) {
	if a.closed {
		return "", nil
	}
	a.closed = true
	var b strings.Builder
	if a.stringOpen {
		b.WriteByte('"')
		a.stringOpen = false
	}
	for i := len(a.stack) - 1; i >= 0; i-- {
		entry := a.stack[i]
		if entry.isArray {
			b.WriteByte(']')
		} else {
			b.WriteByte('}')
		}
	}
	return b.String(), nil
}

func escapeJSONString(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	return b.String()
}
