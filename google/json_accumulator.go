package google

// GoogleJSONAccumulator assembles JSON for a single streaming function-call
// input from a sequence of APIPartialArg updates. The accumulator is scoped
// to a single function-call id and is reset between calls.
//
// The accumulator mirrors the upstream google-json-accumulator.ts behavior:
// it maintains a path stack of CONTAINER entries (one per object/array
// currently open in the output) and emits JSON text fragments that the
// caller concatenates in order. The container kind is inferred from the
// NEXT path segment: a number index means the current container holds an
// array; a string key means it holds an object.
//
// Path semantics: the path's last segment is the LEAF (the value being
// written). All earlier segments are CONTAINERS. The root is implicit
// and is opened by ensureRoot on the first Push.

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// GoogleJSONAccumulator is a single-call scoped JSON builder. Zero value is
// ready to use.
type GoogleJSONAccumulator struct {
	pathStack      []pathEntry
	stringOpen     bool
	closed         bool
	accumulated    map[string]any // structured view of the current state
	hasValue       map[string]bool // path strings that have a set value
}

func (a *GoogleJSONAccumulator) pathKey(segments []any) string {
	var b strings.Builder
	for i, s := range segments {
		if i > 0 {
			b.WriteByte('/')
		}
		switch v := s.(type) {
		case string:
			b.WriteString(v)
		case int:
			b.WriteString(strconv.Itoa(v))
		default:
			fmt.Fprintf(&b, "?%T", v)
		}
	}
	return b.String()
}

func (a *GoogleJSONAccumulator) ensureAccumulated() {
	if a.accumulated == nil {
		a.accumulated = map[string]any{}
		a.hasValue = map[string]bool{}
	}
}

func (a *GoogleJSONAccumulator) getNested(segments []any) (any, bool) {
	if a.accumulated == nil {
		return nil, false
	}
	var cur any = a.accumulated
	for _, seg := range segments {
		switch v := cur.(type) {
		case map[string]any:
			key, ok := seg.(string)
			if !ok {
				return nil, false
			}
			cur, ok = v[key]
			if !ok {
				return nil, false
			}
		case []any:
			idx, ok := seg.(int)
			if !ok {
				return nil, false
			}
			if idx < 0 || idx >= len(v) {
				return nil, false
			}
			cur = v[idx]
		default:
			return nil, false
		}
	}
	return cur, true
}

func (a *GoogleJSONAccumulator) setNested(segments []any, value any) {
	a.ensureAccumulated()
	if len(segments) == 0 {
		return
	}
	var cur any = a.accumulated
	for i, seg := range segments[:len(segments)-1] {
		switch v := cur.(type) {
		case map[string]any:
			key, ok := seg.(string)
			if !ok {
				return
			}
			nv, exists := v[key]
			if !exists {
				nv = map[string]any{}
				v[key] = nv
			}
			cur = nv
			_ = i
		case []any:
			idx, ok := seg.(int)
			if !ok {
				return
			}
			for len(v) <= idx {
				v = append(v, map[string]any{})
			}
			cur = v[idx]
		default:
			return
		}
	}
	last := segments[len(segments)-1]
	switch v := cur.(type) {
	case map[string]any:
		key, ok := last.(string)
		if !ok {
			return
		}
		v[key] = value
	case []any:
		idx, ok := last.(int)
		if !ok {
			return
		}
		for len(v) <= idx {
			v = append(v, nil)
		}
		v[idx] = value
	}
	a.hasValue[a.pathKey(segments)] = true
}

type pathEntry struct {
	segment    any // string for object keys, int for array indices
	isArray    bool
	childCount int
}

// parsePath parses a JSON-path string like "recipe.ingredients[0].name"
// into a list of path segments. Object keys are strings; array indices
// are ints. Returns nil for an empty path.
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
// fragment to append to the call's input. Callers should concatenate each
// returned fragment in order. Once a string is no longer marked
// WillContinue (or no string was ever opened mid-stream), Finalize must
// be called to emit the closing characters.
func (a *GoogleJSONAccumulator) Push(arg internal.APIPartialArg) (string, error) {
	if a.closed {
		return "", fmt.Errorf("google: Push after Finalize")
	}
	segments, err := a.parsePath(arg.JSONPath)
	if err != nil {
		return "", err
	}
	if len(segments) == 0 {
		return "", nil
	}
	// String continuation: if a string was started (willContinue=true) and
	// another string push comes at the same path, append the new value to
	// the existing string and emit just the escaped text fragment.
	if arg.StringValue != nil {
		if existing, ok := a.getNested(segments); ok {
			if prev, ok := existing.(string); ok {
				newStr := prev + *arg.StringValue
				a.setNested(segments, newStr)
				if arg.WillContinue {
					a.stringOpen = true
				}
				// Emit the escaped text (the part to be appended
				// between the previously-opened quote and the
				// eventual close quote).
				return escapeJSONString(*arg.StringValue), nil
			}
		}
	}
	var frag strings.Builder
	if a.stringOpen {
		frag.WriteByte('"')
		a.stringOpen = false
	}
	// Open the root on first call.
	if len(a.pathStack) == 0 {
		a.pathStack = append(a.pathStack, pathEntry{segment: "", isArray: false, childCount: 0})
		frag.WriteByte('{')
	}
	targetContainer := segments[:len(segments)-1]
	leaf := segments[len(segments)-1]
	commonDepth := a.findCommonStackDepth(targetContainer)
	frag.WriteString(a.closeDownTo(commonDepth))
	frag.WriteString(a.openDownTo(targetContainer, leaf))
	valueJSON, emitValueJSON, err := resolvePartialArgValue(arg)
	if err != nil {
		return "", err
	}
	frag.WriteString(a.emitLeaf(leaf, arg, valueJSON, emitValueJSON))
	// Track the value for future continuation detection.
	if err == nil {
		var v any
		if arg.StringValue != nil {
			v = *arg.StringValue
		} else {
			v = valueJSON
		}
		a.setNested(segments, v)
	}
	return frag.String(), nil
}

// findCommonStackDepth returns the deepest stack index (0-based) such that
// the path through [0..commonDepth-1] matches the target container's
// segments. The root is index 0; the first real container is index 1.
func (a *GoogleJSONAccumulator) findCommonStackDepth(target []any) int {
	max := len(a.pathStack) - 1
	if max > len(target) {
		max = len(target)
	}
	common := 0
	for i := 0; i < max; i++ {
		if pathEqual(a.pathStack[i+1].segment, target[i]) {
			common++
		} else {
			break
		}
	}
	return common + 1
}

func pathEqual(a, b any) bool {
	af, aok := a.(float64)
	bf, bok := b.(float64)
	if aok && bok {
		return af == bf
	}
	return a == b
}

// closeDownTo closes containers from the deepest down to (but not
// including) targetDepth. Returns the closing characters.
func (a *GoogleJSONAccumulator) closeDownTo(targetDepth int) string {
	if targetDepth > len(a.pathStack) {
		targetDepth = len(a.pathStack)
	}
	var frag strings.Builder
	for len(a.pathStack) > targetDepth {
		entry := a.pathStack[len(a.pathStack)-1]
		a.pathStack = a.pathStack[:len(a.pathStack)-1]
		if entry.isArray {
			frag.WriteByte(']')
		} else {
			frag.WriteByte('}')
		}
	}
	return frag.String()
}

// openDownTo opens new containers from the current depth down to the
// target container, then prepares to receive the leaf.
func (a *GoogleJSONAccumulator) openDownTo(targetContainer []any, leaf any) string {
	var frag strings.Builder
	startIdx := len(a.pathStack) - 1
	for i := startIdx; i < len(targetContainer); i++ {
		seg := targetContainer[i]
		parent := &a.pathStack[len(a.pathStack)-1]
		if parent.childCount > 0 {
			frag.WriteByte(',')
		}
		parent.childCount++
		if ks, ok := seg.(string); ok {
			frag.WriteByte('"')
			frag.WriteString(escapeJSONString(ks))
			frag.WriteByte('"')
			frag.WriteByte(':')
		}
		// Container kind is determined by the next segment (or leaf).
		var childSeg any
		if i+1 < len(targetContainer) {
			childSeg = targetContainer[i+1]
		} else {
			childSeg = leaf
		}
		isArr := false
		switch childSeg.(type) {
		case int, float64:
			isArr = true
		}
		if isArr {
			frag.WriteByte('[')
		} else {
			frag.WriteByte('{')
		}
		a.pathStack = append(a.pathStack, pathEntry{segment: seg, isArray: isArr, childCount: 0})
	}
	return frag.String()
}

// emitLeaf writes the leaf's separator, key (for objects), and value.
func (a *GoogleJSONAccumulator) emitLeaf(leaf any, arg internal.APIPartialArg, valueJSON, emitValueJSON string) string {
	var frag strings.Builder
	container := &a.pathStack[len(a.pathStack)-1]
	if container.childCount > 0 {
		frag.WriteByte(',')
	}
	container.childCount++
	if ks, ok := leaf.(string); ok {
		frag.WriteByte('"')
		frag.WriteString(escapeJSONString(ks))
		frag.WriteByte('"')
		frag.WriteByte(':')
	}
	if arg.StringValue != nil && arg.WillContinue {
		// Open the string without closing it; remember to close later.
		frag.WriteString(emitValueJSON[:len(emitValueJSON)-1]) // strip the closing quote
		a.stringOpen = true
	} else {
		frag.WriteString(emitValueJSON)
	}
	_ = valueJSON
	return frag.String()
}

// resolvePartialArgValue returns the canonical JSON-encoded value and
// the value-with-trailing-encoding. For strings, emitValueJSON is the
// JSON-quoted form; for willContinue, emitLeaf strips the closing
// quote.
func resolvePartialArgValue(arg internal.APIPartialArg) (string, string, error) {
	switch {
	case arg.StringValue != nil:
		j, _ := jsonStringEncode(*arg.StringValue)
		return j, j, nil
	case arg.NumberValue != nil:
		s := strconv.FormatFloat(*arg.NumberValue, 'f', -1, 64)
		return s, s, nil
	case arg.BoolValue != nil:
		if *arg.BoolValue {
			return "true", "true", nil
		}
		return "false", "false", nil
	case arg.NullValue != nil:
		return "null", "null", nil
	}
	return "", "", fmt.Errorf("google: partialArg has no value field")
}

func jsonStringEncode(s string) (string, error) {
	var b strings.Builder
	b.WriteByte('"')
	b.WriteString(escapeJSONString(s))
	b.WriteByte('"')
	return b.String(), nil
}

// Finalize emits any remaining closing characters. The returned string
// closes any open string and any open containers. After Finalize the
// accumulator must not be used again.
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
	for len(a.pathStack) > 0 {
		entry := a.pathStack[len(a.pathStack)-1]
		a.pathStack = a.pathStack[:len(a.pathStack)-1]
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
