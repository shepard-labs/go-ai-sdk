package google

import (
	"net/http"
	"regexp"
	"strings"
	"unicode"
)

// cloneHeader returns a deep copy of the given header map. Returns an empty
// header when h is nil.
//
// Copied from openaicompatible/utils.go with package attribution.
func cloneHeader(h http.Header) http.Header {
	if h == nil {
		return http.Header{}
	}
	return h.Clone()
}

// cloneStringMap returns a shallow copy of a string-string map.
func cloneStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// cloneRegexpMap returns a shallow copy of a string-to-regexp-slice map.
func cloneRegexpMap(in map[string][]*regexp.Regexp) map[string][]*regexp.Regexp {
	out := make(map[string][]*regexp.Regexp, len(in))
	for k, values := range in {
		out[k] = append([]*regexp.Regexp(nil), values...)
	}
	return out
}

// toCamelCase converts a kebab-case or snake_case string to camelCase.
//
// Copied from openaicompatible/utils.go with package attribution.
func toCamelCase(str string) string {
	var b strings.Builder
	b.Grow(len(str))
	var separator rune
	for _, r := range str {
		if separator != 0 {
			if r >= 'a' && r <= 'z' {
				b.WriteRune(unicode.ToUpper(r))
				separator = 0
				continue
			}
			b.WriteRune(separator)
			separator = 0
		}
		if r == '_' || r == '-' {
			separator = r
			continue
		}
		b.WriteRune(r)
	}
	if separator != 0 {
		b.WriteRune(separator)
	}
	return b.String()
}

// metadataKeyForProviderOptions returns the correct key to use when looking up
// options from a ProviderOptions map, preferring the camelCase form when it
// exists in opts.
//
// Copied from openaicompatible/utils.go with package attribution.
func metadataKeyForProviderOptions(name string, opts ProviderOptions) string {
	camel := toCamelCase(name)
	if camel != name {
		if _, ok := opts[camel]; ok {
			return camel
		}
	}
	return name
}

// maxPositiveOrDefault returns value if positive, otherwise fallback.
func maxPositiveOrDefault(value, fallback int64) int64 {
	if value > 0 {
		return value
	}
	return fallback
}

// getModelPath returns the model endpoint path segment. If id contains a slash
// it is returned verbatim (Vertex-style full resource path); otherwise it is
// wrapped as "models/<id>".
func getModelPath(id string) string {
	if strings.Contains(id, "/") {
		return id
	}
	return "models/" + id
}
