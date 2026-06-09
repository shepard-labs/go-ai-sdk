package openaicompatible

import (
	"net/http"
	"regexp"
	"strings"
	"unicode"
)

func cloneHeader(h http.Header) http.Header {
	if h == nil {
		return http.Header{}
	}
	return h.Clone()
}

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

func cloneRegexpMap(in map[string][]*regexp.Regexp) map[string][]*regexp.Regexp {
	out := make(map[string][]*regexp.Regexp, len(in))
	for k, values := range in {
		out[k] = append([]*regexp.Regexp(nil), values...)
	}
	return out
}

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

func metadataKeyForProviderOptions(name string, opts ProviderOptions) string {
	camel := toCamelCase(name)
	if camel != name {
		if _, ok := opts[camel]; ok {
			return camel
		}
	}
	return name
}
