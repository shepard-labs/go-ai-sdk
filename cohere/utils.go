package cohere

import (
	"encoding/json"
	"net/http"
	"regexp"
)

func cloneHeader(h http.Header) http.Header {
	if h == nil {
		return http.Header{}
	}
	return h.Clone()
}
func cloneRegexpMap(in map[string][]*regexp.Regexp) map[string][]*regexp.Regexp {
	out := make(map[string][]*regexp.Regexp, len(in))
	for k, v := range in {
		out[k] = append([]*regexp.Regexp(nil), v...)
	}
	return out
}
func cloneProviderOptions(in ProviderOptions) ProviderOptions {
	if in == nil {
		return nil
	}
	out := make(ProviderOptions, len(in))
	for k, v := range in {
		inner := make(map[string]any, len(v))
		for ik, iv := range v {
			inner[ik] = iv
		}
		out[k] = inner
	}
	return out
}
func cloneRawMessage(in json.RawMessage) json.RawMessage {
	if in == nil {
		return nil
	}
	return append(json.RawMessage(nil), in...)
}
func intPtr(v int) *int { return &v }
