package cohere

import "net/http"

func cloneHeader(h http.Header) http.Header {
	out := http.Header{}
	for k, values := range h {
		for _, v := range values {
			out.Add(k, v)
		}
	}
	return out
}
