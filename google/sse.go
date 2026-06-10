package google

// SSE stream processor for the google package.
//
// Behavior is byte-for-byte identical to openaicompatible/sse.go. Copied with
// package attribution; a follow-up PR can promote this into a shared internal
// package.

import (
	"bufio"
	"io"
	"strings"
)

// sseProcessor is a callback invoked for each SSE data payload. Return false
// to stop processing.
type sseProcessor func(data []byte) bool

// processSSEStream reads an SSE response body and calls fn for each complete
// data event. It stops when fn returns false, a "[DONE]" sentinel is received,
// or the stream ends.
func processSSEStream(reader io.Reader, fn sseProcessor) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(nil, 64*1024)
	var builder strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if builder.Len() > 0 {
				data := builder.String()
				builder.Reset()
				if data == "[DONE]" {
					return
				}
				if !fn([]byte(data)) {
					return
				}
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if builder.Len() > 0 {
				builder.WriteByte('\n')
			}
			builder.WriteString(data)
		}
	}
	if builder.Len() > 0 {
		data := builder.String()
		if data != "[DONE]" {
			fn([]byte(data))
		}
	}
}
