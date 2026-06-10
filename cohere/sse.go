package cohere

import (
	"bufio"
	"io"
	"strings"
)

type sseProcessor func(data []byte) bool

func processSSEStream(reader io.Reader, fn sseProcessor) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(nil, 64*1024)
	var b strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if b.Len() > 0 {
				data := b.String()
				b.Reset()
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
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(data)
		}
	}
	if b.Len() > 0 {
		data := b.String()
		if data != "[DONE]" {
			fn([]byte(data))
		}
	}
}
