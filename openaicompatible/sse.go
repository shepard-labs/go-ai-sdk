package openaicompatible

import (
	"bufio"
	"io"
	"strings"
)

type sseProcessor func(data []byte) bool

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
