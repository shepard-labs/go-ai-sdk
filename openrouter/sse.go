package openrouter

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

type sseEvent struct {
	Event string
	Data  string
}

func parseSSE(r io.Reader, emit func(sseEvent) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var event string
	var data bytes.Buffer
	flush := func() error {
		if data.Len() == 0 {
			event = ""
			return nil
		}
		s := data.String()
		s = strings.TrimSuffix(s, "\n")
		err := emit(sseEvent{Event: event, Data: s})
		event = ""
		data.Reset()
		return err
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, ok := strings.Cut(line, ":")
		if ok && strings.HasPrefix(value, " ") {
			value = value[1:]
		}
		if !ok {
			value = ""
		}
		switch field {
		case "event":
			event = value
		case "data":
			data.WriteString(value)
			data.WriteByte('\n')
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return flush()
}
