package anthropic

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"
)

func parsePhase3(t *testing.T, sse string) []StreamPart {
	t.Helper()
	out := make(chan StreamPart, 32)
	parseSSEStream(context.Background(), strings.NewReader(sse), out, nil)
	close(out)
	var parts []StreamPart
	for part := range out {
		parts = append(parts, part)
	}
	return parts
}

func event(name, data string) string { return "event: " + name + "\ndata: " + data + "\n\n" }

func TestPhase3DuplicateMessageStartAndUsageMerge(t *testing.T) {
	start := `{"message":{"id":"m","model":"claude","usage":{"input_tokens":3,"cache_read_input_tokens":2}}}`
	parts := parsePhase3(t, event("message_start", start)+event("message_start", start)+event("message_delta", `{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`)+event("message_stop", `{}`))
	if len(parts) != 3 {
		t.Fatalf("parts = %#v", parts)
	}
	finish := parts[2].(StreamFinish)
	if finish.Usage.InputTokens.Total != 3 || finish.Usage.InputTokens.CacheRead != 2 || finish.Usage.OutputTokens.Total != 4 {
		t.Fatalf("usage = %#v", finish.Usage)
	}

	parts = parsePhase3(t, event("message_start", start)+event("message_start", `{"message":{"id":"other","model":"claude","usage":{"input_tokens":3}}}`))
	if _, ok := parts[len(parts)-1].(StreamError); !ok {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestPhase3InvalidTransitions(t *testing.T) {
	cases := []string{
		event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`),
		event("message_start", `{"message":{"id":"m","model":"c"}}`) + event("content_block_delta", `{"index":0,"delta":{"type":"text_delta","text":"x"}}`),
		event("message_start", `{"message":{"id":"m","model":"c"}}`) + event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) + event("content_block_delta", `{"index":0,"delta":{"type":"thinking_delta","thinking":"x"}}`),
		event("message_start", `{"message":{"id":"m","model":"c"}}`) + event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) + event("message_stop", `{}`),
	}
	for i, sse := range cases {
		parts := parsePhase3(t, sse)
		if _, ok := parts[len(parts)-1].(StreamError); !ok {
			t.Fatalf("case %d parts = %#v", i, parts)
		}
	}
}

func TestPhase3MessageDeltaOnlyAllowsMessageStop(t *testing.T) {
	prefix := event("message_start", `{"message":{"id":"m","model":"c"}}`) +
		event("message_delta", `{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`)
	cases := []struct {
		name, trailing string
	}{
		{"duplicate message delta", event("message_delta", `{"delta":{"stop_reason":"end_turn"}}`)},
		{"content block start", event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`)},
		{"content block delta", event("content_block_delta", `{"index":0,"delta":{"type":"text_delta","text":"x"}}`)},
		{"content block stop", event("content_block_stop", `{"index":0}`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			parts := parsePhase3(t, prefix+tc.trailing)
			if _, ok := parts[len(parts)-1].(StreamError); !ok {
				t.Fatalf("parts = %#v", parts)
			}
		})
	}

	parts := parsePhase3(t, prefix+event("ping", `{}`)+event("message_stop", `{}`))
	if _, ok := parts[len(parts)-1].(StreamFinish); !ok {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestPhase3BlockSpecificEventsAndCitation(t *testing.T) {
	sse := event("message_start", `{"message":{"id":"m","model":"c"}}`) +
		event("content_block_start", `{"index":2,"content_block":{"type":"text"}}`) +
		event("content_block_delta", `{"index":2,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://example.com"}}}`) +
		event("content_block_stop", `{"index":2}`) + event("message_stop", `{}`)
	parts := parsePhase3(t, sse)
	delta := parts[3].(StreamTextDelta)
	if len(delta.Citations) != 1 || delta.Citations[0].URL != "https://example.com" {
		t.Fatalf("parts = %#v", parts)
	}
	if _, ok := parts[4].(StreamTextEnd); !ok {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestPhase3ToolInputAccumulationAndValidation(t *testing.T) {
	prefix := event("message_start", `{"message":{"id":"m","model":"c"}}`)
	sse := prefix + event("content_block_start", `{"index":0,"content_block":{"type":"tool_use","id":"call","name":"weather"}}`) +
		event("content_block_delta", `{"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`) +
		event("content_block_delta", `{"index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}`) + event("content_block_stop", `{"index":0}`) + event("message_stop", `{}`)
	parts := parsePhase3(t, sse)
	call := parts[6].(StreamToolCall)
	if string(call.Input) != `{"city":"SF"}` || call.ToolCallID != "call" {
		t.Fatalf("call = %#v", call)
	}

	parts = parsePhase3(t, prefix+event("content_block_start", `{"index":0,"content_block":{"type":"tool_use","id":"call","name":"weather","input":{}}}`)+event("content_block_delta", `{"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}}`)+event("content_block_stop", `{"index":0}`)+event("message_stop", `{}`))
	if got := string(parts[5].(StreamToolCall).Input); got != `{"city":"SF"}` {
		t.Fatalf("delta input = %s", got)
	}

	parts = parsePhase3(t, prefix+event("content_block_start", `{"index":0,"content_block":{"type":"tool_use","id":"call","name":"weather"}}`)+event("content_block_stop", `{"index":0}`)+event("message_stop", `{}`))
	if string(parts[4].(StreamToolCall).Input) != `{}` {
		t.Fatalf("parts = %#v", parts)
	}

	parts = parsePhase3(t, prefix+event("content_block_start", `{"index":0,"content_block":{"type":"tool_use","id":"call","name":"weather"}}`)+event("content_block_delta", `{"index":0,"delta":{"type":"input_json_delta","partial_json":"[]"}}`)+event("content_block_stop", `{"index":0}`))
	if _, ok := parts[len(parts)-1].(StreamError); !ok {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestPhase3EagerBlocksRequireOneStopWithoutLifecyclePart(t *testing.T) {
	prefix := event("message_start", `{"message":{"id":"m","model":"c"}}`)
	for _, block := range []string{
		`{"type":"source","id":"src","url":"https://example.com"}`,
		`{"type":"tool_result","tool_use_id":"call","content":[{"type":"text","text":"ok"}]}`,
	} {
		parts := parsePhase3(t, prefix+event("content_block_start", `{"index":0,"content_block":`+block+`}`)+event("content_block_stop", `{"index":0}`)+event("message_stop", `{}`))
		if len(parts) != 4 {
			t.Fatalf("parts = %#v", parts)
		}
		if _, ok := parts[3].(StreamFinish); !ok {
			t.Fatalf("terminal part = %#v", parts[3])
		}

		parts = parsePhase3(t, prefix+event("content_block_start", `{"index":0,"content_block":`+block+`}`)+event("content_block_stop", `{"index":0}`)+event("content_block_stop", `{"index":0}`))
		if _, ok := parts[len(parts)-1].(StreamError); !ok {
			t.Fatalf("duplicate stop parts = %#v", parts)
		}
	}
}

func TestPhase3MessageStopFinishesWithoutEOF(t *testing.T) {
	reader, writer := io.Pipe()
	out := make(chan StreamPart, 8)
	done := make(chan struct{})
	go func() {
		parseSSEStream(context.Background(), reader, out, nil)
		close(out)
		close(done)
	}()
	_, _ = io.WriteString(writer, event("message_start", `{"message":{"id":"m","model":"c"}}`)+event("message_stop", `{}`))
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("parser waited for EOF after message_stop")
	}
	_ = writer.Close()
	var parts []StreamPart
	for part := range out {
		parts = append(parts, part)
	}
	if len(parts) != 3 {
		t.Fatalf("parts = %#v", parts)
	}
	if _, ok := parts[2].(StreamFinish); !ok {
		t.Fatalf("terminal part = %#v", parts[2])
	}
}

func TestPhase3SSESyntaxVariantsLargeAndFinalRecord(t *testing.T) {
	large := strings.Repeat("x", 128*1024)
	response := &StreamResponse{}
	out := make(chan StreamPart, 8)
	sse := ": a comment\r\nevent:message_start\r\ndata:{\"message\":{\"id\":\"m\",\"model\":\"claude\"}}\r\n\r\n" +
		"event: content_block_start\r\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\r\n\r\n" +
		"event: content_block_delta\r\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"" + large + "\"}}\r\n\r\n" +
		"event: content_block_stop\r\ndata: {\"index\":0}\r\n\r\n" +
		"event: message_stop\ndata: {}"
	parseSSEStream(context.Background(), strings.NewReader(sse), out, response)
	close(out)
	var parts []StreamPart
	for part := range out {
		parts = append(parts, part)
	}
	if len(parts) != 6 || len(parts[3].(StreamTextDelta).Text) != len(large) || response.ID != "m" || response.ModelID != "claude" {
		t.Fatalf("parts = %#v response = %#v", parts, response)
	}
	if _, ok := parts[5].(StreamFinish); !ok {
		t.Fatalf("terminal part = %#v", parts[5])
	}
}

func TestPhase3BlockLifecycleAndDeltaCompatibility(t *testing.T) {
	prefix := event("message_start", `{"message":{"id":"m","model":"c"}}`)
	sse := prefix +
		event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) + event("content_block_delta", `{"index":0,"delta":{"type":"text_delta","text":"text"}}`) + event("content_block_stop", `{"index":0}`) +
		event("content_block_start", `{"index":1,"content_block":{"type":"thinking"}}`) + event("content_block_delta", `{"index":1,"delta":{"type":"thinking_delta","thinking":"thought"}}`) + event("content_block_delta", `{"index":1,"delta":{"type":"signature_delta","signature":"sig"}}`) + event("content_block_stop", `{"index":1}`) +
		event("content_block_start", `{"index":2,"content_block":{"type":"source","id":"source"}}`) + event("content_block_stop", `{"index":2}`) +
		event("content_block_start", `{"index":3,"content_block":{"type":"tool_result","tool_use_id":"call"}}`) + event("content_block_stop", `{"index":3}`) + event("message_stop", `{}`)
	parts := parsePhase3(t, sse)
	if len(parts) != 12 || parts[3].(StreamTextDelta).Text != "text" || parts[6].(StreamReasoningDelta).Delta.(ThinkingDelta).Thinking != "thought" || parts[7].(StreamReasoningDelta).Delta.(SignatureDelta).Signature != "sig" {
		t.Fatalf("parts = %#v", parts)
	}
	if _, ok := parts[9].(StreamSource); !ok {
		t.Fatalf("source part = %#v", parts[9])
	}
	if _, ok := parts[10].(StreamToolResult); !ok {
		t.Fatalf("result part = %#v", parts[10])
	}
	if _, ok := parts[11].(StreamFinish); !ok {
		t.Fatalf("finish part = %#v", parts[11])
	}

	for _, tc := range []struct{ name, sse string }{
		{"duplicate index", prefix + event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) + event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`)},
		{"missing start index", prefix + event("content_block_start", `{"content_block":{"type":"text"}}`)},
		{"negative start", prefix + event("content_block_start", `{"index":-1,"content_block":{"type":"text"}}`)},
		{"missing delta index", prefix + event("content_block_delta", `{"delta":{"type":"text_delta"}}`)},
		{"negative stop", prefix + event("content_block_stop", `{"index":-1}`)},
		{"inactive stop", prefix + event("content_block_stop", `{"index":0}`)},
		{"text for thinking", prefix + event("content_block_start", `{"index":0,"content_block":{"type":"thinking"}}`) + event("content_block_delta", `{"index":0,"delta":{"type":"text_delta"}}`)},
		{"thinking for text", prefix + event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) + event("content_block_delta", `{"index":0,"delta":{"type":"thinking_delta"}}`)},
		{"input for text", prefix + event("content_block_start", `{"index":0,"content_block":{"type":"text"}}`) + event("content_block_delta", `{"index":0,"delta":{"type":"input_json_delta"}}`)},
		{"mcp delta for tool", prefix + event("content_block_start", `{"index":0,"content_block":{"type":"tool_use"}}`) + event("content_block_delta", `{"index":0,"delta":{"type":"mcp_tool_use_delta"}}`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parts := parsePhase3(t, tc.sse)
			if _, ok := parts[len(parts)-1].(StreamError); !ok {
				t.Fatalf("parts = %#v", parts)
			}
		})
	}
}

func TestPhase3ToolInitialInputAndStreamErrors(t *testing.T) {
	prefix := event("message_start", `{"message":{"id":"m","model":"c"}}`)
	parts := parsePhase3(t, prefix+event("content_block_start", `{"index":0,"content_block":{"type":"tool_use","id":"call","name":"weather","input":{"city":"SF"}}}`)+event("content_block_stop", `{"index":0}`)+event("message_stop", `{}`))
	if got := string(parts[4].(StreamToolCall).Input); got != `{"city":"SF"}` {
		t.Fatalf("initial tool input = %s", got)
	}
	for _, payload := range []string{
		`{"error":{"type":"overloaded_error","message":"busy"}}`,
		`{"error":{"type":"api_error"}}`,
		`not json`,
	} {
		parts := parsePhase3(t, prefix+event("error", payload))
		if _, ok := parts[len(parts)-1].(StreamError); !ok {
			t.Fatalf("payload %q parts = %#v", payload, parts)
		}
	}
	parts = parsePhase3(t, prefix+event("content_block_start", `{"index":0,"content_block":{"type":"tool_use"}}`)+event("content_block_delta", `{"index":0,"delta":{"type":"input_json_delta","partial_json":"{"}}`)+event("content_block_stop", `{"index":0}`))
	if _, ok := parts[len(parts)-1].(StreamError); !ok {
		t.Fatalf("malformed tool input parts = %#v", parts)
	}
}
