package google

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func newSSEHandler(t *testing.T, chunks []string) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	})
}

func collectStream(t *testing.T, parts <-chan StreamPart, timeout time.Duration) []StreamPart {
	t.Helper()
	out := []StreamPart{}
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	for {
		select {
		case p, ok := <-parts:
			if !ok {
				return out
			}
			out = append(out, p)
		case <-deadline.C:
			t.Fatalf("stream channel did not close within %v (collected %d parts)", timeout, len(out))
		}
	}
}

func TestDoStream_TextOnly(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"Hello"}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, err := lm.DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	parts := collectStream(t, res.Stream, 2*time.Second)
	if len(parts) < 4 {
		t.Fatalf("parts = %d, want >=4: %+v", len(parts), parts)
	}
	if _, ok := parts[0].(StreamStart); !ok {
		t.Errorf("parts[0] = %T, want StreamStart", parts[0])
	}
	if _, ok := parts[len(parts)-1].(StreamFinish); !ok {
		t.Errorf("last = %T, want StreamFinish", parts[len(parts)-1])
	}
	sawText := false
	for _, p := range parts {
		if _, ok := p.(StreamTextDelta); ok {
			sawText = true
		}
	}
	if !sawText {
		t.Errorf("no StreamTextDelta in %+v", parts)
	}
}

func TestDoStream_Reasoning(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"thinking...","thought":true},{"text":"answer"}]},"finishReason":"STOP"}]}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, err := lm.DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	parts := collectStream(t, res.Stream, 2*time.Second)
	sawReasoningStart := false
	sawReasoningDelta := false
	sawTextStart := false
	sawTextDelta := false
	for _, p := range parts {
		switch p.(type) {
		case StreamReasoningStart:
			sawReasoningStart = true
		case StreamReasoningDelta:
			sawReasoningDelta = true
		case StreamTextStart:
			sawTextStart = true
		case StreamTextDelta:
			sawTextDelta = true
		}
	}
	if !sawReasoningStart || !sawReasoningDelta {
		t.Errorf("missing reasoning events: %+v", parts)
	}
	if !sawTextStart || !sawTextDelta {
		t.Errorf("missing text events: %+v", parts)
	}
}

func TestDoStream_CodeExecution(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1"}}]},"finishReason":"STOP"}]}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, err := lm.DoStream(context.Background(), StreamOptions{
		GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	parts := collectStream(t, res.Stream, 2*time.Second)
	var toolCall *StreamToolCall
	var toolResult *StreamToolResult
	for _, p := range parts {
		if tc, ok := p.(StreamToolCall); ok {
			tc := tc
			toolCall = &tc
		}
		if tr, ok := p.(StreamToolResult); ok {
			tr := tr
			toolResult = &tr
		}
	}
	if toolCall == nil {
		t.Fatalf("no StreamToolCall: %+v", parts)
	}
	if toolCall.ToolCall.ToolName != "code_execution" {
		t.Errorf("ToolName = %q", toolCall.ToolCall.ToolName)
	}
	if !toolCall.ToolCall.ProviderExecuted {
		t.Errorf("expected ProviderExecuted true")
	}
	if toolResult == nil {
		t.Fatalf("no StreamToolResult: %+v", parts)
	}
	if toolResult.ToolResult.ToolCallID != toolCall.ToolCall.ToolCallID {
		t.Errorf("result id %q != call id %q", toolResult.ToolResult.ToolCallID, toolCall.ToolCall.ToolCallID)
	}
}

func TestDoStream_FunctionCall_StreamingChunk(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"f1","name":"get_weather","args":{},"partialArgs":[{"jsonPath":"city","stringValue":"San ","willContinue":true}]}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"f1","name":"get_weather","args":{},"partialArgs":[{"jsonPath":"city","stringValue":"Francisco"}]}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, err := lm.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}}})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	parts := collectStream(t, res.Stream, 2*time.Second)
	var deltas []string
	var toolCall *StreamToolCall
	for _, p := range parts {
		switch v := p.(type) {
		case StreamToolInputDelta:
			deltas = append(deltas, v.Delta)
		case StreamToolCall:
			tc := v
			toolCall = &tc
		}
	}
	if toolCall == nil {
		t.Fatalf("no StreamToolCall: %+v", parts)
	}
	if string(toolCall.ToolCall.Input) != `{"city":"San Francisco"}` {
		t.Errorf("Input = %s, want %q", toolCall.ToolCall.Input, `{"city":"San Francisco"}`)
	}
	if len(deltas) < 2 {
		t.Errorf("expected >=2 deltas, got %d: %+v", len(deltas), deltas)
	}
}

func TestDoStream_FunctionCall_Complete(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"f1","name":"sum","args":{"a":1,"b":2}}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, _ := lm.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}}})
	parts := collectStream(t, res.Stream, 2*time.Second)
	var sawStart, sawDelta, sawEnd, sawCall bool
	for _, p := range parts {
		switch p.(type) {
		case StreamToolInputStart:
			sawStart = true
		case StreamToolInputDelta:
			sawDelta = true
		case StreamToolInputEnd:
			sawEnd = true
		case StreamToolCall:
			sawCall = true
		}
	}
	if !sawStart || !sawDelta || !sawEnd || !sawCall {
		t.Errorf("missing events: %+v", parts)
	}
}

func TestDoStream_FunctionCall_NoArgs(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"f1","name":"ping"}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, _ := lm.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}}})
	parts := collectStream(t, res.Stream, 2*time.Second)
	var sawStart, sawEnd, sawCall bool
	var toolCall *StreamToolCall
	for _, p := range parts {
		switch v := p.(type) {
		case StreamToolInputStart:
			sawStart = true
		case StreamToolInputEnd:
			sawEnd = true
		case StreamToolCall:
			tc := v
			toolCall = &tc
			sawCall = true
		}
	}
	if !sawStart || !sawEnd || !sawCall {
		t.Errorf("missing events: %+v", parts)
	}
	if toolCall == nil {
		t.Fatalf("no StreamToolCall: %+v", parts)
	}
	if string(toolCall.ToolCall.Input) != "{}" {
		t.Errorf("Input = %s, want {}", toolCall.ToolCall.Input)
	}
}

func TestDoStream_ServerToolCall(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"toolCall":{"toolType":"google_search","id":"sc1","args":{"q":"weather"}}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"toolResponse":{"toolType":"google_search","id":"sc1","response":{"results":["sunny"]}}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, _ := lm.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}}})
	parts := collectStream(t, res.Stream, 2*time.Second)
	var call *StreamToolCall
	var result *StreamToolResult
	for _, p := range parts {
		if v, ok := p.(StreamToolCall); ok {
			c := v
			call = &c
		}
		if v, ok := p.(StreamToolResult); ok {
			r := v
			result = &r
		}
	}
	if call == nil {
		t.Fatalf("no tool call: %+v", parts)
	}
	if !call.ToolCall.Dynamic {
		t.Errorf("expected Dynamic=true")
	}
	if result == nil {
		t.Fatalf("no tool result: %+v", parts)
	}
	if result.ToolResult.ToolCallID != call.ToolCall.ToolCallID {
		t.Errorf("result id %q != call id %q", result.ToolResult.ToolCallID, call.ToolCall.ToolCallID)
	}
}

func TestDoStream_Sources(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"hi"}]},"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://a.example","title":"A"}},{"web":{"uri":"https://a.example","title":"A dup"}},{"image":{"sourceUri":"https://i.example/x.png","title":"I"}},{"retrievedContext":{"uri":"https://r.example/p.pdf","title":"R"}},{"maps":{"uri":"https://m.example","title":"M"}}]}}]}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}`,
	}
	p := newTestProvider(t, newSSEHandler(t, chunks))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, _ := lm.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}}})
	parts := collectStream(t, res.Stream, 2*time.Second)
	urls := map[string]int{}
	for _, p := range parts {
		if v, ok := p.(StreamSource); ok {
			urls[v.Source.URL]++
		}
	}
	if len(urls) != 4 {
		t.Errorf("unique sources = %d, want 4: %+v", len(urls), urls)
	}
	if urls["https://a.example"] != 1 {
		t.Errorf("dedup failed: %+v", urls)
	}
}

func TestDoStream_StreamRaw(t *testing.T) {
	raw := `{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`
	p := newTestProvider(t, newSSEHandler(t, []string{raw}))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, _ := lm.DoStream(context.Background(), StreamOptions{
		IncludeRawChunks: true,
		GenerateOptions:  GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}},
	})
	parts := collectStream(t, res.Stream, 2*time.Second)
	sawRaw := false
	for _, p := range parts {
		if v, ok := p.(StreamRaw); ok {
			sawRaw = true
			if !strings.Contains(string(v.Raw), "STOP") {
				t.Errorf("raw missing STOP: %q", v.Raw)
			}
			if v.Decoded == nil {
				t.Errorf("raw.Decoded is nil")
			}
		}
	}
	if !sawRaw {
		t.Errorf("no StreamRaw: %+v", parts)
	}
}

func TestDoStream_AbortDuringStream(t *testing.T) {
	p := newTestProvider(t, newSSEHandler(t, []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"hi"}]}}]}`,
	}))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	ctx, cancel := context.WithCancel(context.Background())
	res, _ := lm.DoStream(ctx, StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}}})
	cancel()
	parts := collectStream(t, res.Stream, 1*time.Second)
	_ = parts
}

func TestDoStream_MalformedSSE(t *testing.T) {
	p := newTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, "data: {not-json}\n\n")
		flusher, _ := w.(http.Flusher)
		if flusher != nil {
			flusher.Flush()
		}
		_, _ = io.WriteString(w, `data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	lm := &googleLanguageModel{provider: p, modelID: ModelGemini25Flash}
	res, _ := lm.DoStream(context.Background(), StreamOptions{GenerateOptions: GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "q"}}}}}})
	parts := collectStream(t, res.Stream, 2*time.Second)
	var finish *StreamFinish
	for _, p := range parts {
		if v, ok := p.(StreamFinish); ok {
			f := v
			finish = &f
		}
	}
	if finish == nil {
		t.Fatalf("no StreamFinish after malformed chunk: %+v", parts)
	}
}
