package anthropic

import (
	"context"
	"errors"
	"net/http"
	"testing"
)

func TestCitationsParsing(t *testing.T) {
	body := []byte(`{"content":[{"type":"text","text":"hello","citations":[{"type":"web_search_result_location","cited_text":"hello","url":"https://example.com","title":"Example","encrypted_index":"enc","document_index":1,"document_title":"Doc","start_page_number":2,"end_page_number":3,"start_char_index":4,"end_char_index":9}]}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	text := result.Content[0].(TextContent)
	if text.Citations[0].CitedText != "hello" || text.Citations[0].URL != "https://example.com" || text.Citations[0].DocumentIndex != 1 || text.Citations[0].StartCharIndex != 4 {
		t.Fatalf("text = %#v", text)
	}
}

func TestCitationsStreaming(t *testing.T) {
	sse := "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"citations_delta\",\"citations\":[{\"type\":\"char_location\",\"cited_text\":\"x\",\"start_char_index\":1,\"end_char_index\":2}]}}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	part := <-result.Stream
	delta := part.(StreamTextDelta)
	if delta.Citations[0].CitedText != "x" || delta.Citations[0].StartCharIndex != 1 {
		t.Fatalf("delta = %#v", delta)
	}
}

func TestProviderToolResultErrorParsing(t *testing.T) {
	body := []byte(`{"content":[{"type":"tool_result","tool_use_id":"call","content":[{"type":"web_fetch_error","error_code":"not_found"},{"type":"web_fetch_result","url":"https://example.com","retrieved_at":"now","media_type":"text/html","data":"doc"},{"type":"web_search_error","error_code":"blocked"},{"type":"web_search_result","url":"https://search","title":"Search","encrypted_content":"enc","page_age":"1d"},{"type":"code_execution_error","error_code":"timeout"},{"type":"code_execution_result","stdout":"out","stderr":"err","return_code":1,"content":[{"type":"text","text":"x"}]},{"type":"encrypted_code_execution_result","encrypted_stdout":"encout","stderr":"err","return_code":2,"content":[{"type":"text","text":"y"}]},{"type":"bash_code_execution_error","error_code":"bad"},{"type":"bash_code_execution_result","stdout":"bout","stderr":"berr","return_code":3,"content":[{"type":"text","text":"z"}]},{"type":"tool_search_error","error_code":"none"},{"type":"tool_search_result","tool_references":[{"name":"tool","description":"desc"}]},{"type":"advisor_error","error_code":"nope"},{"type":"advisor_result","text":"advice"},{"type":"advisor_redacted_result","encrypted_content":"secret"}]}]}`)
	result, err := parseGenerateResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	parts := result.Content[0].(ToolResultContent).Result
	checks := []bool{
		parts[0].(WebFetchError).ErrorCode == "not_found",
		parts[1].(WebFetchResult).URL == "https://example.com",
		parts[2].(WebSearchError).ErrorCode == "blocked",
		parts[3].(WebSearchResult).EncryptedContent == "enc",
		parts[4].(CodeExecutionError).ErrorCode == "timeout",
		parts[5].(CodeExecutionResult).ReturnCode == 1,
		parts[6].(EncryptedCodeExecutionResult).EncryptedStdout == "encout",
		parts[7].(BashCodeExecutionError).ErrorCode == "bad",
		parts[8].(BashCodeExecutionResult).Stdout == "bout",
		parts[9].(ToolSearchError).ErrorCode == "none",
		parts[10].(ToolSearchResult).ToolReferences[0].Name == "tool",
		parts[11].(AdvisorError).ErrorCode == "nope",
		parts[12].(AdvisorResult).Text == "advice",
		parts[13].(AdvisorRedactedResult).EncryptedContent == "secret",
	}
	for i, ok := range checks {
		if !ok {
			t.Fatalf("check %d failed: %#v", i, parts[i])
		}
	}
}

func TestToolResultErrorStreaming(t *testing.T) {
	sse := "event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"tool_result\",\"tool_use_id\":\"call\",\"content\":[{\"type\":\"advisor_error\",\"error_code\":\"bad\"}]}}\n\n"
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return textResponse(200, sse), nil }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoStream(context.Background(), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	part := <-result.Stream
	if part.(StreamToolResult).Result[0].(AdvisorError).ErrorCode != "bad" {
		t.Fatalf("part = %#v", part)
	}
}

type captureLogger struct{ calls []string }

func (l *captureLogger) Debug(msg string, _ ...any) { l.calls = append(l.calls, "debug:"+msg) }
func (l *captureLogger) Info(msg string, _ ...any)  { l.calls = append(l.calls, "info:"+msg) }
func (l *captureLogger) Warn(msg string, _ ...any)  { l.calls = append(l.calls, "warn:"+msg) }
func (l *captureLogger) Error(msg string, _ ...any) { l.calls = append(l.calls, "error:"+msg) }

func TestNoOpLogger(t *testing.T) {
	logger := noopLogger{}
	logger.Debug("debug", "k", "v")
	logger.Info("info")
	logger.Warn("warn")
	logger.Error("error")
}

func TestStructuredLogger(t *testing.T) {
	logger := &captureLogger{}
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		return jsonResponse(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher, Logger: logger}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(logger.calls) < 2 || logger.calls[0] != "debug:anthropic generate request" {
		t.Fatalf("calls = %#v", logger.calls)
	}
}

func TestDoGenerateNetworkError(t *testing.T) {
	want := errors.New("network")
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) { return nil, want }}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	if !errors.Is(err, want) {
		t.Fatalf("err = %v", err)
	}
}

func TestDoGenerateWithAllContentTypes(t *testing.T) {
	fetcher := roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
		return jsonResponse(200, `{"content":[{"type":"text","text":"ok","citations":[{"cited_text":"ok"}]},{"type":"thinking","thinking":"think"},{"type":"compaction","text":"compact"},{"type":"source","url":"https://example.com"},{"type":"tool_use","id":"call","name":"tool","input":{"x":1}},{"type":"tool_result","tool_use_id":"call","content":[{"type":"advisor_result","text":"advice"}]}]}`), nil
	}}
	model := CreateAnthropic(ProviderSettings{APIKey: "key", BaseURL: "https://example.test", Fetch: fetcher}).Model("claude-3-haiku-20240307")
	result, err := model.DoGenerate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 6 || result.Content[0].(TextContent).Citations[0].CitedText != "ok" || result.Content[5].(ToolResultContent).Result[0].(AdvisorResult).Text != "advice" {
		t.Fatalf("result = %#v", result)
	}
}
