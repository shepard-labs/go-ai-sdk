package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

type phase7ReadCloser struct {
	data []byte
	err  error
	done bool
}

func (r *phase7ReadCloser) Read(p []byte) (int, error) {
	if r.done {
		return 0, r.err
	}
	r.done = true
	return copy(p, r.data), r.err
}

func (*phase7ReadCloser) Close() error { return nil }

func TestPhase7SuccessfulResponseReadErrors(t *testing.T) {
	readErr := errors.New("response read failed")
	for _, tt := range []struct {
		name          string
		body          []byte
		limit         int64
		tools         []Tool
		wantBody      string
		wantTruncated bool
		wantRetryable bool
	}{
		{name: "immediate failure is typed and replay safe", limit: 32, wantRetryable: true},
		{name: "partial response is captured without parsing", body: []byte(`{"type":"message"`), limit: 32, wantBody: `{"type":"message"`, wantRetryable: true},
		{name: "failure at capture limit is incomplete", body: []byte("abcd"), limit: 4, wantBody: "abcd", wantTruncated: true, wantRetryable: true},
		{name: "provider tool response ambiguity is not retryable", limit: 32, tools: []Tool{{ID: "anthropic.web_search_20250305"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			headers := http.Header{"X-Request-Id": []string{"phase7-id"}, "X-Diagnostic": []string{"before"}}
			calls := 0
			model := CreateAnthropic(ProviderSettings{
				MaxResponseBodyBytes: tt.limit,
				Retry:                &RetryOptions{MaxRetries: 2},
				Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
					calls++
					return &http.Response{StatusCode: http.StatusOK, Header: headers, Body: &phase7ReadCloser{data: tt.body, err: readErr}}, nil
				}},
			}).Model("claude-3-haiku")

			result, err := model.DoGenerate(context.Background(), GenerateOptions{Tools: tt.tools})
			if result != nil {
				t.Fatalf("result = %#v, want nil", result)
			}
			var apiErr *APICallError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %T %v, want *APICallError", err, err)
			}
			if apiErr.Status != http.StatusOK || apiErr.Message != "failed to read successful response body" || apiErr.Type != "" || apiErr.RequestID != "phase7-id" || string(apiErr.Body) != tt.wantBody || apiErr.Truncated != tt.wantTruncated || apiErr.Retryable != tt.wantRetryable || !errors.Is(apiErr, readErr) {
				t.Fatalf("APICallError = %#v", apiErr)
			}
			headers["X-Diagnostic"][0] = "after"
			headers["X-Diagnostic"] = append(headers["X-Diagnostic"], "later")
			if got := apiErr.Headers.Get("X-Diagnostic"); got != "before" {
				t.Fatalf("cloned header = %q, want before", got)
			}
			if got := apiErr.Headers.Values("X-Diagnostic"); len(got) != 1 {
				t.Fatalf("cloned diagnostic values = %#v, want one value", got)
			}
			if calls != 1 {
				t.Fatalf("transport calls = %d, want 1", calls)
			}
		})
	}
}

func TestPhase7NonSuccessReadErrorKeepsDiagnostics(t *testing.T) {
	readErr := io.ErrUnexpectedEOF
	model := CreateAnthropic(ProviderSettings{
		MaxErrorResponseBytes: 32,
		Retry:                 &RetryOptions{MaxRetries: 0},
		Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusBadGateway, Header: http.Header{"Request-Id": []string{"error-id"}}, Body: &phase7ReadCloser{data: []byte("upstream"), err: readErr}}, nil
		}},
	}).Model("claude-3-haiku")

	_, err := model.DoGenerate(context.Background(), GenerateOptions{})
	var apiErr *APICallError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadGateway || apiErr.RequestID != "error-id" || string(apiErr.Body) != "upstream" || !errors.Is(apiErr, readErr) {
		t.Fatalf("error = %#v", err)
	}
}

func TestPhase7StreamNonSuccessReadErrorKeepsDiagnostics(t *testing.T) {
	readErr := io.ErrUnexpectedEOF
	headers := http.Header{"X-Amzn-RequestId": []string{"stream-error-id"}, "X-Diagnostic": []string{"before"}}
	model := CreateAnthropic(ProviderSettings{
		MaxErrorResponseBytes: 32,
		Retry:                 &RetryOptions{MaxRetries: 0},
		Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusBadGateway, Header: headers, Body: &phase7ReadCloser{data: []byte("upstream"), err: readErr}}, nil
		}},
	}).Model("claude-3-haiku")

	result, err := model.DoStream(context.Background(), StreamOptions{})
	if result != nil {
		t.Fatalf("result = %#v, want nil", result)
	}
	var apiErr *APICallError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadGateway || apiErr.Message != "upstream" || apiErr.Type != "" || apiErr.RequestID != "stream-error-id" || string(apiErr.Body) != "upstream" || !errors.Is(apiErr, readErr) {
		t.Fatalf("error = %#v", err)
	}
	headers["X-Diagnostic"][0] = "after"
	if got := apiErr.Headers.Get("X-Diagnostic"); got != "before" {
		t.Fatalf("cloned header = %q, want before", got)
	}
}

func TestPhase7NilResponseBodyIsTypedForGenerateAndStream(t *testing.T) {
	for _, tt := range []struct {
		status        int
		wantRetryable bool
	}{
		{status: http.StatusContinue},
		{status: http.StatusOK, wantRetryable: true},
		{status: http.StatusFound},
		{status: http.StatusBadRequest},
		{status: http.StatusTooManyRequests, wantRetryable: true},
		{status: http.StatusInternalServerError, wantRetryable: true},
	} {
		for _, stream := range []bool{false, true} {
			name := http.StatusText(tt.status)
			if stream {
				name += " stream"
			} else {
				name += " generate"
			}
			t.Run(name, func(t *testing.T) {
				headers := http.Header{"X-Request-Id": []string{"nil-body-id"}, "X-Diagnostic": []string{"before"}}
				model := CreateAnthropic(ProviderSettings{
					Retry: &RetryOptions{MaxRetries: 0},
					Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
						return &http.Response{StatusCode: tt.status, Header: headers}, nil
					}},
				}).Model("claude-3-haiku")

				var err error
				if stream {
					_, err = model.DoStream(context.Background(), StreamOptions{})
				} else {
					_, err = model.DoGenerate(context.Background(), GenerateOptions{})
				}
				var apiErr *APICallError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error = %T %v, want *APICallError", err, err)
				}
				if apiErr.Status != tt.status || apiErr.RequestID != "nil-body-id" || apiErr.Message != "HTTP response has nil body" || apiErr.Retryable != tt.wantRetryable {
					t.Fatalf("APICallError = %#v", apiErr)
				}
				headers.Set("X-Diagnostic", "after")
				if got := apiErr.Headers.Get("X-Diagnostic"); got != "before" {
					t.Fatalf("cloned header = %q, want before", got)
				}
			})
		}
	}
}

func TestPhase7NilResponseBodyIsNeverRetryableWhenReplayUnsafe(t *testing.T) {
	for _, status := range []int{http.StatusContinue, http.StatusOK, http.StatusFound, http.StatusBadRequest, http.StatusTooManyRequests, http.StatusInternalServerError} {
		for _, stream := range []bool{false, true} {
			name := http.StatusText(status)
			if stream {
				name += " stream"
			} else {
				name += " generate"
			}
			t.Run(name, func(t *testing.T) {
				model := CreateAnthropic(ProviderSettings{
					Retry: &RetryOptions{MaxRetries: 0},
					Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
						return &http.Response{StatusCode: status}, nil
					}},
				}).Model("claude-3-haiku")

				tools := []Tool{{ID: "anthropic.web_search_20250305"}}
				var err error
				if stream {
					_, err = model.DoStream(context.Background(), StreamOptions{Tools: tools})
				} else {
					_, err = model.DoGenerate(context.Background(), GenerateOptions{Tools: tools})
				}
				var apiErr *APICallError
				if !errors.As(err, &apiErr) || apiErr.Retryable {
					t.Fatalf("error = %#v, want non-retryable APICallError", err)
				}
			})
		}
	}
}

func TestPhase7HTTPErrorRetryabilityRequiresReplaySafety(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusInternalServerError} {
		for _, stream := range []bool{false, true} {
			for _, tt := range []struct {
				name          string
				tools         []Tool
				wantRetryable bool
				wantCalls     int
			}{
				{name: "ordinary request", wantRetryable: true, wantCalls: 3},
				{name: "provider tool request", tools: []Tool{{ID: "anthropic.web_search_20250305"}}, wantRetryable: false, wantCalls: 1},
			} {
				name := http.StatusText(status) + " " + tt.name
				if stream {
					name += " stream"
				} else {
					name += " generate"
				}
				t.Run(name, func(t *testing.T) {
					calls := 0
					model := CreateAnthropic(ProviderSettings{
						Retry: &RetryOptions{MaxRetries: 2, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond},
						Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
							calls++
							return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"failed"}}`))}, nil
						}},
					}).Model("claude-3-haiku")

					var err error
					if stream {
						_, err = model.DoStream(context.Background(), StreamOptions{Tools: tt.tools})
					} else {
						_, err = model.DoGenerate(context.Background(), GenerateOptions{Tools: tt.tools})
					}
					var apiErr *APICallError
					if !errors.As(err, &apiErr) || apiErr.Retryable != tt.wantRetryable {
						t.Fatalf("error = %#v, want retryable %v", err, tt.wantRetryable)
					}
					if calls != tt.wantCalls {
						t.Fatalf("transport calls = %d, want %d", calls, tt.wantCalls)
					}
				})
			}
		}
	}
}

func TestPhase7NilResponseRetryabilityMatchesReplaySafety(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, tt := range []struct {
			name          string
			modelOptions  ModelOptions
			tools         []Tool
			wantRetryable bool
		}{
			{name: "ordinary request", wantRetryable: true},
			{name: "provider tool request", tools: []Tool{{ID: "anthropic.web_search_20250305"}}},
			{name: "MCP request", modelOptions: ModelOptions{MCPServers: []MCPServer{{Name: "server"}}}},
		} {
			name := tt.name
			if stream {
				name += " stream"
			} else {
				name += " generate"
			}
			t.Run(name, func(t *testing.T) {
				calls := 0
				model := CreateAnthropic(ProviderSettings{
					Retry: &RetryOptions{MaxRetries: 2, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond},
					Fetch: roundTripFetcher{fn: func(*http.Request) (*http.Response, error) {
						calls++
						return nil, nil
					}},
				}).Model("claude-3-haiku", tt.modelOptions)

				var err error
				if stream {
					_, err = model.DoStream(context.Background(), StreamOptions{Tools: tt.tools})
				} else {
					_, err = model.DoGenerate(context.Background(), GenerateOptions{Tools: tt.tools})
				}
				var apiErr *APICallError
				if !errors.As(err, &apiErr) || apiErr.Message != "fetcher returned nil response and nil error" || apiErr.Retryable != tt.wantRetryable {
					t.Fatalf("error = %#v, want retryable %v", err, tt.wantRetryable)
				}
				if calls != 1 {
					t.Fatalf("transport calls = %d, want 1", calls)
				}
			})
		}
	}
}
