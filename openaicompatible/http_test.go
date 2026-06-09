package openaicompatible

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

type recordingFetcher struct {
	mu        sync.Mutex
	requests  []*http.Request
	responses []*http.Response
	errors    []error
	calls     int
}

func (f *recordingFetcher) Do(req *http.Request) (*http.Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, cloneRequest(req))
	idx := f.calls
	f.calls++
	var err error
	if idx < len(f.errors) {
		err = f.errors[idx]
	}
	var resp *http.Response
	if idx < len(f.responses) {
		resp = f.responses[idx]
	}
	return resp, err
}

func cloneRequest(req *http.Request) *http.Request {
	clone := req.Clone(req.Context())
	clone.Header = req.Header.Clone()
	if req.Body != nil {
		body, _ := io.ReadAll(req.Body)
		_ = req.Body.Close()
		req.Body = io.NopCloser(strings.NewReader(string(body)))
		clone.Body = io.NopCloser(strings.NewReader(string(body)))
	}
	return clone
}

func response(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{"X-Request-Id": []string{"resp-id"}}, Body: io.NopCloser(strings.NewReader(body))}
}

func TestHTTPHeadersURLAndRequestID(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{}`)}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL:     "https://example.test/v1/",
		Name:        "acme",
		APIKey:      "secret",
		Headers:     http.Header{"Authorization": []string{"Bearer override"}, "X-Provider": []string{"provider"}},
		QueryParams: map[string]string{"a": "1", "b": "2"},
		Fetch:       f,
		GenerateID:  func() string { return "req-id" },
		Retry:       &RetryOptions{MaxRetries: 0},
	}).(*openAICompatibleProvider)
	perCall := http.Header{"Authorization": []string{"Bearer per-call"}, "X-Provider": []string{"call"}}
	_, err := p.executeJSON(context.Background(), endpointEmbeddings, []byte(`{}`), perCall)
	if err != nil {
		t.Fatalf("executeJSON error = %v", err)
	}
	req := f.requests[0]
	if req.URL.String() != "https://example.test/v1/embeddings?a=1&b=2" {
		t.Fatalf("URL = %q", req.URL.String())
	}
	if got := req.Header.Get("Authorization"); got != "Bearer per-call" {
		t.Fatalf("Authorization = %q", got)
	}
	if got := req.Header.Get("X-Provider"); got != "call" {
		t.Fatalf("X-Provider = %q", got)
	}
	if got := req.Header.Get("User-Agent"); got != "ai-sdk-go/openai-compatible/"+Version {
		t.Fatalf("User-Agent = %q", got)
	}
	if got := req.Header.Get("x-request-id"); got != "req-id" {
		t.Fatalf("x-request-id = %q", got)
	}
}

func TestHTTPProviderHeaderCanOverrideAuthorization(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", APIKey: "secret", Headers: http.Header{"Authorization": []string{"custom"}}, Fetch: f, Retry: &RetryOptions{MaxRetries: 0}}).(*openAICompatibleProvider)
	_, err := p.executeJSON(context.Background(), endpointChatCompletions, []byte(`{}`), nil)
	if err != nil {
		t.Fatalf("executeJSON error = %v", err)
	}
	if got := f.requests[0].Header.Get("Authorization"); got != "custom" {
		t.Fatalf("Authorization = %q", got)
	}
}

func TestHTTPDefaultClientTransportAndRetryDefaults(t *testing.T) {
	client := defaultHTTPClient()
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T", client.Transport)
	}
	if transport.MaxIdleConns != 100 || transport.MaxIdleConnsPerHost != 20 || transport.IdleConnTimeout != 90*time.Second || transport.TLSHandshakeTimeout != 10*time.Second || transport.ExpectContinueTimeout != time.Second || transport.ResponseHeaderTimeout != 2*time.Minute {
		t.Fatalf("transport defaults do not match Anthropic defaults: %+v", transport)
	}
	retry := defaultRetryOptions(nil)
	if retry.MaxRetries != 2 || retry.BaseDelay != 200*time.Millisecond || retry.MaxDelay != 2*time.Second || !retry.Jitter {
		t.Fatalf("retry defaults = %+v", retry)
	}
}

func TestHTTPRetryNetwork429And5xx(t *testing.T) {
	for name, tc := range map[string]struct {
		responses []*http.Response
		errors    []error
	}{
		"network": {responses: []*http.Response{nil, response(200, `{}`)}, errors: []error{errors.New("net"), nil}},
		"429":     {responses: []*http.Response{response(429, `{"error":{"message":"slow"}}`), response(200, `{}`)}},
		"5xx":     {responses: []*http.Response{response(500, `{"error":{"message":"down"}}`), response(200, `{}`)}},
	} {
		t.Run(name, func(t *testing.T) {
			f := &recordingFetcher{responses: tc.responses, errors: tc.errors}
			p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 1, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond}}).(*openAICompatibleProvider)
			_, err := p.executeJSON(context.Background(), endpointCompletions, []byte(`{}`), nil)
			if err != nil {
				t.Fatalf("executeJSON error = %v", err)
			}
			if f.calls != 2 {
				t.Fatalf("calls = %d, want 2", f.calls)
			}
		})
	}
}

func TestHTTPRetryAfterAndContextCancellation(t *testing.T) {
	headerDate := time.Now().Add(-time.Second).UTC().Format(http.TimeFormat)
	for _, value := range []string{"0", headerDate} {
		if _, ok := retryAfterDelay(value); !ok {
			t.Fatalf("Retry-After %q was not parsed", value)
		}
	}
	f := &recordingFetcher{responses: []*http.Response{response(500, `{"error":{"message":"down"}}`)}}
	f.responses[0].Header.Set("Retry-After", "10")
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 1, BaseDelay: time.Hour, MaxDelay: time.Hour}}).(*openAICompatibleProvider)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := p.executeJSON(ctx, endpointCompletions, []byte(`{}`), nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

func TestHTTPNilFetcherResponse(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{nil}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}}).(*openAICompatibleProvider)
	_, err := p.executeJSON(context.Background(), endpointChatCompletions, []byte(`{}`), nil)
	apiErr := new(APICallError)
	if !errors.As(err, &apiErr) || !apiErr.Retryable || apiErr.Message != "fetcher returned nil response and nil error" {
		t.Fatalf("error = %#v", err)
	}
}

func TestHTTPBodyLimits(t *testing.T) {
	t.Run("success truncated", func(t *testing.T) {
		f := &recordingFetcher{responses: []*http.Response{response(200, "abcd")}}
		p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, MaxResponseBodyBytes: 3}).(*openAICompatibleProvider)
		_, err := p.executeJSON(context.Background(), endpointChatCompletions, []byte(`{}`), nil)
		apiErr := new(APICallError)
		if !errors.As(err, &apiErr) || !apiErr.Truncated || string(apiErr.Body) != "abc" || apiErr.Retryable {
			t.Fatalf("error = %#v", err)
		}
	})
	t.Run("error truncated", func(t *testing.T) {
		f := &recordingFetcher{responses: []*http.Response{response(400, "abcdef")}}
		p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, MaxErrorResponseBytes: 3}).(*openAICompatibleProvider)
		_, err := p.executeJSON(context.Background(), endpointChatCompletions, []byte(`{}`), nil)
		apiErr := new(APICallError)
		if !errors.As(err, &apiErr) || !apiErr.Truncated || string(apiErr.Body) != "abc" {
			t.Fatalf("error = %#v", err)
		}
	})
}

func TestHTTPResponseHeadersAreCloned(t *testing.T) {
	headers := http.Header{"X-Test": []string{"before"}}
	f := &recordingFetcher{responses: []*http.Response{{StatusCode: 200, Header: headers, Body: io.NopCloser(strings.NewReader(`{}`))}}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}}).(*openAICompatibleProvider)
	resp, err := p.executeJSON(context.Background(), endpointChatCompletions, []byte(`{}`), nil)
	if err != nil {
		t.Fatalf("executeJSON error = %v", err)
	}
	headers.Set("X-Test", "after")
	if got := resp.Headers.Get("X-Test"); got != "before" {
		t.Fatalf("cloned header = %q", got)
	}
}

func TestRequestURLReplacesBaseURLQuery(t *testing.T) {
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test/v1?old=1", Name: "acme", QueryParams: map[string]string{"new": "2"}}).(*openAICompatibleProvider)
	got, err := p.requestURL(endpointImageGenerations)
	if err != nil {
		t.Fatal(err)
	}
	u, err := url.Parse(got)
	if err != nil {
		t.Fatal(err)
	}
	if u.Query().Get("old") != "" || u.Query().Get("new") != "2" {
		t.Fatalf("query = %q", u.RawQuery)
	}
}
