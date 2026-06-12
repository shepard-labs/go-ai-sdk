package openai

import (
	"bytes"
	"context"
	"net/http"
	"testing"
)

// TestChatErrorReturnedOnBadRequest verifies that a 4xx response is
// surfaced as an API call error.
func TestChatErrorReturnedOnBadRequest(t *testing.T) {
	respBody := `{"error":{"message":"invalid api key","type":"invalid_request_error","code":"invalid_api_key"}}`
	f := &recordingFetcher{responses: []*http.Response{response(401, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	apiErr, ok := err.(*APICallError)
	if !ok {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Status != 401 {
		t.Errorf("status: %d", apiErr.Status)
	}
	if apiErr.Message == "" {
		t.Errorf("message empty")
	}
}

// TestChatErrorOnRateLimit verifies a 429 is surfaced as an API call error.
func TestChatErrorOnRateLimit(t *testing.T) {
	respBody := `{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit"}}`
	f := &recordingFetcher{responses: []*http.Response{response(429, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Chat("gpt-4o").DoGenerate(context.Background(), GenerateOptions{
		Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	apiErr, ok := err.(*APICallError)
	if !ok {
		t.Fatalf("expected *APICallError, got %T: %v", err, err)
	}
	if apiErr.Status != 429 {
		t.Errorf("status: %d", apiErr.Status)
	}
}

// TestDefaultRetryableStatus verifies default 5xx + 429 are retryable.
func TestDefaultRetryableStatus(t *testing.T) {
	cases := []struct {
		status int
		want   bool
	}{
		{200, false},
		{400, false},
		{401, false},
		{403, false},
		{404, false},
		{429, true},
		{500, true},
		{502, true},
		{503, true},
		{504, true},
	}
	for _, c := range cases {
		if got := defaultRetryableStatus(c.status); got != c.want {
			t.Errorf("status %d: got %v, want %v", c.status, got, c.want)
		}
	}
}

// TestRetryDelayExponentialBackoff verifies the delay grows with attempt
// and is capped at MaxDelay.
func TestRetryDelayExponentialBackoff(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	// Use jitter=off to make the test deterministic.
	p.retry.Jitter = false
	p.retry.BaseDelay = 1_000_000    // 1ms
	p.retry.MaxDelay = 1_000_000_000 // 1s
	d0 := p.retryDelay(0)
	d1 := p.retryDelay(1)
	d2 := p.retryDelay(2)
	if !(d0 < d1 && d1 < d2) {
		t.Errorf("expected increasing: %v %v %v", d0, d1, d2)
	}
	// Should be capped at MaxDelay.
	dLarge := p.retryDelay(50)
	if dLarge > p.retry.MaxDelay {
		t.Errorf("delay %v exceeds MaxDelay %v", dLarge, p.retry.MaxDelay)
	}
}

// TestRequestURLBuildsCorrectPath verifies the request URL is built from
// the base URL and the path.
func TestRequestURLBuildsCorrectPath(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://api.example.com/v1")
	got, err := p.requestURL("/chat/completions")
	if err != nil {
		t.Fatalf("requestURL: %v", err)
	}
	want := "https://api.example.com/v1/chat/completions"
	if got != want {
		t.Errorf("URL: %q, want %q", got, want)
	}
}

// TestHeadersForCallMergesPerCallOverDefaults verifies per-call headers
// override the default provider headers.
func TestHeadersForCallMergesPerCallOverDefaults(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	perCall := http.Header{
		"X-Per-Call": []string{"call-value"},
		"X-Provider": []string{"override-value"},
	}
	got := p.headersForCall(perCall)
	if got.Get("X-Per-Call") != "call-value" {
		t.Errorf("X-Per-Call: %q", got.Get("X-Per-Call"))
	}
	if got.Get("Authorization") == "" {
		t.Errorf("Authorization should be set by default")
	}
}

// TestHeadersSpreadOverridesAuth verifies the spec rule that the
// user-supplied spread ProviderSettings.Headers can override the auth
// header when explicitly provided.
func TestHeadersSpreadOverridesAuth(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	p.headers = http.Header{
		"Authorization": []string{"Bearer my-custom-key"},
		"X-Custom":      []string{"custom-value"},
	}
	got := p.headersForCall(nil)
	if got.Get("Authorization") != "Bearer my-custom-key" {
		t.Errorf("Authorization = %q, want %q", got.Get("Authorization"), "Bearer my-custom-key")
	}
	if got.Get("X-Custom") != "custom-value" {
		t.Errorf("X-Custom = %q, want %q", got.Get("X-Custom"), "custom-value")
	}
	if got.Get("User-Agent") == "" {
		t.Errorf("User-Agent should still be set")
	}
}

// TestReadLimitedReadsTruncated verifies the truncated flag is set when
// the body exceeds the limit.
func TestReadLimitedReadsTruncated(t *testing.T) {
	body := []byte("a longer body than the limit")
	data, truncated, err := readLimited(bytes.NewReader(body), 5)
	if err != nil {
		t.Fatalf("readLimited: %v", err)
	}
	if !truncated {
		t.Errorf("expected truncated=true")
	}
	if string(data) != "a lon" {
		t.Errorf("data: %q", string(data))
	}
}

// TestReadLimitedReadsFull verifies the truncated flag is false when the
// body fits within the limit.
func TestReadLimitedReadsFull(t *testing.T) {
	body := []byte("short")
	data, truncated, err := readLimited(bytes.NewReader(body), 100)
	if err != nil {
		t.Fatalf("readLimited: %v", err)
	}
	if truncated {
		t.Errorf("expected truncated=false")
	}
	if string(data) != "short" {
		t.Errorf("data: %q", string(data))
	}
}

// TestCompatHeadersOrgAndProjectOnlyWhenSet verifies that the
// OpenAI-Organization and OpenAI-Project headers are only emitted
// when the corresponding setting is non-empty (per spec).
func TestCompatHeadersOrgAndProjectOnlyWhenSet(t *testing.T) {
	p := newOpenAIForTest(&recordingFetcher{}, "https://example.test/v1")
	// With organization and project set.
	p.organization = "org-1"
	p.project = "proj-1"
	h := p.compatHeaders()
	if h.Get("OpenAI-Organization") != "org-1" {
		t.Errorf("org: %q", h.Get("OpenAI-Organization"))
	}
	if h.Get("OpenAI-Project") != "proj-1" {
		t.Errorf("project: %q", h.Get("OpenAI-Project"))
	}
	// With both empty: must be absent.
	p.organization = ""
	p.project = ""
	h2 := p.compatHeaders()
	if _, present := h2["Openai-Organization"]; present {
		t.Errorf("OpenAI-Organization should be absent when empty")
	}
	if _, present := h2["Openai-Project"]; present {
		t.Errorf("OpenAI-Project should be absent when empty")
	}
	// User-Agent is always set.
	if h2.Get("User-Agent") == "" {
		t.Errorf("User-Agent should always be set")
	}
}

// TestExecuteBufferedSendsOrgHeader verifies the OpenAI-Organization
// header actually appears on the wire when set.
func TestExecuteBufferedSendsOrgHeader(t *testing.T) {
	var capturedReq *http.Request
	capturingFetcher := &captureFetcher{
		response: response(200, `{}`),
		capture:  func(r *http.Request) { capturedReq = r },
	}
	p := newOpenAIForTest(capturingFetcher, "https://example.test/v1")
	p.organization = "my-org"
	_, _ = p.executeBuffered(context.Background(), http.MethodPost, "/chat/completions", []byte(`{}`), nil)
	if capturedReq == nil {
		t.Fatal("no request captured")
	}
	if got := capturedReq.Header.Get("OpenAI-Organization"); got != "my-org" {
		t.Errorf("OpenAI-Organization on wire: %q", got)
	}
}

type captureFetcher struct {
	response *http.Response
	capture  func(*http.Request)
}

func (c *captureFetcher) Do(req *http.Request) (*http.Response, error) {
	if c.capture != nil {
		c.capture(req)
	}
	return c.response, nil
}
