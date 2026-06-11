package google

import (
	"net/http"
	"net/http/httptest"
	"net/url"
)

// mockFetcher is a Fetcher that routes requests to a test server handler.
// Defined here so it is shared across all _test.go files in this package.
type mockFetcher struct {
	server *httptest.Server
	signal <-chan struct{}
}

// newMockFetcher creates a test server that routes requests to handler.
// signal, if non-nil, is checked (non-blocking) before each handler call;
// if already closed, the handler is skipped and the request hangs until the
// test's context is cancelled — useful for abort/timeout tests.
func newMockFetcher(handler func(http.ResponseWriter, *http.Request)) *mockFetcher {
	return newMockFetcherWithSignal(handler, nil)
}

func newMockFetcherWithSignal(handler func(http.ResponseWriter, *http.Request), signal <-chan struct{}) *mockFetcher {
	m := &mockFetcher{signal: signal}
	m.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if signal != nil {
			select {
			case <-signal:
				// Context cancelled; let the handler block so the client sees ctx cancellation.
				handler(w, r)
			default:
				// Signal not yet closed; block until it is, then respond.
				<-signal
				handler(w, r)
			}
		} else {
			handler(w, r)
		}
	}))
	return m
}

func (f *mockFetcher) Do(req *http.Request) (*http.Response, error) {
	base, _ := url.Parse(f.server.URL)
	req.URL = base.ResolveReference(req.URL)
	return http.DefaultClient.Do(req)
}

func (f *mockFetcher) URL() string { return f.server.URL }

func (f *mockFetcher) Close() { f.server.Close() }