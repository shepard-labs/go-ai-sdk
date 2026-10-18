package typesafe

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	defaultBaseURL                    = "https://api.typesafe.ai/v1"
	defaultMaxResponseBodyBytes int64 = 32 << 20
	defaultMaxErrorBodyBytes    int64 = 1 << 20
)

// Fetcher executes an HTTP request. http.Client satisfies it.
type Fetcher interface {
	Do(req *http.Request) (*http.Response, error)
}

// ProviderSettings configures a TypeSafe AI provider.
type ProviderSettings struct {
	// BaseURL overrides the API base URL. Defaults to the
	// TYPESAFE_AI_BASE_URL environment variable, then
	// https://api.typesafe.ai/v1. Trailing slashes are trimmed.
	BaseURL string
	// APIKey overrides the TYPESAFE_AI_API_KEY environment variable. When
	// empty, the environment variable is read lazily per request.
	APIKey string
	// Headers are sent with every request and may override Authorization.
	Headers http.Header
	// Fetch executes HTTP requests. Defaults to a tuned http.Client.
	Fetch Fetcher
	// GenerateID generates image reference strings. Defaults to a
	// crypto-random hex ID.
	GenerateID func() string
	// MaxResponseBodyBytes caps buffered response bodies (default 32MiB).
	MaxResponseBodyBytes int64
	// MaxErrorResponseBytes caps buffered error bodies (default 1MiB).
	MaxErrorResponseBytes int64
}

// TypeSafeAI is the default provider, configured from the environment.
var TypeSafeAI Provider = CreateTypeSafeAI(ProviderSettings{})

// Provider creates TypeSafe AI models. Only decision models are supported.
type Provider interface {
	DecisionModel(modelID string) DecisionModel
	LanguageModel(modelID string) (any, error)
	EmbeddingModel(modelID string) (any, error)
	ImageModel(modelID string) (any, error)
	// Err reports ErrMissingAPIKey when no API key is configured.
	Err() error
}

type typeSafeProvider struct {
	baseURL                                     string
	explicitAPIKey                              string
	headers                                     http.Header
	fetch                                       Fetcher
	generateID                                  func() string
	maxResponseBodyBytes, maxErrorResponseBytes int64
}

// CreateTypeSafeAI creates a TypeSafe AI provider. Construction never
// panics; a missing API key is reported via Err and at call time.
func CreateTypeSafeAI(settings ProviderSettings) Provider {
	base := settings.BaseURL
	if base == "" {
		base = os.Getenv("TYPESAFE_AI_BASE_URL")
	}
	base = strings.TrimRight(base, "/")
	if base == "" {
		base = defaultBaseURL
	}
	p := &typeSafeProvider{
		baseURL:               base,
		explicitAPIKey:        settings.APIKey,
		headers:               cloneHeader(settings.Headers),
		fetch:                 settings.Fetch,
		generateID:            settings.GenerateID,
		maxResponseBodyBytes:  maxPositiveOrDefault(settings.MaxResponseBodyBytes, defaultMaxResponseBodyBytes),
		maxErrorResponseBytes: maxPositiveOrDefault(settings.MaxErrorResponseBytes, defaultMaxErrorBodyBytes),
	}
	if p.fetch == nil {
		p.fetch = defaultHTTPClient()
	}
	if p.generateID == nil {
		p.generateID = cryptoID
	}
	return p
}

func (p *typeSafeProvider) apiKey() string {
	if p.explicitAPIKey != "" {
		return p.explicitAPIKey
	}
	return os.Getenv("TYPESAFE_AI_API_KEY")
}

func (p *typeSafeProvider) DecisionModel(modelID string) DecisionModel {
	return &decisionModel{provider: p, modelID: modelID}
}

func (p *typeSafeProvider) LanguageModel(modelID string) (any, error) {
	return nil, &NoSuchModelError{ModelID: modelID, ModelType: "languageModel"}
}

func (p *typeSafeProvider) EmbeddingModel(modelID string) (any, error) {
	return nil, &NoSuchModelError{ModelID: modelID, ModelType: "embeddingModel"}
}

func (p *typeSafeProvider) ImageModel(modelID string) (any, error) {
	return nil, &NoSuchModelError{ModelID: modelID, ModelType: "imageModel"}
}

func (p *typeSafeProvider) Err() error {
	if p.apiKey() == "" {
		return ErrMissingAPIKey
	}
	return nil
}

func cryptoID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

func defaultHTTPClient() *http.Client {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns = 100
	tr.MaxIdleConnsPerHost = 20
	tr.IdleConnTimeout = 90 * time.Second
	tr.TLSHandshakeTimeout = 10 * time.Second
	tr.ExpectContinueTimeout = time.Second
	tr.ResponseHeaderTimeout = 2 * time.Minute
	return &http.Client{Transport: tr}
}

func cloneHeader(h http.Header) http.Header {
	out := http.Header{}
	for k, values := range h {
		for _, v := range values {
			out.Add(k, v)
		}
	}
	return out
}

func maxPositiveOrDefault(v, fallback int64) int64 {
	if v > 0 {
		return v
	}
	return fallback
}
