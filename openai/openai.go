package openai

import (
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	defaultName    = "openai"
	userAgent      = "ai-sdk-go/openai/" + Version
)

// Version is the package version used in the User-Agent header.
const Version = "0.1.0"

// Fetcher executes HTTP requests.
type Fetcher = openaicompatible.Fetcher

// IDGenerator generates request IDs.
type IDGenerator = openaicompatible.IDGenerator

// RetryOptions configures retry behavior.
type RetryOptions = openaicompatible.RetryOptions

// Logger receives operational request logs.
type Logger = openaicompatible.Logger

// ProviderSettings configures an OpenAI provider.
type ProviderSettings struct {
	BaseURL                  string
	Name                     string
	APIKey                   string
	Organization             string
	Project                  string
	Headers                  http.Header
	Fetch                    Fetcher
	GenerateID               IDGenerator
	Logger                   Logger
	Retry                    *RetryOptions
	MaxResponseBodyBytes     int64
	MaxErrorResponseBytes    int64
	FileIDPrefixes           []string
	PassThroughUnsupportedFiles bool
}

type openaiProvider struct {
	compat              openaicompatible.Provider
	apiKey              string
	organization        string
	project             string
	baseURL             string
	fetch               Fetcher
	generateID          IDGenerator
	logger              Logger
	retry               RetryOptions
	maxResponseBodyBytes int64
	maxErrorResponseBytes int64
	err                 error
	files               Files
	skills              Skills
	realtime            ExperimentalRealtimeFactory
	tools               openaiTools
	headers             http.Header
	fileIDPrefixes      []string
	passThroughUnsupportedFiles bool
}

// CreateOpenAI creates an OpenAI provider.
func CreateOpenAI(settings ProviderSettings) Provider {
	baseURL := strings.TrimRight(valueOrEnv(settings.BaseURL, "OPENAI_BASE_URL"), "/")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	apiKey := settings.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	name := settings.Name
	if name == "" {
		name = defaultName
	}
	headers := http.Header{}
	if apiKey != "" {
		headers.Set("Authorization", "Bearer "+apiKey)
	}
	if settings.Organization != "" {
		headers.Set("OpenAI-Organization", settings.Organization)
	}
	if settings.Project != "" {
		headers.Set("OpenAI-Project", settings.Project)
	}
	headers.Set("User-Agent", userAgent)
	for k, values := range settings.Headers {
		headers.Del(k)
		for _, v := range values {
			headers.Add(k, v)
		}
	}

	compatSettings := openaicompatible.ProviderSettings{
		BaseURL:               baseURL,
		Name:                  name,
		APIKey:                apiKey,
		Headers:               headers,
		Fetch:                 settings.Fetch,
		GenerateID:            settings.GenerateID,
		Logger:                settings.Logger,
		Retry:                 settings.Retry,
		MaxResponseBodyBytes:  settings.MaxResponseBodyBytes,
		MaxErrorResponseBytes: settings.MaxErrorResponseBytes,
		IncludeUsage:          true,
	}
	compat := openaicompatible.CreateOpenAICompatible(compatSettings)

	p := &openaiProvider{
		compat:               compat,
		apiKey:               apiKey,
		organization:         settings.Organization,
		project:              settings.Project,
		baseURL:              baseURL,
		fetch:                settings.Fetch,
		generateID:           settings.GenerateID,
		logger:               settings.Logger,
		retry:                defaultOpenAIRetry(settings.Retry),
		maxResponseBodyBytes: openaiDefaultIfZero(settings.MaxResponseBodyBytes, 32<<20),
		maxErrorResponseBytes: openaiDefaultIfZero(settings.MaxErrorResponseBytes, 1<<20),
		headers:              headers,
	}
	if p.logger == nil {
		p.logger = openaiNoopLogger{}
	}
	if p.fetch == nil {
		p.fetch = openaiDefaultHTTPClient()
	}
	if apiKey == "" {
		p.err = ErrMissingAPIKey
	}
	p.fileIDPrefixes = settings.FileIDPrefixes
	if p.fileIDPrefixes == nil {
		p.fileIDPrefixes = []string{"file-"}
	}
	p.passThroughUnsupportedFiles = settings.PassThroughUnsupportedFiles
	p.files = newFilesClient(p)
	p.skills = newSkillsClient(p)
	p.realtime = newRealtimeFactory(p)
	p.tools = openaiTools{}
	return p
}

// OpenAI returns a default OpenAI provider instance, configured from the
// OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
func OpenAI() Provider {
	return CreateOpenAI(ProviderSettings{})
}

func (p *openaiProvider) Err() error   { return p.err }
func (p *openaiProvider) Name() string { return p.compat.Name() }

// ErrMissingAPIKey is returned by [CreateOpenAI] when no API key is supplied
// either through [ProviderSettings.APIKey] or the OPENAI_API_KEY environment
// variable.
var ErrMissingAPIKey = errors.New("openai: API key is required (set ProviderSettings.APIKey or OPENAI_API_KEY)")

func valueOrDefault(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}

func valueOrEnv(value, envName string) string {
	if value != "" {
		return value
	}
	return os.Getenv(envName)
}
