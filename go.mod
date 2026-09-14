module github.com/shepard-labs/go-ai-sdk

go 1.25.8

require (
	github.com/jackc/pgx/v5 v5.11.0
	github.com/shepard-labs/go-ai-sdk/anthropic v1.0.9
	github.com/shepard-labs/go-ai-sdk/cohere v1.0.9
	github.com/shepard-labs/go-ai-sdk/google v1.0.9
	github.com/shepard-labs/go-ai-sdk/openai v1.0.9
	github.com/shepard-labs/go-ai-sdk/openaicompatible v1.0.9
	github.com/shepard-labs/go-ai-sdk/openrouter v1.0.9
	github.com/shepard-labs/go-clients/storage v1.0.3
)

replace (
	github.com/shepard-labs/go-ai-sdk/anthropic => ./anthropic
	github.com/shepard-labs/go-ai-sdk/cohere => ./cohere
	github.com/shepard-labs/go-ai-sdk/google => ./google
	github.com/shepard-labs/go-ai-sdk/openai => ./openai
	github.com/shepard-labs/go-ai-sdk/openaicompatible => ./openaicompatible
	github.com/shepard-labs/go-ai-sdk/openrouter => ./openrouter
	github.com/shepard-labs/go-clients/storage => ../go-clients/storage
)

require (
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/text v0.41.0 // indirect
)
