module github.com/shepard-labs/go-ai-sdk

go 1.25.8

require (
	github.com/jackc/pgx/v5 v5.10.0
	github.com/shepard-labs/go-ai-sdk/anthropic v1.0.6
	github.com/shepard-labs/go-ai-sdk/cohere v1.0.5
	github.com/shepard-labs/go-ai-sdk/google v1.0.6
	github.com/shepard-labs/go-ai-sdk/openai v1.0.5
	github.com/shepard-labs/go-ai-sdk/openaicompatible v1.0.6
	github.com/shepard-labs/go-ai-sdk/openrouter v1.0.6
	github.com/shepard-labs/go-clients/storage v1.0.2
)

replace (
	github.com/shepard-labs/go-ai-sdk/anthropic => ./anthropic
	github.com/shepard-labs/go-ai-sdk/cohere => ./cohere
	github.com/shepard-labs/go-ai-sdk/google => ./google
	github.com/shepard-labs/go-ai-sdk/openai => ./openai
	github.com/shepard-labs/go-ai-sdk/openaicompatible => ./openaicompatible
	github.com/shepard-labs/go-ai-sdk/openrouter => ./openrouter
)

require (
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	golang.org/x/sync v0.21.0 // indirect
	golang.org/x/text v0.38.0 // indirect
)
