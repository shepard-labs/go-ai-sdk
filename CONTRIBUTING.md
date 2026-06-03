# Contributing

Thanks for contributing to `go-ai-sdk-anthropic`.

## Development Setup

Requirements:

- Go 1.26 or newer as used by the current development environment
- Standard Go tooling: `gofmt`, `go test`, `go vet`

Clone and verify:

```bash
go build ./...
go test ./...
go vet ./...
```

## Project Structure

- `anthropic/`: public provider package
- `anthropic/tools/`: provider tool constructors and helpers
- `anthropic/internal/`: internal API and streaming helpers
- `specs/`: phase specifications that drove the implementation

## Coding Guidelines

- Keep changes small and focused.
- Prefer explicit types over reflection-heavy abstractions.
- Preserve existing public APIs unless a spec or issue requires a breaking change.
- Add tests with every behavioral change.
- Use `context.Context` for request cancellation and timeouts.
- Keep real API calls out of unit tests; use mock `Fetcher` implementations.
- Do not log by default. Use the configured `Logger` only.
- Avoid global mutable state beyond the default `Anthropic` provider.

## Public API Changes

When adding or changing public types/functions:

- Update `README.md` examples if user-facing behavior changes.
- Add or update unit tests.
- Consider whether JSON tags are required for request/response DTO use.
- Keep exported names stable and idiomatic Go.

## Testing

Before submitting changes, run:

```bash
gofmt -w anthropic
go build ./...
go test ./...
go vet ./...
```

The test suite should cover:

- request building
- response parsing
- stream parsing
- error handling
- auth/header behavior
- beta header behavior
- any new public helper or provider-tool constructor

## Mocking HTTP

The provider accepts a custom fetcher:

```go
type Fetcher interface {
    Do(req *http.Request) (*http.Response, error)
}
```

Use this for tests instead of live network calls.

## Adding Provider Tools

When adding a new provider tool:

- Add a constructor in `anthropic/tools/prepare.go`.
- Add the normalized name to `ToolNameMapping`.
- Add beta header mapping in `betaHeadersForOptions` when required.
- Add tests for constructor fields and beta headers.
- Document the tool in `README.md`.

## Adding Response Types

When adding new response content or tool-result types:

- Add the public type in `types.go`.
- Implement the appropriate marker interface.
- Parse it in `model.go`.
- Add streaming support if Anthropic can emit it over SSE.
- Add tests for both non-streaming and streaming forms when applicable.

## Commit Hygiene

- Keep commits logically scoped.
- Do not commit generated artifacts, secrets, or local scratch files.
- Never include real API keys in tests or examples.

## Security

- Treat API keys and bearer tokens as secrets.
- Do not print auth headers in logs or test failures.
- Prefer mocked responses in examples and tests.
