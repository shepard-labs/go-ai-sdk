# llm

Provider-neutral language model API for production application code. Use this
package when you want one contract for generation, streaming, tool loops,
metadata, warnings, retry, failover, caching, schemas, stores, and local
toolkits across supported providers.

Use the provider packages directly when an application needs full access to a
vendor API surface such as embeddings, image generation, speech, realtime,
file uploads, or provider-specific administration APIs.

## Stability and scope

The production-neutral `llm` layer targets chat/completion-style calls through
`Client.Generate` and `Client.Stream`.

In scope:

- Anthropic
- OpenAI
- OpenAI-compatible providers
- Google
- OpenRouter

Out of scope for the production-neutral matrix right now:

- Cohere
- provider-neutral embeddings
- image generation
- speech and transcription
- realtime APIs
- file upload APIs
- provider administration APIs

Provider-specific packages can still expose these features independently of
`llm`.

## Provider matrix

| Provider | Registry prefix | Generate | Stream | Tools | Tool Choice | Structured Output | Images | Metadata / Warnings | Conformance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anthropic | `anthropic:` | ✅ | ✅ | ✅ | ⚠️ `none` unsupported | ❌ neutral `ResponseFormat` not implemented | ✅ | ✅ | ✅ |
| OpenAI | `openai:` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI-compatible | `openaicompatible:` | ✅ | ✅ | ✅ | ✅ | ✅ endpoint-dependent | ✅ endpoint-dependent | ✅ | ✅ |
| Google | `google:` | ✅ | ✅ | ✅ | ⚠️ `none` unsupported | ✅ | ✅ | ✅ | ✅ |
| OpenRouter | `openrouter:` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cohere | `cohere:` | ❌ not implemented | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## Packages

```go
github.com/shepard-labs/go-ai-sdk/llm                    // Client, messages, tools, streams, middleware
github.com/shepard-labs/go-ai-sdk/llm/adapters           // direct adapter constructors
github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic // blank-import registry packages
github.com/shepard-labs/go-ai-sdk/llm/adapters/google
github.com/shepard-labs/go-ai-sdk/llm/adapters/openai
github.com/shepard-labs/go-ai-sdk/llm/adapters/openaicompatible
github.com/shepard-labs/go-ai-sdk/llm/adapters/openrouter
github.com/shepard-labs/go-ai-sdk/llm/registry           // provider-name registry and NewClient
github.com/shepard-labs/go-ai-sdk/llm/schema             // Go structs to JSON-schema tools
github.com/shepard-labs/go-ai-sdk/llm/store              // run persistence interfaces and codecs
github.com/shepard-labs/go-ai-sdk/llm/store/file
github.com/shepard-labs/go-ai-sdk/llm/store/gcs
github.com/shepard-labs/go-ai-sdk/llm/store/memory
github.com/shepard-labs/go-ai-sdk/llm/store/postgres
github.com/shepard-labs/go-ai-sdk/llm/store/r2
github.com/shepard-labs/go-ai-sdk/llm/toolkit            // file, shell, and git tools for agent loops
```

## Client contract

```go
type Client interface {
    Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error)
    Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error)
}
```

`GenerateOptions` is the portable request shape. It includes:

- system prompt and message history
- text, reasoning, tool-use, tool-result, and image content
- function tools
- max tokens
- `Temperature`, `TopP`, `TopK`, stop sequences, and seed
- `ToolChoice`
- `ResponseFormat`
- `Reasoning` for provider-neutral reasoning/thinking control
- request headers and metadata
- `ProviderOptions` for provider-specific escape hatches
- `UnsupportedFeaturePolicy`

Explicit unsupported features use strict errors by default. Set
`UnsupportedFeaturePolicyWarn` when an application wants the adapter to continue
with a documented fallback and return a warning instead.

`GenerateResult` preserves:

- normalized content
- structured finish reason with both `Unified` and raw provider values
- expanded token usage
- warnings
- request metadata
- response metadata
- provider metadata

## Reasoning

Use `GenerateOptions.Reasoning` for portable per-request reasoning or thinking
control:

```go
result, err := client.Generate(ctx, llm.GenerateOptions{
    Messages: []llm.Message{{
        Role:    "user",
        Content: []llm.Content{llm.TextContent{Text: "Solve this carefully."}},
    }},
    Reasoning: &llm.ReasoningOptions{
        Effort: llm.ReasoningHigh,
    },
})
```

Supported neutral efforts are `ReasoningNone`, `ReasoningMinimal`,
`ReasoningLow`, `ReasoningMedium`, `ReasoningHigh`, and `ReasoningXHigh`.
`Reasoning: nil` means "use provider defaults"; `ReasoningNone` explicitly
disables reasoning when the provider supports it.

`BudgetTokens` requests an exact reasoning budget where supported:

```go
budget := 4096
result, err := client.Generate(ctx, llm.GenerateOptions{
    Messages: messages,
    Reasoning: &llm.ReasoningOptions{
        BudgetTokens: &budget,
    },
})
```

Provider behavior:

- Anthropic maps neutral reasoning to per-request `ThinkingConfig`; exact budgets
  are supported, and Anthropic adds the thinking budget to `MaxTokens`.
- Google maps neutral effort to Gemini thinking; exact neutral budgets are not
  mapped by the neutral adapter, so use `ProviderOptions["google"].thinkingConfig`
  for provider-specific Gemini budgets.
- OpenAI maps neutral effort to `ProviderOptions["openai"].reasoningEffort`;
  exact budgets are unsupported.
- Generic OpenAI-compatible adapters do not assume reasoning support and return
  `UnsupportedFeatureError` unless warn policy is enabled.

If both `Reasoning` and provider-specific reasoning options are set, the neutral
`Reasoning` field wins because it is the explicit per-request instruction.

## Provider options

Common portable fields should be set directly on `GenerateOptions`. Put
provider-specific options under a provider key:

```go
result, err := client.Generate(ctx, llm.GenerateOptions{
    Messages: []llm.Message{{
        Role:    "user",
        Content: []llm.Content{llm.TextContent{Text: "Answer in one sentence."}},
    }},
    ProviderOptions: llm.ProviderOptions{
        "openrouter": {
            "provider": map[string]any{"sort": "throughput"},
        },
    },
})
```

Adapters either forward provider options, return `UnsupportedFeatureError`, or
return warnings depending on support and `UnsupportedFeaturePolicy`.

## Registry usage

```go
package main

import (
    "context"
    "log"

    "github.com/shepard-labs/go-ai-sdk/llm"
    _ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
    "github.com/shepard-labs/go-ai-sdk/llm/registry"
)

func main() {
    client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
        APIKey: "sk-ant-...",
    })
    if err != nil {
        log.Fatal(err)
    }

    result, err := client.Generate(context.Background(), llm.GenerateOptions{
        System:    "You are concise.",
        MaxTokens: 256,
        Messages: []llm.Message{{
            Role:    "user",
            Content: []llm.Content{llm.TextContent{Text: "Say hello in one sentence."}},
        }},
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, part := range result.Content {
        if text, ok := part.(llm.TextContent); ok {
            log.Println(text.Text)
        }
    }
}
```

## Streaming

`Client.Stream` returns typed `StreamPart` events. Streams emit text deltas,
reasoning deltas, tool input deltas, warnings, metadata, finish, errors, and raw
provider bytes where available.

Exactly one `StreamFinish` or `StreamError` is emitted before the channel closes
on normal adapter streams. Providers that do not implement streaming return
`ErrStreamNotImplemented` directly from `Stream`.

Use `CollectStream` when you want a streamed response reconstructed into the
same shape as `Generate`:

```go
parts, err := client.Stream(ctx, opts)
if err != nil {
    return err
}

result, err := llm.CollectStream(ctx, parts)
```

`CollectStream` preserves content, tool calls, warnings, metadata, finish
reason, usage, provider metadata, and partial content returned before a stream
error.

## Errors

The neutral error model includes:

- `UnsupportedFeatureError`
- `APIError`
- `IsRateLimit(err)`
- `IsAuth(err)`
- `IsInvalidRequest(err)`
- `IsUnsupported(err)`
- `RetryAfter(err)`

Adapters should preserve provider errors while exposing neutral classification
where possible.

## Middleware

`llm` includes composable client wrappers:

- `WithRetry` retries retryable errors with backoff and `Retry-After` support.
- `WithFailover` moves to fallback clients according to `FailoverConfig`.
- `WithCache` caches deterministic `Generate` results through `CacheBackend`.
- Observation hooks record generate, stream, retry, failover, and cache events.

`WithCache` does not cache streams. It forwards `Stream` calls to the wrapped
client.

## Agent loops and tools

`AgentLoopWithOptions` runs a multi-turn tool loop over any `Client` and
`ToolDispatcher`. It supports terminal submit-result tools, validation policies,
tool repair attempts, token budgets, and transcript usage tracking.

Use `llm/schema` to derive JSON-schema tools from Go structs. Use `llm/toolkit`
for scoped file, shell, and git tools in examples or internal agents. Production
apps should configure narrow roots and command allowlists for local tools.

## Persistence

`llm/store` defines durable run persistence. Backends are available for memory,
files, Postgres, GCS, and R2. Stores persist normalized transcripts and metadata
for resuming or inspecting agent runs.

## Examples

Runnable examples live in `examples/llm` and cover registry selection, schema
tools, toolkit agents, stores, streaming, reasoning, failover, cache, vision,
validation, and token budgets.
