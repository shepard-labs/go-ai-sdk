# go-ai-sdk

[![Go Reference](https://pkg.go.dev/badge/github.com/shepard-labs/go-ai-sdk.svg)](https://pkg.go.dev/github.com/shepard-labs/go-ai-sdk)
[![Go Report Card](https://goreportcard.com/badge/github.com/shepard-labs/go-ai-sdk)](https://goreportcard.com/report/github.com/shepard-labs/go-ai-sdk)
[![License](https://img.shields.io/github/license/shepard-labs/go-ai-sdk)](https://github.com/shepard-labs/go-ai-sdk/blob/main/LICENSE)

Go provider SDK with first-class packages per vendor. Use the `llm` package tree for a provider-neutral `Client`, streaming, multi-turn tool agent loops, failover, caching, typed schemas, durable stores, and local toolkits—or import a provider package directly for full API access. The Anthropic package includes text generation, streaming, tools, thinking, structured output, cache control, MCP, context management, citations, and typed provider-tool results. The OpenRouter package includes chat, completion, embeddings, image generation, video generation, provider routing, BYOK headers, and OpenRouter usage metadata. The Cohere package includes chat generation, chat streaming, embeddings, reranking, citations, thinking, and Cohere RAG documents.

Root module path:

```bash
github.com/shepard-labs/go-ai-sdk
```

Provider modules:

```go
github.com/shepard-labs/go-ai-sdk/anthropic
github.com/shepard-labs/go-ai-sdk/cohere
github.com/shepard-labs/go-ai-sdk/google
github.com/shepard-labs/go-ai-sdk/openai
github.com/shepard-labs/go-ai-sdk/openaicompatible
github.com/shepard-labs/go-ai-sdk/openrouter
```

Provider-neutral `llm` packages:

```go
github.com/shepard-labs/go-ai-sdk/llm                    // Client, messages, tools, agent loops, failover, cache
github.com/shepard-labs/go-ai-sdk/llm/adapters           // direct adapter constructors
github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic // registry blank import for "anthropic:<model>"
github.com/shepard-labs/go-ai-sdk/llm/adapters/cohere    // available, not in the production-neutral matrix yet
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
github.com/shepard-labs/go-ai-sdk/llm/toolkit            // file and shell tools for agent loops
```

## llm

The `llm` package sits above provider subpackages. It defines a small `Client` interface (`Generate` and `Stream`), normalized messages and tools, optional `WithFailover` and `WithCache` decorators, and `AgentLoopWithOptions` for multi-turn tool use with terminal tools, validation policies, token budgets, and usage tracking. Implement `ToolDispatcher` to run tools (any registry that exposes `Dispatch(ctx, name, input)` works).

Provider adapters can be used directly from `llm/adapters`, or registered by blank-importing one of the adapter registration packages and calling `registry.NewClient`.

### Production-Neutral Contract

The production-neutral `llm` contract targets Anthropic, OpenAI, OpenAI-compatible providers, Google, and OpenRouter. Cohere remains available as a native provider package and adapter, but it is not implemented in the production-neutral compatibility matrix yet.

The neutral request API includes messages, tools, max tokens, sampling controls (`Temperature`, `TopP`, `TopK`), stop sequences, seed, tool choice, response format, per-request `Reasoning`, request headers, request metadata, provider options, and an unsupported-feature policy. Explicit unsupported options default to strict errors; set `UnsupportedFeaturePolicyWarn` when you want documented fallback behavior plus warnings.

The neutral result API preserves content, structured finish reasons (`Unified` plus raw provider value), expanded usage, warnings, request metadata, response metadata, and provider metadata. Provider-specific escape hatches live under `ProviderOptions` keys such as `"anthropic"`, `"openai"`, `"google"`, or `"openrouter"`.

| Provider | Adapter | Generate | Stream | Tools | Tool Choice | Structured Output | Images | Metadata / Warnings | Conformance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Anthropic | ✅ | ✅ | ✅ | ✅ | ⚠️ `none` unsupported | ❌ neutral `ResponseFormat` not implemented | ✅ | ✅ | ✅ |
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI-compatible | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ endpoint-dependent | ✅ endpoint-dependent | ✅ | ✅ |
| Google | ✅ | ✅ | ✅ | ✅ | ⚠️ `none` unsupported | ✅ | ✅ | ✅ | ✅ |
| OpenRouter | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cohere | ❌ not implemented | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Streaming uses typed `StreamPart` events for text, reasoning, tool input deltas, warnings, metadata, finish, errors, and raw provider bytes where available. Use `CollectStream` to drain a stream into a `GenerateResult` while preserving warnings, metadata, usage, provider metadata, and partial content on errors.

Portable reasoning control lives on `llm.GenerateOptions.Reasoning`:

```go
result, err := client.Generate(ctx, llm.GenerateOptions{
    Messages: messages,
    Reasoning: &llm.ReasoningOptions{Effort: llm.ReasoningHigh},
})
```

`nil` reasoning preserves provider defaults. `ReasoningNone` explicitly disables reasoning where supported. Anthropic supports exact `BudgetTokens`; Google and OpenAI expose portable effort levels, while exact budgets remain provider-specific or unsupported.

The neutral error model includes `UnsupportedFeatureError`, `APIError`, and helpers such as `IsRateLimit`, `IsAuth`, `IsInvalidRequest`, `IsUnsupported`, and `RetryAfter`. Middleware includes retry, failover, and cache wrappers; cache stores `Generate` results only and forwards `Stream` calls uncached.

```go
package main

import (
    "context"
    "log"

    _ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
    "github.com/shepard-labs/go-ai-sdk/llm"
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
    for _, c := range result.Content {
        if t, ok := c.(llm.TextContent); ok {
            log.Println(t.Text)
        }
    }
}
```

## Features

- Provider-neutral `llm.Client`, agent tool loops, failover and response caching
- Provider-neutral streaming via `Client.Stream` and typed `StreamPart` events
- Runtime provider selection with `llm/registry` and blank-imported adapters
- Direct adapter constructors in `llm/adapters`
- Struct-derived JSON schemas with `llm/schema`
- Durable run storage with `llm/store` backends for memory, files, Postgres, GCS, and R2
- Local file and shell tools for agent loops with `llm/toolkit`
- Multimodal neutral message parts for text, reasoning, tool calls/results, and images

- Provider creation with API key or bearer token auth
- Environment fallback from `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`
- Default provider via `anthropic.Anthropic`
- Context-aware HTTP requests
- Custom HTTP client via `ProviderSettings.Fetch`
- Optional request ID generator via `ProviderSettings.GenerateID`
- Optional structured logging via `ProviderSettings.Logger`
- Text generation with `DoGenerate`
- SSE streaming with `DoStream`
- Text, image, document, system, user, and assistant prompt conversion
- Thinking, redacted thinking, compaction, source, and citation parsing
- Function tools and Anthropic provider tools
- Tool choice modes: `auto`, `required`, `none`
- Tool result parsing and tool streaming
- Extended thinking request modes: `enabled`, `adaptive`, `disabled`
- Structured output modes: `outputFormat`, `jsonTool`, `auto`
- JSON schema sanitization for structured output
- Cache control on supported prompt blocks
- MCP server and tool configuration request support
- Container and skills request support
- Context management request and response support
- Typed provider-tool result/error parsing
- Model capability lookup with `ModelCapabilitiesForID`

## Installation

```bash
go get github.com/shepard-labs/go-ai-sdk/llm
go get github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic
```

Use `llm/registry` plus a blank-imported adapter when you want provider names such as `"anthropic:claude-sonnet-4-6"`, `"openai:gpt-4o"`, or `"openrouter:openai/gpt-4o-mini"`. Use `llm/adapters` directly when you want constructor functions such as `adapters.NewAnthropicClient`.

For provider-only use without the agent layer:

```bash
go get github.com/shepard-labs/go-ai-sdk/anthropic
```

Add other provider paths as needed (for example `.../openai`, `.../openrouter`, `.../cohere`). Each provider package is also its own module, so it can be consumed independently of the root module.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/shepard-labs/go-ai-sdk/anthropic"
)

func main() {
    provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{})
    if err := provider.Err(); err != nil {
        log.Fatal(err)
    }

    model := provider.Model("claude-3-haiku-20240307")
    result, err := model.DoGenerate(context.Background(), anthropic.GenerateOptions{
        MaxTokens: 256,
        Messages: []anthropic.Message{
            anthropic.UserMessage{Content: []anthropic.UserContent{
                anthropic.TextContent{Text: "Write a haiku about Go."},
            }},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, content := range result.Content {
        if text, ok := content.(anthropic.TextContent); ok {
            fmt.Println(text.Text)
        }
    }
}
```

## OpenAI-Compatible Providers

Use the `openaicompatible` package for APIs shaped like OpenAI's API. An OpenAI-Compatible Provider requires both `BaseURL` and `Name`; `APIKey` is sent as an `Authorization: Bearer` header.

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

func main() {
    provider := openaicompatible.CreateOpenAICompatible(openaicompatible.ProviderSettings{
        BaseURL: "https://api.example.com/v1",
        Name:    "example",
        APIKey:  os.Getenv("EXAMPLE_API_KEY"),
    })
    if err := provider.Err(); err != nil {
        log.Fatal(err)
    }

    model := provider.Chat("gpt-4o")
    result, err := model.DoGenerate(context.Background(), openaicompatible.GenerateOptions{
        Messages: []openaicompatible.Message{
            openaicompatible.UserMessage{Content: []openaicompatible.UserContent{
                openaicompatible.TextContent{Text: "Write a haiku about Go."},
            }},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, content := range result.Content {
        if text, ok := content.(openaicompatible.TextContent); ok {
            fmt.Println(text.Text)
        }
    }
}
```

The same provider can create chat, completion, embedding, and image model families with `Chat`, `Completion`, `Embedding`, and `Image`.

## OpenRouter

Use the `openrouter` package for OpenRouter's native API. `CreateOpenRouter` defaults to compatible mode, while the package-level `openrouter.OpenRouter` provider uses strict mode. `APIKey` falls back to `OPENROUTER_API_KEY` at request time.

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/shepard-labs/go-ai-sdk/openrouter"
)

func main() {
    provider := openrouter.CreateOpenRouter(openrouter.ProviderSettings{
        APIKey:  os.Getenv("OPENROUTER_API_KEY"),
        AppName: "Example App",
        AppURL:  "https://example.com",
    })

    model := provider.Chat("openai/gpt-4o-mini")
    result, err := model.DoGenerate(context.Background(), openrouter.GenerateOptions{
        Messages: []openrouter.Message{
            openrouter.UserMessage{Content: []openrouter.UserContent{
                openrouter.TextContent{Text: "Write a haiku about Go."},
            }},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, content := range result.Content {
        if text, ok := content.(openrouter.TextContent); ok {
            fmt.Println(text.Text)
        }
    }
}
```

The same OpenRouter provider can create chat, completion, embedding, image, and video model families with `Chat`, `Completion`, `Embedding`, `Image`, and `VideoModel`. `Model` routes `openai/gpt-3.5-turbo-instruct` to completions and other model IDs to chat.

## Cohere

Use the `cohere` package for Cohere's native v2 API. `CreateCohere` defaults to `https://api.cohere.com/v2`, sends `Authorization: Bearer <key>`, and falls back to `COHERE_API_KEY` when `ProviderSettings.APIKey` is empty.

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/shepard-labs/go-ai-sdk/cohere"
)

func main() {
    provider := cohere.CreateCohere(cohere.ProviderSettings{
        APIKey: os.Getenv("COHERE_API_KEY"),
    })
    if err := provider.Err(); err != nil {
        log.Fatal(err)
    }

    model := provider.Model("command-a-03-2025")
    result, err := model.DoGenerate(context.Background(), cohere.GenerateOptions{
        MaxOutputTokens: intPtr(256),
        Messages: []cohere.Message{
            cohere.UserMessage{Content: []cohere.UserContent{
                cohere.TextContent{Text: "Write a haiku about Go."},
            }},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, content := range result.Content {
        switch part := content.(type) {
        case cohere.TextContent:
            fmt.Println(part.Text)
        case cohere.SourceContent:
            fmt.Println("source:", part.Title)
        }
    }
}

func intPtr(v int) *int { return &v }
```

The same Cohere provider can create chat, embedding, and reranking model families with `Model`/`LanguageModel`, `Embedding`, and `Reranking`. The embedding provider string is `cohere.textEmbedding`; reranking uses `cohere.reranking`.

Cohere streaming emits typed stream parts:

```go
stream, err := provider.Model("command-a-03-2025").DoStream(ctx, cohere.StreamOptions{
    GenerateOptions: cohere.GenerateOptions{
        Messages: []cohere.Message{
            cohere.UserMessage{Content: []cohere.UserContent{
                cohere.TextContent{Text: "Stream a short answer about Go."},
            }},
        },
    },
})
if err != nil {
    log.Fatal(err)
}

for part := range stream.Stream {
    switch p := part.(type) {
    case cohere.StreamTextDelta:
        fmt.Print(p.Text)
    case cohere.StreamReasoningDelta:
        fmt.Print(p.Text)
    case cohere.StreamToolCall:
        fmt.Println("tool call:", p.ToolName, string(p.Input))
    case cohere.StreamError:
        log.Println(p.Err)
    }
}
```

Cohere embeddings use `/embed` with float embeddings and default `input_type` of `search_query`:

```go
embeddings, err := provider.Embedding("embed-english-v3.0").DoEmbed(ctx, cohere.EmbedOptions{
    Values: []string{"What is Go?", "Go is a programming language."},
    ProviderOptions: cohere.ProviderOptions{
        "cohere": {
            "inputType": "search_document",
            "truncate": "END",
        },
    },
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(len(embeddings.Embeddings), embeddings.Usage.Tokens)
```

Cohere reranking supports text documents and object documents. Object documents are JSON-stringified and return a compatibility warning.

```go
ranking, err := provider.Reranking("rerank-v3.5").DoRerank(ctx, cohere.RerankOptions{
    Query:     "best Go web framework",
    Documents: cohere.TextDocuments("net/http", "gin", "chi"),
    TopN:      intPtr(2),
})
if err != nil {
    log.Fatal(err)
}

for _, item := range ranking.Ranking {
    fmt.Println(item.Index, item.RelevanceScore)
}
```

## Authentication

Explicit API key:

```go
provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
    APIKey: "sk-ant-...",
})
```

Explicit bearer token:

```go
provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
    AuthToken: "token",
})
```

Environment fallback:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The provider always includes:

- `anthropic-version: 2023-06-01`
- `User-Agent: ai-sdk-go/anthropic/0.1.0`

If both `APIKey` and `AuthToken` are set, `provider.Err()` returns `ErrMultipleAuthMethods`.

## Provider Settings

```go
provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
    BaseURL: "https://api.anthropic.com/v1",
    APIKey: "sk-ant-...",
    Headers: http.Header{"X-App": []string{"example"}},
    Fetch: http.DefaultClient,
    GenerateID: func() string { return "request-id" },
    Name: "anthropic",
    Logger: myLogger,
    Retry: &anthropic.RetryOptions{
        MaxRetries: 2,
        BaseDelay:  200 * time.Millisecond,
        MaxDelay:   2 * time.Second,
        Jitter:     true,
    },
    MaxResponseBodyBytes:  32 << 20,
    MaxErrorResponseBytes: 1 << 20,
})
```

`GenerateID` is sent as `x-request-id` when configured.

If `Fetch` is not provided, the provider uses an internal `http.Client` with a tuned transport, connection pooling, TLS handshake timeout, idle connection timeout, and response-header timeout. Request lifetimes should still be controlled with `context.Context` deadlines or cancellation.

Retry behavior defaults to two retries for network errors, HTTP 429, and HTTP 5xx responses. Retries use exponential backoff, jitter, and honor `Retry-After` when present. Disable retries with:

```go
provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
    Retry: &anthropic.RetryOptions{MaxRetries: 0},
})
```

Successful non-streaming responses are limited by `MaxResponseBodyBytes` and error responses by `MaxErrorResponseBytes`. Defaults are 32 MiB for successful responses and 1 MiB for error responses. If a response exceeds the configured limit, `DoGenerate` returns an `*anthropic.APICallError` with `Truncated: true` and the retained bytes in `Body`.

API errors expose structured metadata via `*anthropic.APICallError`, including `Status`, `Type`, `Message`, `Retryable`, `Headers`, `RequestID`, `Body`, `Truncated`, and `Cause`.

## Streaming

```go
stream, err := model.DoStream(context.Background(), anthropic.StreamOptions{
    MaxTokens: 256,
    Messages: []anthropic.Message{
        anthropic.UserMessage{Content: []anthropic.UserContent{
            anthropic.TextContent{Text: "Stream a short story."},
        }},
    },
})
if err != nil {
    log.Fatal(err)
}

for part := range stream.Stream {
    switch p := part.(type) {
    case anthropic.StreamTextDelta:
        fmt.Print(p.Text)
    case anthropic.StreamReasoningDelta:
        fmt.Printf("reasoning: %#v\n", p.Delta)
    case anthropic.StreamToolCall:
        fmt.Printf("tool call: %s\n", p.ToolName)
    case anthropic.StreamToolResult:
        fmt.Printf("tool result: %#v\n", p.Result)
    case anthropic.StreamError:
        log.Println(p.Err)
    }
}
```

## Images And Documents

```go
result, err := model.DoGenerate(ctx, anthropic.GenerateOptions{
    MaxTokens: 256,
    Messages: []anthropic.Message{
        anthropic.UserMessage{Content: []anthropic.UserContent{
            anthropic.TextContent{Text: "Describe this image."},
            anthropic.ImageContent{Source: anthropic.ImageSource{
                Type: "base64",
                MediaType: "image/png",
                Data: imageBase64,
            }},
        }},
    },
})
```

Document content uses `DocumentContent` and `DocumentSource` with the same source shape.

## Function Tools

```go
weather := anthropic.Tool{
    Name: "get_weather",
    Description: "Get weather for a city.",
    InputSchema: map[string]any{
        "type": "object",
        "properties": map[string]any{
            "city": map[string]any{"type": "string"},
        },
        "required": []any{"city"},
    },
}

result, err := model.DoGenerate(ctx, anthropic.GenerateOptions{
    MaxTokens: 256,
    Tools: []anthropic.Tool{weather},
    ToolChoice: &anthropic.ToolChoice{Type: "auto"},
    Messages: []anthropic.Message{
        anthropic.UserMessage{Content: []anthropic.UserContent{
            anthropic.TextContent{Text: "What is the weather in San Francisco?"},
        }},
    },
})
```

Tool calls are returned as `ToolCallContent`:

```go
for _, content := range result.Content {
    if call, ok := content.(anthropic.ToolCallContent); ok {
        fmt.Println(call.ToolName, string(call.Input))
    }
}
```

## Provider Tools

Provider tools live under `anthropic/tools`:

```go
import anthropictools "github.com/shepard-labs/go-ai-sdk/anthropic/tools"

tools := []anthropic.Tool{
    anthropictools.CodeExecution_20250522(),
    anthropictools.WebSearch_20250305(nil, nil, nil, nil),
}

result, err := model.DoGenerate(ctx, anthropic.GenerateOptions{
    MaxTokens: 512,
    Tools: tools,
    Messages: []anthropic.Message{
        anthropic.UserMessage{Content: []anthropic.UserContent{
            anthropic.TextContent{Text: "Search the web and summarize the result."},
        }},
    },
})
```

Available provider tools include:

- Bash: `Bash_20241022`, `Bash_20250124`
- Code execution: `CodeExecution_20250522`, `CodeExecution_20250825`, `CodeExecution_20260120`
- Computer: `Computer_20241022`, `Computer_20250124`, `Computer_20251124`
- Text editor: `TextEditor_20241022`, `TextEditor_20250124`, `TextEditor_20250429`, `TextEditor_20250728`
- Memory: `Memory_20250818`
- Web fetch: `WebFetch_20250910`, `WebFetch_20260209`
- Web search: `WebSearch_20250305`, `WebSearch_20260209`
- Tool search: `ToolSearchRegex_20251119`, `ToolSearchBm25_20251119`
- Advisor: `Advisor_20260301`

## Tool Results

Tool results are represented as `ToolResultContent` with typed parts:

```go
if result, ok := content.(anthropic.ToolResultContent); ok {
    for _, part := range result.Result {
        switch p := part.(type) {
        case anthropic.ToolResultText:
            fmt.Println(p.Text)
        case anthropic.WebSearchResult:
            fmt.Println(p.Title, p.URL)
        case anthropic.CodeExecutionResult:
            fmt.Println(p.Stdout, p.Stderr, p.ReturnCode)
        case anthropic.AdvisorError:
            fmt.Println(p.ErrorCode)
        }
    }
}
```

Typed provider result/error parts include web fetch/search, code execution, bash execution, tool search, and advisor variants.

## Extended Thinking

```go
model := provider.Model("claude-sonnet-4-5", anthropic.ModelOptions{
    Thinking: &anthropic.ThinkingConfig{
        Type: anthropic.ThinkingTypeEnabled,
        BudgetTokens: 2048,
    },
})
```

When thinking is enabled:

- `thinking` is included in the request
- thinking budget is added to `max_tokens`
- `temperature`, `topK`, and `topP` are ignored with warnings

Adaptive thinking:

```go
model := provider.Model("claude-sonnet-4-5", anthropic.ModelOptions{
    Thinking: &anthropic.ThinkingConfig{
        Type: anthropic.ThinkingTypeAdaptive,
        Display: "summarized",
    },
})
```

Disabled thinking:

```go
model := provider.Model("claude-sonnet-4-5", anthropic.ModelOptions{
    Thinking: &anthropic.ThinkingConfig{Type: anthropic.ThinkingTypeDisabled},
})
```

## Structured Output

Structured output supports three modes:

- `StructuredOutputModeOutputFormat`
- `StructuredOutputModeJSONTool`
- `StructuredOutputModeAuto`

Auto mode uses native output format when supported by the model and falls back to jsonTool otherwise.

```go
model := provider.Model("claude-sonnet-4-5", anthropic.ModelOptions{
    StructuredOutputMode: anthropic.StructuredOutputModeAuto,
})

result, err := model.DoGenerate(ctx, anthropic.GenerateOptions{
    MaxTokens: 256,
    StructuredOutput: &anthropic.StructuredOutput{
        Name: "answer",
        Schema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "summary": map[string]any{"type": "string"},
            },
            "required": []any{"summary"},
        },
    },
    Messages: []anthropic.Message{
        anthropic.UserMessage{Content: []anthropic.UserContent{
            anthropic.TextContent{Text: "Summarize Go in one sentence."},
        }},
    },
})
```

Schemas are sanitized with `SanitizeSchema` before request building.

## Cache Control

Use cache-controlled wrapper content types when a specific block should carry cache control:

```go
cache := &anthropic.CacheControl{Type: "ephemeral", TTL: "5m"}

messages := []anthropic.Message{
    anthropic.SystemMessage{Content: "You are concise.", CacheControl: cache},
    anthropic.UserMessage{Content: []anthropic.UserContent{
        anthropic.CacheControlledTextContent{
            Text: "This prompt can be cached.",
            CacheControl: cache,
        },
    }},
}
```

## MCP Servers

```go
model := provider.Model("claude-sonnet-4-5", anthropic.ModelOptions{
    MCPServers: []anthropic.MCPServer{
        {
            Type: "url",
            Name: "docs",
            URL: "https://mcp.example.com",
            AuthorizationToken: "token",
            ToolConfiguration: &anthropic.ToolConfiguration{
                Enabled: true,
                AllowedTools: []string{"search_docs"},
            },
        },
    },
})
```

## Containers And Skills

```go
model := provider.Model("claude-sonnet-4-5", anthropic.ModelOptions{
    Container: &anthropic.Container{
        ID: "container-id",
        Skills: []anthropic.Skill{
            {Type: "skill", SkillID: "my-skill", Version: "1"},
        },
    },
})
```

Container forwarding helper:

```go
opts, err := anthropictools.ForwardAnthropicContainerIdFromLastStep(steps)
```

## Context Management

```go
model := provider.Model("claude-sonnet-4-5", anthropic.ModelOptions{
    ContextManagement: &anthropic.ContextManagement{
        Edits: []anthropic.ContextManagementEdit{
            anthropic.ClearToolUsesEdit{
                Type: "clear_tool_uses",
                Trigger: anthropic.InputTokensTrigger{Value: 100000},
                Keep: anthropic.ToolUsesCount{Value: 2},
            },
            anthropic.CompactEdit{
                Type: "compact",
                Instructions: "Keep user goals and final decisions.",
            },
        },
    },
})
```

Context-management responses are available through `GenerateResult.MessageMetadata["contextManagement"]` as `ContextManagementResponse`.

## Citations

```go
for _, content := range result.Content {
    if text, ok := content.(anthropic.TextContent); ok {
        fmt.Println(text.Text)
        for _, citation := range text.Citations {
            fmt.Println(citation.CitedText, citation.URL, citation.Title)
        }
    }
}
```

Streaming citation deltas are emitted as `StreamTextDelta` with `Citations` populated.

## Logging

Logging is no-op by default. Configure a logger with:

```go
type Logger interface {
    Debug(msg string, keysAndValues ...any)
    Info(msg string, keysAndValues ...any)
    Warn(msg string, keysAndValues ...any)
    Error(msg string, keysAndValues ...any)
}
```

```go
provider := anthropic.CreateAnthropic(anthropic.ProviderSettings{
    APIKey: "sk-ant-...",
    Logger: logger,
})
```

## Errors

API errors are returned as `*anthropic.APICallError`:

```go
result, err := model.DoGenerate(ctx, opts)
if err != nil {
    var apiErr *anthropic.APICallError
    if errors.As(err, &apiErr) {
        fmt.Println(apiErr.Status, apiErr.Retryable, apiErr.Message)
    }
}
```

## Model Capabilities

```go
caps := anthropic.ModelCapabilitiesForID("claude-sonnet-4-5")
fmt.Println(caps.MaxOutputTokens, caps.StructuredOutput, caps.RejectsSampling)
```

## Testing

Run the full local verification suite:

```bash
gofmt -w anthropic
go build ./...
go test ./...
go vet ./...
```

Current suite coverage includes:

- provider creation and auth
- env fallback
- header generation
- model capabilities
- beta header selection
- prompt conversion
- request building
- temperature/thinking warnings
- response parsing
- usage conversion
- SSE parsing
- streaming text/reasoning/tool/citation events
- function/provider tools
- structured output modes
- schema sanitization
- cache control
- MCP/server/container/context management config
- provider tool result/error parsing
- logging
- network/API error handling

## Package Layout

```text
anthropic/             Anthropic native provider module
cohere/                Cohere native provider module
google/                Google Gemini native provider module
openai/                OpenAI native provider module
openaicompatible/      OpenAI-compatible provider module
openrouter/            OpenRouter native provider module

llm/                   Provider-neutral Client, messages, tools, agent loop
├── adapters/          Adapters from native provider models to llm.Client
│   ├── anthropic/     Blank-import registry registration
│   ├── cohere/
│   ├── google/
│   ├── openai/
│   ├── openaicompatible/
│   └── openrouter/
├── registry/          Provider-name registry and registry.NewClient
├── schema/            Go struct reflection to JSON-schema tool definitions
├── store/             Run persistence interfaces and codecs
│   ├── file/
│   ├── gcs/
│   ├── memory/
│   ├── postgres/
│   └── r2/
└── toolkit/           File and shell tools for llm.AgentLoop

examples/              Runnable examples by provider and llm capability
```

## Notes

- Integration tests in this repository use mocked HTTP clients by default.
- Real Anthropic calls require a valid API key and network access.
- Provider tools automatically add the relevant `anthropic-beta` headers during request building.
