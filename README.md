# go-ai-sdk

Go provider SDK for Anthropic models, including text generation, streaming, tools, thinking, structured output, cache control, MCP, context management, citations, and typed provider-tool results.

Module path:

```bash
github.com/shepard-labs/go-ai-sdk
```

Package path:

```go
github.com/shepard-labs/go-ai-sdk/anthropic
```

## Status

This repository implements the five foundation phases in `specs/`:

- Phase 1: provider, auth, model capabilities, core types
- Phase 2: text generation, text streaming, prompt conversion, request/response parsing
- Phase 3: function tools, provider tools, tool choice, tool streaming
- Phase 4: thinking, structured output, cache control, MCP/context request support
- Phase 5: citations, typed provider tool results/errors, structured logging, polish

## Features

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
go get github.com/shepard-labs/go-ai-sdk
```

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
anthropic/
├── anthropic.go
├── model.go
├── types.go
├── options.go
├── errors.go
├── usage.go
├── stop-reason.go
├── cache-control.go
├── schema-sanitize.go
├── prompt-convert.go
├── tools/
│   ├── prepare.go
│   └── container-id.go
└── internal/
    ├── api-types.go
    ├── api-schema.go
    └── streaming.go
```

## Notes

- Integration tests in this repository use mocked HTTP clients by default.
- Real Anthropic calls require a valid API key and network access.
- Provider tools automatically add the relevant `anthropic-beta` headers during request building.
