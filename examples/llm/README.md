# llm Package Examples

Runnable examples for the provider-neutral `llm` package and its subpackages
(`registry`, `schema`, `toolkit`, `store`). Each example makes a real
end-to-end LLM call and is focused on one capability.

Examples select a provider by name with `registry.NewClient("provider:model-id", ...)`
and pass the API key directly so initialization is visible in code. Do not
commit a real key — replace the placeholder locally:

```go
const apiKey = "sk-ant-api03-your-api-key"
```

Most examples use `anthropic:claude-sonnet-4-6`; `failover` also uses
`openai:gpt-4o`. A provider becomes available by blank-importing its adapter
package, e.g. `_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"`.
Run commands from the repository root.

The production-neutral example surface targets Anthropic, OpenAI,
OpenAI-compatible providers, Google, and OpenRouter. Cohere is available in the
repo, but it is not implemented in the production-neutral compatibility matrix
yet and should be treated as later work for this layer.

## Feature → example map

| Example | Demonstrates | Key API |
|---|---|---|
| `registry` | Selecting a provider by name at runtime | `registry.NewClient`, `registry.ProviderOptions` |
| `schema` | Struct → tool schema → typed structured output via a terminal tool | `schema.Tool`, `AgentLoopOptions.SubmitResultTool` |
| `toolkit` | Scoped file/shell tools driving a real agent loop that does on-disk work | `toolkit.Files`/`Shell`/`Tools`/`Merge`, `AgentLoop` |
| `store` | Durable multi-turn runs: persist a transcript, resume from a fresh load | `store.RunStore`, `store/file`, `Client.Generate` |
| `stream` | Token-by-token output; reasoning vs. answer deltas; finish/error contract | `Client.Stream`, the `StreamPart` union |
| `failover` | Retrying against a fallback provider on a retryable error | `llm.WithFailover`, `llm.FailoverConfig` |
| `cache` | Read-through response caching; identical calls skip the provider | `llm.WithCache`, `llm.CacheBackend` |
| `vision` | Multimodal input — text + image in one message | `llm.ImageContent`, `llm.ImageURLSource` / `ImageInlineSource` |
| `validation` | Cross-field input validation with model self-correction (repair) | `AgentLoopOptions.ToolPolicies`, `ToolPolicy.Validate`, `MaxToolRepairs` |
| `budget` | Bounding history sent per turn by trimming oldest tool pairs | `AgentLoopOptions.TokenBudget` / `TokenCounter` |

## Production-neutral provider matrix

| Provider | Registry prefix | Production-neutral status |
|---|---|---:|
| Anthropic | `anthropic:` | ✅ implemented |
| OpenAI | `openai:` | ✅ implemented |
| OpenAI-compatible | `openaicompatible:` | ✅ implemented |
| Google | `google:` | ✅ implemented |
| OpenRouter | `openrouter:` | ✅ implemented |
| Cohere | `cohere:` | ❌ not implemented |

For explicit unsupported options, the neutral layer returns an
`UnsupportedFeatureError` by default. Set `UnsupportedFeaturePolicyWarn` when an
application prefers a warning-backed fallback. Stream examples can be collected
into a `GenerateResult` with `llm.CollectStream` when you need the same content,
usage, warning, and metadata shape as non-streaming generation.

## Running

```bash
go run ./examples/llm/registry
go run ./examples/llm/schema
go run ./examples/llm/toolkit
go run ./examples/llm/store
go run ./examples/llm/stream
go run ./examples/llm/failover
go run ./examples/llm/cache
go run ./examples/llm/vision
go run ./examples/llm/validation
go run ./examples/llm/budget
```

## Not shown here

- **Other implemented production-neutral providers** (`google`, `openrouter`,
  `openaicompatible`): the selection mechanism is identical to `registry` —
  change the `"provider:model-id"` string and blank-import that adapter. For an
  OpenAI-compatible endpoint (Ollama, LM Studio), also pass
  `registry.ProviderOptions{BaseURL: ...}`.
- **Cohere**: native Cohere examples and packages exist elsewhere in the repo,
  but Cohere is not implemented in the production-neutral matrix yet.
- **Other store backends** (`postgres`, `gcs`, `r2`): drop-in replacements for
  the `file` backend in `store`, but they require external services to run.
