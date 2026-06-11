# OpenRouter Provider Examples

This folder contains runnable examples for using the `openrouter` package.
Each example passes an API key directly through `openrouter.ProviderSettings`
so the initialization is visible in code.

Do not commit a real API key. Replace the placeholder value locally:

```go
const apiKey = "your-openrouter-api-key"
```

For production apps, prefer loading the key from the `OPENROUTER_API_KEY`
environment variable — the provider reads it automatically when `APIKey` is
empty.

## Examples

### Basic Generation

Creates a provider, selects a chat model, sends a user message, and prints the
text response plus token usage.

```bash
go run examples/openrouter/generate/main.go
```

### Streaming

Uses `DoStream` and prints `StreamTextDelta` chunks as they arrive, then prints
usage from the final `StreamFinish` event.

```bash
go run examples/openrouter/stream/main.go
```

### Request Parameters

Shows common generation controls: `MaxTokens`, `Temperature`, `TopP`, `TopK`,
`Seed`, `FrequencyPenalty`, `PresencePenalty`, and stop sequences.

```bash
go run examples/openrouter/parameters/main.go
```

### System Prompt

Uses `SystemMessage` to give the model a role before the user message.

```bash
go run examples/openrouter/system-prompt/main.go
```

### Tool Calling

Defines a local `get_weather` tool, detects the model's tool request, executes
a simulated tool locally, and sends the result back via `ToolMessage`.

```bash
go run examples/openrouter/tools/main.go
```

### Web Search

Shows two ways to give models live web access: the `WebPlugin` plugin (passed
via `ChatOptions.Plugins`) and the `web_search` provider-defined tool returned
by `provider.Tools().WebSearch(...)`.

```bash
go run examples/openrouter/web-search/main.go
```

### Structured Output

Requests a schema-shaped JSON response using `ResponseFormat` with `type: "json"`
and a JSON schema. Enables `StructuredOutputs.Strict` in `ChatOptions` for strict
schema enforcement on supported models.

```bash
go run examples/openrouter/structured-output/main.go
```

### Vision

Sends mixed text and image content using `FileContent` with a string URL as
`Data` and `MediaType: "image/jpeg"`, which maps to an `image_url` part.

```bash
go run examples/openrouter/vision/main.go
```

### Embeddings

Uses `Embedding()` and `DoEmbed` to embed multiple texts and prints per-vector
dimension counts and usage.

```bash
go run examples/openrouter/embedding/main.go
```

### Image Generation

Uses `Image()` and `DoGenerate` to generate an image via OpenRouter's chat
completions image modality. Prints the returned base64 data prefix.

```bash
go run examples/openrouter/image-generation/main.go
```

### Legacy Text Completion

Uses `Completion()` to call the `/completions` endpoint. Shows `CompletionOptions`
(suffix) passed at model construction time.

```bash
go run examples/openrouter/completion/main.go
```

### Reasoning

Enables chain-of-thought reasoning via `ChatOptions.IncludeReasoning` and
`ChatOptions.Reasoning`. Prints `ReasoningContent` and `TextContent` separately,
and shows reasoning token counts from `Usage.OutputTokensDetails`.

```bash
go run examples/openrouter/reasoning/main.go
```

### Provider Settings

Shows all provider-level settings: retry options, custom headers, `AppName`,
`AppURL`, response body limits, and a custom `Logger`.

```bash
go run examples/openrouter/settings/main.go
```

### Provider Routing

Uses `ProviderRouting` in `ChatOptions` to control which upstream providers
OpenRouter may use: ordered preference list, fallback policy, data collection
opt-out, quantization filter, and cost-based sorting.

```bash
go run examples/openrouter/provider-routing/main.go
```
