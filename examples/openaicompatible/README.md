# OpenAI-Compatible Provider Examples

This folder contains runnable examples for using the `openaicompatible` package.
Each example passes an API key directly through `openaicompatible.ProviderSettings`
so the initialization is visible in code.

Do not commit a real API key. Replace the placeholder value locally:

```go
const apiKey = "your-openai-api-key"
```

For production apps, prefer environment variables or a secret manager. The examples
target the OpenAI API (`https://api.openai.com/v1`), but any OpenAI-compatible
endpoint works — swap `BaseURL` and `Name` to point at a different provider.

## Examples

### Basic Generation

Creates a provider, selects a chat model, sends a user message, and prints the
text response plus token usage.

```bash
go run examples/openaicompatible/generate/main.go
```

### Streaming

Uses `DoStream` and prints `StreamTextDelta` chunks as they arrive, then prints
usage from the final `StreamFinish` event.

```bash
go run examples/openaicompatible/stream/main.go
```

### Request Parameters

Shows common generation controls: `MaxOutputTokens`, `Temperature`, `TopP`,
`Seed`, `FrequencyPenalty`, `PresencePenalty`, and stop sequences.

```bash
go run examples/openaicompatible/parameters/main.go
```

### System Prompt

Uses `SystemMessage` to give the model a role before the user message.

```bash
go run examples/openaicompatible/system-prompt/main.go
```

### Tool Calling

Defines a local `get_weather` tool, detects the model's tool request, executes
a simulated tool locally, and sends the result back via `ToolMessage`.

```bash
go run examples/openaicompatible/tools/main.go
```

### Structured Output

Requests a schema-shaped JSON response using `StructuredOutput`. Enables
`SupportsStructuredOutputs` in provider settings for strict schema enforcement.

```bash
go run examples/openaicompatible/structured-output/main.go
```

### Vision

Sends mixed text and image content using `FileContent` with a `*url.URL` as
`Data` and `MediaType: "image/jpeg"`, which maps to an `image_url` part.

```bash
go run examples/openaicompatible/vision/main.go
```

### Embeddings

Uses `EmbeddingModel` and `DoEmbed` to embed multiple texts. Passes typed
embedding options (dimensions) via `ProviderOptions`.

```bash
go run examples/openaicompatible/embedding/main.go
```

### Image Generation

Uses `ImageModel` and `DoGenerate` to generate an image with DALL-E. Prints the
returned image URL(s).

```bash
go run examples/openaicompatible/image-generation/main.go
```

### Legacy Text Completion

Uses `CompletionModel` to call the `/completions` endpoint. Shows how to pass
`CompletionOptions` (suffix) through `ProviderOptions`.

```bash
go run examples/openaicompatible/completion/main.go
```

### Provider Settings

Shows all provider-level settings: retry options, custom headers, query
parameters, response body limits, `IncludeUsage`, `TransformRequestBody`, and
a custom `Logger`.

```bash
go run examples/openaicompatible/settings/main.go
```
