# OpenAI Provider Examples

This folder contains runnable examples for the `openai` package. Each example
passes an API key through `openai.ProviderSettings` so initialization is
visible in code.

Do not commit a real API key. Replace the placeholder locally:

```go
const apiKey = "your-openai-api-key"
```

For production, prefer environment variables or a secret manager. If you omit
`ProviderSettings.APIKey`, the provider reads `OPENAI_API_KEY`. Optional
`OPENAI_BASE_URL` overrides the default `https://api.openai.com/v1`.

The `openai` provider exposes several model surfaces:

- **Chat** — `/chat/completions` with OpenAI-specific capabilities (reasoning
  models, file inputs, provider tools).
- **Responses** — `/responses` (default for `Model`, `LanguageModel`, and
  callable `provider("gpt-4o")`).
- **Completion** — legacy `/completions`.
- **Embedding**, **Image**, **Speech**, **Transcription** — dedicated endpoints.
- **Files**, **Skills** — upload APIs.
- **ExperimentalRealtime** — client secrets and WebSocket helpers (no built-in
  WebSocket client).

Provider-defined tools (`web_search`, `file_search`, `code_interpreter`, etc.)
are created via `provider.Tools()`. Standalone constructors also live in
`github.com/shepard-labs/go-ai-sdk/openai/tools`.

## Examples

### Basic Chat Generation

Creates a provider, uses `Chat("gpt-4o")`, sends a user message, and prints
text plus token usage.

```bash
go run examples/openai/generate/main.go
```

### Responses API

Uses `Responses("gpt-4o")` with `Instructions` and prints the response id and
finish reason.

```bash
go run examples/openai/responses/main.go
```

### Provider Settings

Shows `Organization`, `Project`, custom headers, retry options, response body
limits, and a custom `Logger`.

```bash
go run examples/openai/settings/main.go
```

### Request Parameters (Chat)

Demonstrates `MaxOutputTokens`, `Temperature`, `TopP`, `Seed`, stop sequences,
and frequency/presence penalties on the chat model.

```bash
go run examples/openai/parameters/main.go
```

### System Prompt

Uses `SystemMessage` before the user turn on the chat model.

```bash
go run examples/openai/system-prompt/main.go
```

### Vision (Chat)

Sends text plus `FileContent` with an image URL (`*url.URL` and
`MediaType: "image/jpeg"`).

```bash
go run examples/openai/vision/main.go
```

### Streaming (Chat)

Uses `DoStream` on the chat model and prints `StreamTextDelta` chunks and usage
from `StreamFinish`.

```bash
go run examples/openai/stream/main.go
```

### Streaming (Responses)

Streams from the Responses API, including optional `StreamReasoningDelta` on
reasoning models.

```bash
go run examples/openai/responses-stream/main.go
```

### Structured Output

Requests JSON matching a schema via `StructuredOutput` on the Responses API.

```bash
go run examples/openai/structured-output/main.go
```

### Reasoning

Sets `ReasoningConfig` (`effort`, `summary`) on a reasoning model and prints
`ReasoningContent` and final text separately.

```bash
go run examples/openai/reasoning/main.go
```

### Tool Calling (Responses)

Defines a `function` tool, handles `ToolCallContent`, returns
`ToolResultContent`, and completes the second turn.

```bash
go run examples/openai/tools/main.go
```

### Web Search (Provider Tool)

Attaches `provider.Tools().WebSearch(...)` so OpenAI executes search and may
return citations (`SourceContent`).

```bash
go run examples/openai/web-search/main.go
```

### Provider Tools (Combined)

Attaches several built-in tools (`web_search`, `code_interpreter`) in one
Responses call.

```bash
go run examples/openai/provider-tools/main.go
```

### Embeddings

Embeds multiple strings with `Embedding("text-embedding-3-small")` and optional
`dimensions` in `ProviderOptions["openai"]`.

```bash
go run examples/openai/embedding/main.go
```

### Image Generation

Generates with `Image("dall-e-3")`, prints URLs or writes decoded PNG files.

```bash
go run examples/openai/image-generation/main.go
```

### Speech (TTS)

Synthesizes speech with `Speech("tts-1")`, `VoiceNova`, and writes `speech.mp3`.

```bash
go run examples/openai/speech/main.go
```

### Transcription (STT)

Transcribes a local audio file with `Transcription("whisper-1")`.

```bash
go run examples/openai/transcription/main.go /path/to/audio.mp3
```

### Legacy Completion

Uses `Completion("gpt-3.5-turbo-instruct")` and passes `suffix` via
`ProviderOptions["openai"]`.

```bash
go run examples/openai/completion/main.go
```

### Files Upload

Uploads a file with `Files().UploadFile` and prints the `file-...` reference.

```bash
go run examples/openai/files/main.go /path/to/document.pdf
```

### Skills Upload

Uploads a skill zip with `Skills().UploadSkill`.

```bash
go run examples/openai/skills/main.go /path/to/skill.zip
```

### Realtime Helpers

Mints a client secret with `DoCreateClientSecret`, prints WebSocket URL and
protocols, and demonstrates `ParseServerEvent` / `SerializeClientEvent` (no live
WebSocket connection).

```bash
go run examples/openai/realtime/main.go
```