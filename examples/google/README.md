# Google Generative AI Provider Examples

This folder contains runnable examples for using the `google` package.
Each example passes an API key directly through `google.ProviderSettings`
so the initialization is visible in code.

Do not commit a real API key. Replace the placeholder value locally:

```go
const apiKey = "your-google-api-key"
```

For production apps, prefer environment variables or a secret manager. If you
omit `ProviderSettings.APIKey`, the provider falls back to
`GOOGLE_GENERATIVE_AI_API_KEY`.
Run commands from the repository root.

## Examples

### Basic Generation

Creates a provider, selects a Gemini model, sends a user message, and prints
the text response plus token usage.

```bash
go run ./examples/google/generate/main.go
```

### Provider Settings

Shows provider-level settings: direct API key, base URL, custom headers,
query params, retry options, and response body limits.

```bash
go run ./examples/google/settings/main.go
```

### Request Parameters

Shows common generation controls: `MaxOutputTokens`, `Temperature`, `TopP`,
`TopK`, `StopSequences`, `Seed`, and `FrequencyPenalty`.

```bash
go run ./examples/google/parameters/main.go
```

### System Prompt

Uses `google.SystemMessage` to give Gemini a role and behavior before the
user message.

```bash
go run ./examples/google/system-prompt/main.go
```

### Vision

Sends mixed text and image content using `google.ImageContent` with a URL
source.

```bash
go run ./examples/google/vision/main.go
```

### Streaming

Uses `DoStream` and prints `google.StreamTextDelta` chunks as they arrive.

```bash
go run ./examples/google/stream/main.go
```

### Structured Output

Requests a schema-shaped JSON response using `google.StructuredOutput`.

```bash
go run ./examples/google/structured-output/main.go
```

### Thinking

Enables extended thinking through `ProviderOptions["google"].thinkingConfig`,
requests reasoning content, and prints reasoning and final answer blocks
separately.

```bash
go run ./examples/google/thinking/main.go
```

### Tool Calling

Defines a local `get_weather` tool, detects Gemini's tool request, executes
a simulated tool locally, and sends the result back to Gemini.

```bash
go run ./examples/google/tools/main.go
```

### Grounding (Google Search)

Uses the `googleSearch` provider tool to ground Gemini's answer in live web
results, then prints the answer and the returned `groundingMetadata`.

```bash
go run ./examples/google/grounding/main.go
```

### Embedding

Embeds a batch of strings with the Gemini embedding model and prints the
resulting vector dimensions and a sample value.

```bash
go run ./examples/google/embedding/main.go
```

### Image Generation

Generates images with Imagen, decodes the base64 response, and writes PNG
files to disk.

```bash
go run ./examples/google/image-generation/main.go
```

### Speech (TTS)

Synthesizes speech from text with the Gemini TTS model and writes a WAV
file to disk.

```bash
go run ./examples/google/speech/main.go
```

### Video (Veo)

Generates a video with Veo. The call is long-running; the SDK polls the
operation and returns the final download URIs.

```bash
go run ./examples/google/video/main.go
```

### Files

Uploads a local file to the Google Files API and prints the resulting
provider reference. Pass the file path as the first argument.

```bash
go run ./examples/google/files/main.go /path/to/file.pdf
```
