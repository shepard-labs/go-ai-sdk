# Anthropic Provider Examples

This folder contains runnable examples for using the `anthropic` package.
Each example passes an API key directly through `anthropic.ProviderSettings`
so the initialization is visible in code.

Do not commit a real API key. Replace the placeholder value locally:

```go
const apiKey = "sk-ant-api03-your-api-key"
```

For production apps, prefer environment variables or a secret manager. If you
omit `ProviderSettings.APIKey`, the provider falls back to `ANTHROPIC_API_KEY`.
Run commands from the repository root.

## Examples

### Basic Generation

Creates a provider, selects a Claude model, sends a user message, and prints the
text response plus token usage.

```bash
go run ./examples/anthropic/generate/main.go
```

### Provider Settings

Shows provider-level settings: direct API key, base URL, custom headers, client
name, retry options, and response body limits.

```bash
go run ./examples/anthropic/settings/main.go
```

### Request Parameters

Shows common generation controls: `MaxTokens`, `Temperature`, `TopP`, `TopK`,
stop sequences, and metadata.

```bash
go run ./examples/anthropic/parameters/main.go
```

### System Prompt

Uses `anthropic.SystemMessage` to give Claude a role and behavior before the user
message.

```bash
go run ./examples/anthropic/system-prompt/main.go
```

### Vision

Sends mixed text and image content using `anthropic.ImageContent` with a URL
source.

```bash
go run ./examples/anthropic/vision/main.go
```

### Streaming

Uses `DoStream` and prints `anthropic.StreamTextDelta` chunks as they arrive.

```bash
go run ./examples/anthropic/stream/main.go
```

### Structured Output

Requests a schema-shaped JSON response using `anthropic.StructuredOutput`.

```bash
go run ./examples/anthropic/structured-output/main.go
```

### Thinking

Enables extended thinking through model options, requests reasoning content, and
prints reasoning and final answer blocks separately.

The provider-neutral `llm` layer also supports per-request reasoning through
`llm.GenerateOptions.Reasoning`; this native example shows the Anthropic package's
direct `ModelOptions.Thinking` API.

```bash
go run ./examples/anthropic/thinking/main.go
```

### Tool Calling

Defines a local `get_weather` tool, detects Claude's tool request, executes a
simulated tool locally, and sends the result back to Claude.

```bash
go run ./examples/anthropic/tools/main.go
```
