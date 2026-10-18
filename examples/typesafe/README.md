# TypeSafe Provider Examples

Runnable examples for the `typesafe` package — TypeSafe AI's decision model
that answers structured questions (choice, score, boolean, box) against shared
state in a single request.

All examples read the API key from `TYPESAFE_AI_API_KEY`; `settings`, `headers`,
and the local-validation half of `errors` run without one. Run commands from
the repository root.

## Examples

| Example | Demonstrates |
|---|---|
| `decision` | Minimal provider + `DecisionModel` + `DoDecide` — choice and boolean questions, answers, confidence, rounding, usage |
| `question-types` | All question types in one call — `ChoiceQuestion`, `ScoreQuestion`, `BooleanQuestion` (and the `BoxQuestion` API shape) |
| `inputs` | The four `Input` constructors — `StringInput`, `JSONInput`, `RawInput`, `NullInput` — mixed across instructions and criteria |
| `state` | `TextStatePart` vs `JSONStatePart` — a single JSON part is sent raw, multiple parts join with `\n` |
| `images` | `FileStatePart` inline JPEG, auto-generated vs explicit `Reference`, multipart upload |
| `box` | `BoxQuestion` — normalized top/bottom/left/right coordinates |
| `long-context` | `ProviderOptions["typesafe"]["longContext"]` for oversized state |
| `provider-options` | `longContext` (only supported key), warnings for unknown keys, foreign-namespace isolation |
| `headers` | Provider-level `ProviderSettings.Headers` vs per-call `DecisionCallOptions.Headers` |
| `settings` | `ProviderSettings` — explicit vs env `APIKey`, `BaseURL`, `Headers`, body caps, `Err()` pre-flight |
| `errors` | `InvalidArgumentError`, `UnsupportedFunctionalityError`, `APICallError`, `ErrMissingAPIKey` |
| `adapter` | Provider-neutral `llm/decision` API via the typesafe adapter — swap providers without changing call shape |

## Running

```bash
go run ./examples/typesafe/decision
go run ./examples/typesafe/question-types
go run ./examples/typesafe/inputs
go run ./examples/typesafe/state
go run ./examples/typesafe/images
go run ./examples/typesafe/box
go run ./examples/typesafe/long-context
go run ./examples/typesafe/provider-options
go run ./examples/typesafe/headers       # mock Fetcher — no key needed
go run ./examples/typesafe/settings     # no key needed
go run ./examples/typesafe/errors        # local errors always, API errors if key set
go run ./examples/typesafe/adapter
```

## Model notes

- `jev-latest` currently maps to `jev-1.13` and rejects image state and `BoxQuestion` with `400 Invalid request`. Use `jev-1.14.0` (or newer) for `images` and `box`.
- The API requires a real decodable image — tiny synthetic PNGs are rejected with `Invalid image dimensions`. `imgdata/photo.jpg` is bundled via `go:embed` for this reason.

## Package shape

```go
provider := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{
    // APIKey falls back to TYPESAFE_AI_API_KEY; BaseURL falls back to
    // TYPESAFE_AI_BASE_URL then https://api.typesafe.ai/v1.
})
model := provider.DecisionModel("jev-latest")
result, err := model.DoDecide(ctx, typesafe.DecisionCallOptions{
    State:     []typesafe.StatePart{...},
    Questions: map[string]typesafe.Question{...},
})
```

Answers arrive in `result.Answers` (a `map[string]Answer` of `ChoiceAnswer`,
`ScoreAnswer`, `BooleanAnswer`, `BoxAnswer`); TypeSafe-specific per-question
confidence is at `result.ProviderMetadata["typesafe"]["confidence"]`. The
`Rounding` field declares that probabilities and scores are already rounded
to 2 decimals by the API.
