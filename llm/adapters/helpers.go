package adapters

import (
	"context"
	"encoding/json"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// GeneratorFunc adapts a function to the llm.Client interface.
type GeneratorFunc func(ctx context.Context, opts llm.GenerateOptions) (*llm.GenerateResult, error)

// Generate calls f with the provided options.
func (f GeneratorFunc) Generate(ctx context.Context, opts llm.GenerateOptions) (*llm.GenerateResult, error) {
	return f(ctx, opts)
}

// Stream returns ErrStreamNotImplemented. GeneratorFunc is a Generate-only
// adapter; tests that need streaming should use a custom Client implementation.
// spec §1.1
func (GeneratorFunc) Stream(ctx context.Context, opts llm.GenerateOptions) (<-chan llm.StreamPart, error) {
	return nil, llm.ErrStreamNotImplemented
}

// streamNotSupported is embedded by provider adapters that do not implement
// streaming in the spike. spec §1.1
type streamNotSupported struct{}

func (streamNotSupported) Stream(ctx context.Context, opts llm.GenerateOptions) (<-chan llm.StreamPart, error) {
	return nil, llm.ErrStreamNotImplemented
}

func decodeSchema(schema json.RawMessage) (any, error) {
	if len(schema) == 0 {
		return nil, nil
	}
	var decoded any
	if err := json.Unmarshal(schema, &decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

func cloneRawMessage(input json.RawMessage) json.RawMessage {
	if input == nil {
		return nil
	}
	cloned := make(json.RawMessage, len(input))
	copy(cloned, input)
	return cloned
}

func toolResultOutputType(isError bool) string {
	if isError {
		return "error-text"
	}
	return "text"
}

func fromUnifiedFinishReason(unified string) llm.FinishReason {
	switch unified {
	case "stop":
		return llm.FinishReasonStop
	case "tool-calls":
		return llm.FinishReasonToolCalls
	case "length":
		return llm.FinishReasonLength
	default:
		return llm.FinishReasonError
	}
}

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
