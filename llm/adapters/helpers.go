package adapters

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

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
// streaming. spec §1.1
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
		return llm.FinishReason{Unified: llm.FinishReasonStop, Raw: unified}
	case "tool-calls":
		return llm.FinishReason{Unified: llm.FinishReasonToolCalls, Raw: unified}
	case "length":
		return llm.FinishReason{Unified: llm.FinishReasonLength, Raw: unified}
	case "content-filter":
		return llm.FinishReason{Unified: llm.FinishReasonContentFilter, Raw: unified}
	case "other":
		return llm.FinishReason{Unified: llm.FinishReasonOther, Raw: unified}
	default:
		return llm.FinishReason{Unified: llm.FinishReasonError, Raw: unified}
	}
}

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// unsupportedFeature either returns an UnsupportedFeatureError (policy
// error/default) or appends a Warning to warnings and returns nil (policy warn).
func unsupportedFeature(policy llm.UnsupportedFeaturePolicy, provider, feature, msg string, warnings *[]llm.Warning) error {
	if policy == llm.UnsupportedFeaturePolicyWarn {
		*warnings = append(*warnings, llm.Warning{
			Code:     "unsupported_feature",
			Message:  fmt.Sprintf("%s: %s", feature, msg),
			Provider: provider,
		})
		return nil
	}
	return &llm.UnsupportedFeatureError{Provider: provider, Feature: feature, Message: msg}
}

// toHTTPHeader converts a neutral single-valued header map to an http.Header.
// Returns nil for an empty map.
func toHTTPHeader(headers map[string]string) http.Header {
	if len(headers) == 0 {
		return nil
	}
	out := make(http.Header, len(headers))
	for k, v := range headers {
		out.Set(k, v)
	}
	return out
}

// flattenHeader converts an http.Header to a neutral single-valued header map,
// keeping the first value per key. Returns nil for an empty header.
func flattenHeader(header http.Header) map[string]string {
	if len(header) == 0 {
		return nil
	}
	out := make(map[string]string, len(header))
	for k, v := range header {
		if len(v) > 0 {
			out[k] = v[0]
		}
	}
	return out
}
