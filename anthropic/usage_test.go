package anthropic

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// Pinned upstream authority (published package contents, inspected 2026-08-30):
// @ai-sdk/anthropic@3.0.115 package/src/convert-anthropic-messages-usage.ts:54-113 and
// package/src/anthropic-messages-language-model.ts:2399-2407,2486-2510,2542-2545;
// @ai-sdk/anthropic@4.0.46 package/src/convert-anthropic-usage.ts:54-113 (same formulas).
// Vercel ai repo commit 69428b1f8b037e4d118fb4853428d5c4e620493c contains the inspected 4.0.46 source.
// Equations: input.total = executor input + cache_creation + cache_read;
// output.total = executor output; output.text = output.total - thinking_tokens;
// aggregate total = input.total + output.total. With non-empty iterations and no
// fallback_message, executor input/output are sums of compaction/message only;
// fallback_message or no executor iterations uses top-level input/output. Cache is
// added once from the top level, never summed from iterations. Streaming assigns
// message_start input/cache, then message_delta replaces every present usage field
// (including repeated, zero, or decreased values); absent fields preserve prior raw values.
func TestUsageUsageEquationsAndRawFields(t *testing.T) {
	thinking := uint64(4)
	raw := json.RawMessage(`{"input_tokens":99,"unknown":{"kept":true}}`)
	usage, err := usageFromAPIUsage(apiResponseUsage{
		InputTokens: 99, OutputTokens: 99, CacheCreationInputTokens: 3, CacheReadInputTokens: 2,
		ServiceTier: "standard", InferenceGeo: "us", CacheCreation: &CacheCreationUsage{Ephemeral5mInputTokens: 3},
		ServerToolUse: &ServerToolUsage{WebSearchRequests: 1}, OutputTokensDetails: &OutputTokensDetails{ThinkingTokens: &thinking},
		Iterations: []apiUsageIteration{{Type: "compaction", InputTokens: 10, OutputTokens: 2}, {Type: "advisor_message", InputTokens: 100, OutputTokens: 100}, {Type: "message", InputTokens: 7, OutputTokens: 8}}, Raw: raw,
	})
	if err != nil {
		t.Fatal(err)
	}
	if usage.InputTokens.Total != 22 || usage.OutputTokens.Total != 10 || usage.OutputTokens.Reasoning != 4 || usage.OutputTokens.Text != 6 || usage.TotalTokens != 32 || string(usage.Raw) != string(raw) {
		t.Fatalf("usage = %#v raw=%s", usage, usage.Raw)
	}
}

func TestUsageUsageIterationsFallbackAndCacheAreAdditive(t *testing.T) {
	for _, tc := range []struct {
		name                       string
		iterations                 []apiUsageIteration
		wantInput, wantOutput, all int
	}{
		{
			name: "no executor iterations uses top level",
			iterations: []apiUsageIteration{
				{Type: "advisor_message", InputTokens: 100, OutputTokens: 100},
			},
			wantInput: 15, wantOutput: 9, all: 24,
		},
		{
			name: "fallback uses top level instead of executor iterations",
			iterations: []apiUsageIteration{
				{Type: "compaction", InputTokens: 100, OutputTokens: 100},
				{Type: "fallback_message", InputTokens: 1, OutputTokens: 1},
			},
			wantInput: 15, wantOutput: 9, all: 24,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			usage, err := usageFromAPIUsage(apiResponseUsage{
				InputTokens: 10, OutputTokens: 9, CacheCreationInputTokens: 2, CacheReadInputTokens: 3, Iterations: tc.iterations,
			})
			if err != nil {
				t.Fatal(err)
			}
			if usage.InputTokens.Total != tc.wantInput || usage.InputTokens.CacheWrite != 2 || usage.InputTokens.CacheRead != 3 || usage.OutputTokens.Total != tc.wantOutput || usage.TotalTokens != tc.all {
				t.Fatalf("usage = %#v", usage)
			}
		})
	}
}

func TestUsageStreamPresentFieldsReplaceAndAbsentPreserve(t *testing.T) {
	sse := event("message_start", `{"message":{"id":"m","model":"c","usage":{"input_tokens":9,"cache_read_input_tokens":5,"service_tier":"standard","inference_geo":"us","cache_creation":{"ephemeral_5m_input_tokens":2},"server_tool_use":{"web_search_requests":1},"raw_start":"kept"}}}`) +
		event("message_delta", `{"delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3,"output_tokens":7,"cache_read_input_tokens":0,"output_tokens_details":{"thinking_tokens":2},"raw_delta":"kept"}}`) + event("message_stop", `{}`)
	parts := parseStream(t, sse)
	finish := parts[len(parts)-1].(StreamFinish)
	if finish.Usage.InputTokens.Total != 3 || finish.Usage.InputTokens.CacheRead != 0 || finish.Usage.OutputTokens.Text != 5 || finish.Usage.OutputTokens.Reasoning != 2 {
		t.Fatalf("usage = %#v", finish.Usage)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(finish.Usage.Raw, &raw); err != nil {
		t.Fatal(err)
	}
	if raw["service_tier"] == nil || raw["inference_geo"] == nil || raw["cache_creation"] == nil || raw["server_tool_use"] == nil || raw["raw_start"] == nil || raw["raw_delta"] == nil {
		t.Fatalf("raw = %s", finish.Usage.Raw)
	}
	const wantRaw = `{"cache_creation":{"ephemeral_5m_input_tokens":2},"cache_read_input_tokens":0,"inference_geo":"us","input_tokens":3,"output_tokens":7,"output_tokens_details":{"thinking_tokens":2},"raw_delta":"kept","raw_start":"kept","server_tool_use":{"web_search_requests":1},"service_tier":"standard"}`
	if string(finish.Usage.Raw) != wantRaw {
		t.Fatalf("raw = %s", finish.Usage.Raw)
	}
}

func TestUsageDuplicateStartsProduceOneFinishOrOneError(t *testing.T) {
	start := `{"message":{"id":"m","model":"c","usage":{"input_tokens":1}}}`
	parts := parseStream(t, event("message_start", start)+event("message_start", start)+event("message_stop", `{}`))
	if len(parts) != 3 {
		t.Fatalf("parts = %#v", parts)
	}
	if _, ok := parts[2].(StreamFinish); !ok {
		t.Fatalf("parts = %#v", parts)
	}

	parts = parseStream(t, event("message_start", start)+event("message_start", `{"message":{"id":"other","model":"c","usage":{"input_tokens":1}}}`))
	if len(parts) != 3 {
		t.Fatalf("parts = %#v", parts)
	}
	if _, ok := parts[2].(StreamError); !ok {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestUsageUsageOverflowTypedAndTerminal(t *testing.T) {
	_, err := usageFromAPIUsage(apiResponseUsage{InputTokens: math.MaxUint64, CacheReadInputTokens: 1})
	var conversion *UsageConversionError
	if !errors.As(err, &conversion) {
		t.Fatalf("error = %v", err)
	}
	sse := event("message_start", `{"message":{"id":"m","model":"c","usage":{"input_tokens":18446744073709551615}}}`) +
		event("message_delta", `{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`) + event("message_stop", `{}`)
	parts := parseStream(t, sse)
	if len(parts) != 3 {
		t.Fatalf("parts = %#v", parts)
	}
	streamErr, ok := parts[2].(StreamError)
	if !ok || !errors.As(streamErr.Err, &conversion) {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestUsageLegacyUsageAPICompatibility(t *testing.T) {
	var _ int = ResponseUsage{}.InputTokens
	usage := UsageFromResponseUsage(ResponseUsage{
		InputTokens: 3, OutputTokens: 2,
		ServerToolUse: &UsageIteration{}, Executor: &ExecutorIteration{}, Advisor: &AdvisorIteration{},
	})
	if usage.TotalTokens != 5 {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestUsageNegativeTextEquation(t *testing.T) {
	thinking := uint64(7)
	usage, err := usageFromAPIUsage(apiResponseUsage{OutputTokens: 3, OutputTokensDetails: &OutputTokensDetails{ThinkingTokens: &thinking}})
	if err != nil {
		t.Fatal(err)
	}
	if usage.OutputTokens.Text != -4 {
		t.Fatalf("text = %d", usage.OutputTokens.Text)
	}
}

func TestUsageNonStreamRawExactAndDeepCopied(t *testing.T) {
	raw := []byte("{ \"output_tokens\" : 2, \"input_tokens\":1, \"unknown\": true }")
	var api apiUsage
	if err := json.Unmarshal(raw, &api); err != nil {
		t.Fatal(err)
	}
	usage, err := convertAPIUsage(api)
	if err != nil {
		t.Fatal(err)
	}
	if string(usage.Raw) != string(raw) {
		t.Fatalf("raw = %q", usage.Raw)
	}
	raw[0] = '['
	api.rawBytes[1] = 'X'
	if string(usage.Raw) != "{ \"output_tokens\" : 2, \"input_tokens\":1, \"unknown\": true }" {
		t.Fatalf("mutated raw = %q", usage.Raw)
	}
}
