package anthropic

import (
	"encoding/json"
	"fmt"
	"math"
)

type UsageConversionError struct {
	Field string
	Value uint64
}

func (e *UsageConversionError) Error() string {
	return fmt.Sprintf("anthropic usage %s value %d exceeds int range", e.Field, e.Value)
}

type apiResponseUsage struct {
	InputTokens, OutputTokens                      uint64
	CacheCreationInputTokens, CacheReadInputTokens uint64
	ServiceTier, InferenceGeo                      string
	CacheCreation                                  *CacheCreationUsage
	ServerToolUse                                  *ServerToolUsage
	OutputTokensDetails                            *OutputTokensDetails
	Iterations                                     []apiUsageIteration
	Raw                                            json.RawMessage
}

func UsageFromResponseUsage(usage ResponseUsage) Usage {
	result := Usage{
		InputTokens:  TokenUsage{Total: usage.InputTokens, CacheWrite: usage.CacheCreationInputTokens, CacheRead: usage.CacheReadInputTokens},
		OutputTokens: TokenUsage{Total: usage.OutputTokens, Text: usage.OutputTokens},
		TotalTokens:  usage.InputTokens + usage.OutputTokens, Iterations: usage.Iterations, Raw: append([]byte(nil), usage.Raw...),
	}
	if usage.OutputTokensDetails != nil && usage.OutputTokensDetails.ThinkingTokens != nil {
		reasoning := int(*usage.OutputTokensDetails.ThinkingTokens)
		result.OutputTokens.Reasoning = reasoning
		result.OutputTokens.Text = usage.OutputTokens - reasoning
	}
	return result
}

func usageFromAPIUsage(usage apiResponseUsage) (Usage, error) {
	input, output := usage.InputTokens, usage.OutputTokens
	servedByFallback := false
	for _, iteration := range usage.Iterations {
		if iteration.Type == "fallback_message" {
			servedByFallback = true
			break
		}
	}
	if len(usage.Iterations) > 0 && !servedByFallback {
		var iterationInput, iterationOutput uint64
		hasExecutor := false
		for _, iteration := range usage.Iterations {
			if iteration.Type != "compaction" && iteration.Type != "message" {
				continue
			}
			hasExecutor = true
			var err error
			iterationInput, err = checkedAddUint64("iterations.input_tokens", iterationInput, iteration.InputTokens)
			if err != nil {
				return Usage{}, err
			}
			iterationOutput, err = checkedAddUint64("iterations.output_tokens", iterationOutput, iteration.OutputTokens)
			if err != nil {
				return Usage{}, err
			}
		}
		if hasExecutor {
			input, output = iterationInput, iterationOutput
		}
	}
	inputTotal, err := checkedAddUint64("input_tokens.total", input, usage.CacheCreationInputTokens)
	if err != nil {
		return Usage{}, err
	}
	inputTotal, err = checkedAddUint64("input_tokens.total", inputTotal, usage.CacheReadInputTokens)
	if err != nil {
		return Usage{}, err
	}
	reasoning := uint64(0)
	if usage.OutputTokensDetails != nil && usage.OutputTokensDetails.ThinkingTokens != nil {
		reasoning = *usage.OutputTokensDetails.ThinkingTokens
	}
	total, err := checkedAddUint64("total_tokens", inputTotal, output)
	if err != nil {
		return Usage{}, err
	}
	values := []struct {
		name  string
		value uint64
	}{
		{"input_tokens.total", inputTotal}, {"input_tokens.no_cache", input},
		{"cache_creation_input_tokens", usage.CacheCreationInputTokens}, {"cache_read_input_tokens", usage.CacheReadInputTokens},
		{"output_tokens", output}, {"output_tokens.reasoning", reasoning}, {"total_tokens", total},
	}
	converted := make(map[string]int, len(values))
	for _, value := range values {
		if value.value > uint64(math.MaxInt) {
			return Usage{}, &UsageConversionError{Field: value.name, Value: value.value}
		}
		converted[value.name] = int(value.value)
	}
	text, err := checkedSubtractToInt("output_tokens.text", output, reasoning)
	if err != nil {
		return Usage{}, err
	}
	iterations := make([]UsageIteration, len(usage.Iterations))
	for i, iteration := range usage.Iterations {
		input, err := checkedUint64ToInt("iterations.input_tokens", iteration.InputTokens)
		if err != nil {
			return Usage{}, err
		}
		output, err := checkedUint64ToInt("iterations.output_tokens", iteration.OutputTokens)
		if err != nil {
			return Usage{}, err
		}
		iterations[i] = UsageIteration{InputTokens: input, OutputTokens: output}
	}
	return Usage{
		InputTokens:  TokenUsage{Total: converted["input_tokens.total"], CacheWrite: converted["cache_creation_input_tokens"], CacheRead: converted["cache_read_input_tokens"]},
		OutputTokens: TokenUsage{Total: converted["output_tokens"], Text: text, Reasoning: converted["output_tokens.reasoning"]},
		TotalTokens:  converted["total_tokens"], Iterations: iterations, Raw: append([]byte(nil), usage.Raw...),
	}, nil
}

func checkedUint64ToInt(field string, value uint64) (int, error) {
	if value > uint64(math.MaxInt) {
		return 0, &UsageConversionError{Field: field, Value: value}
	}
	return int(value), nil
}

func checkedSubtractToInt(field string, left, right uint64) (int, error) {
	if left >= right {
		return checkedUint64ToInt(field, left-right)
	}
	difference := right - left
	if difference > uint64(math.MaxInt)+1 {
		return 0, &UsageConversionError{Field: field, Value: difference}
	}
	if difference == uint64(math.MaxInt)+1 {
		return math.MinInt, nil
	}
	return -int(difference), nil
}

func checkedAddUint64(field string, left, right uint64) (uint64, error) {
	if math.MaxUint64-left < right {
		return 0, &UsageConversionError{Field: field, Value: math.MaxUint64}
	}
	return left + right, nil
}
