package anthropic

func FinishReasonFromStopReason(reason string) FinishReason {
	switch reason {
	case "pause_turn", "end_turn", "stop_sequence":
		return FinishReasonStop
	case "refusal":
		return FinishReasonContentFilter
	case "max_tokens", "model_context_window_exceeded":
		return FinishReasonLength
	case "tool_use":
		return FinishReasonToolCalls
	case "compaction":
		return FinishReason("other")
	default:
		return FinishReasonUnknown
	}
}
