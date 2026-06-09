package openaicompatible

func finishReasonFromOpenAI(raw string) FinishReason {
	switch raw {
	case "stop":
		return FinishReason{Unified: "stop", Raw: raw}
	case "length":
		return FinishReason{Unified: "length", Raw: raw}
	case "content_filter":
		return FinishReason{Unified: "content-filter", Raw: raw}
	case "function_call", "tool_calls":
		return FinishReason{Unified: "tool-calls", Raw: raw}
	default:
		return FinishReason{Unified: "other", Raw: raw}
	}
}

func errorFinishReason() FinishReason { return FinishReason{Unified: "error", Raw: "error"} }
