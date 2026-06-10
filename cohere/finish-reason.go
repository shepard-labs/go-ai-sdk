package cohere

func mapCohereFinishReason(raw string) string {
	switch raw {
	case "COMPLETE", "STOP_SEQUENCE":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "ERROR":
		return "error"
	case "TOOL_CALL":
		return "tool-calls"
	default:
		return "other"
	}
}
