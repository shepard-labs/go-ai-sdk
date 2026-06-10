package google

// mapGoogleFinishReason maps a Google API finishReason string to the SDK
// FinishReason type.
//
// Mapping (per spec §"Finish Reason Mapping"):
//
//	STOP (no tool calls)      → "stop"
//	STOP (has tool calls)     → "tool-calls"
//	MAX_TOKENS                → "length"
//	IMAGE_SAFETY              → "content-filter"
//	RECITATION                → "content-filter"
//	SAFETY                    → "content-filter"
//	BLOCKLIST                 → "content-filter"
//	PROHIBITED_CONTENT        → "content-filter"
//	SPII                      → "content-filter"
//	MALFORMED_FUNCTION_CALL   → "error"
//	anything else             → "other"
//
// hasToolCalls should be true when any assistant content part is a non-provider-
// executed tool call (i.e. a client-side function call).
func mapGoogleFinishReason(raw string, hasToolCalls bool) FinishReason {
	var unified string
	switch raw {
	case "STOP":
		if hasToolCalls {
			unified = "tool-calls"
		} else {
			unified = "stop"
		}
	case "MAX_TOKENS":
		unified = "length"
	case "IMAGE_SAFETY", "RECITATION", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		unified = "content-filter"
	case "MALFORMED_FUNCTION_CALL":
		unified = "error"
	default:
		unified = "other"
	}
	return FinishReason{Unified: unified, Raw: raw}
}
