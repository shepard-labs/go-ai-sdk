package cohere

import "encoding/json"

type CohereUsageTokens struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func convertCohereUsage(t *CohereUsageTokens) Usage {
	if t == nil {
		return Usage{}
	}
	in, out := t.InputTokens, t.OutputTokens
	raw, _ := json.Marshal(t)
	return Usage{InputTokens: TokenCounts{Total: &in, NoCache: &in}, OutputTokens: OutputTokenCounts{Total: &out, Text: &out}, Raw: raw}
}
