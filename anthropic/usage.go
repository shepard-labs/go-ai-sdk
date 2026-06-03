package anthropic

func UsageFromResponseUsage(usage ResponseUsage) Usage {
	return Usage{
		InputTokens: TokenUsage{
			Total:      usage.InputTokens,
			CacheWrite: usage.CacheCreationInputTokens,
			CacheRead:  usage.CacheReadInputTokens,
		},
		OutputTokens: TokenUsage{Total: usage.OutputTokens},
		TotalTokens:  usage.InputTokens + usage.OutputTokens,
		Iterations:   usage.Iterations,
	}
}
