package openaicompatible

const deprecatedProviderOptionsWarningMessage = "The 'openai-compatible' key in providerOptions is deprecated. Use 'openaiCompatible' instead."

func mergeChatProviderOptions(name string, opts ProviderOptions) (map[string]any, []Warning) {
	return mergeProviderOptions(name, opts, true)
}

func mergeEmbeddingProviderOptions(name string, opts ProviderOptions) (map[string]any, []Warning) {
	return mergeProviderOptions(name, opts, true)
}

func mergeCompletionProviderOptions(name string, opts ProviderOptions) map[string]any {
	merged, _ := mergeProviderOptions(name, opts, false)
	return merged
}

func mergeImageProviderOptions(name string, opts ProviderOptions) map[string]any {
	merged, _ := mergeProviderOptions(name, opts, false)
	return merged
}

func mergeProviderOptions(name string, opts ProviderOptions, includeCompatibilityKeys bool) (map[string]any, []Warning) {
	merged := map[string]any{}
	var warnings []Warning
	mergeKey := func(key string) {
		for k, v := range opts[key] {
			merged[k] = v
		}
	}
	if includeCompatibilityKeys {
		if _, ok := opts["openai-compatible"]; ok {
			warnings = append(warnings, Warning{Type: "other", Message: deprecatedProviderOptionsWarningMessage})
			mergeKey("openai-compatible")
		}
		mergeKey("openaiCompatible")
	}
	mergeKey(name)
	mergeKey(toCamelCase(name))
	return merged, warnings
}
