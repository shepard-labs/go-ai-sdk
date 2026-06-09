package openaicompatible

import "testing"

func TestOptionsMergeOrders(t *testing.T) {
	opts := ProviderOptions{
		"openai-compatible": {"a": 1, "shared": "deprecated"},
		"openaiCompatible":  {"b": 2, "shared": "compat"},
		"my-provider":       {"c": 3, "shared": "raw"},
		"myProvider":        {"d": 4, "shared": "camel"},
	}
	merged, warnings := mergeChatProviderOptions("my-provider", opts)
	if merged["a"] != 1 || merged["b"] != 2 || merged["c"] != 3 || merged["d"] != 4 || merged["shared"] != "camel" {
		t.Fatalf("chat merged = %#v", merged)
	}
	if len(warnings) != 1 || warnings[0].Message != deprecatedProviderOptionsWarningMessage {
		t.Fatalf("warnings = %#v", warnings)
	}
	merged, warnings = mergeEmbeddingProviderOptions("my-provider", opts)
	if merged["d"] != 4 || len(warnings) != 1 {
		t.Fatalf("embedding merged = %#v warnings=%#v", merged, warnings)
	}
	completion := mergeCompletionProviderOptions("my-provider", opts)
	if completion["a"] != nil || completion["b"] != nil || completion["shared"] != "camel" {
		t.Fatalf("completion merged = %#v", completion)
	}
	image := mergeImageProviderOptions("my-provider", opts)
	if image["shared"] != "camel" {
		t.Fatalf("image merged = %#v", image)
	}
}

func TestToCamelCaseAndMetadataKey(t *testing.T) {
	for input, want := range map[string]string{
		"provider-name": "providerName",
		"provider_name": "providerName",
		"providerName":  "providerName",
		"provider-Name": "provider-Name",
	} {
		if got := toCamelCase(input); got != want {
			t.Fatalf("toCamelCase(%q) = %q, want %q", input, got, want)
		}
	}
	if got := metadataKeyForProviderOptions("my-provider", ProviderOptions{"myProvider": {}}); got != "myProvider" {
		t.Fatalf("metadata key = %q", got)
	}
	if got := metadataKeyForProviderOptions("my-provider", ProviderOptions{"my-provider": {}}); got != "my-provider" {
		t.Fatalf("metadata key = %q", got)
	}
}
