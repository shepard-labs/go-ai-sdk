package llm

import (
	"context"
	"errors"
	"testing"
)

func TestREQCACHE001_KeyIncludesToolsChangeBustsCache(t *testing.T) {
	underlying := &mockClient{results: []*GenerateResult{
		{Content: []Content{TextContent{Text: "first"}}, FinishReason: FinishReason{Unified: FinishReasonStop}},
		{Content: []Content{TextContent{Text: "second"}}, FinishReason: FinishReason{Unified: FinishReasonStop}},
	}}
	client := WithCache(underlying, newMemoryCache())
	base := GenerateOptions{System: "system", Messages: []Message{{Role: "user", Content: []Content{TextContent{Text: "hello"}}}}}

	first, err := client.Generate(context.Background(), base)
	if err != nil {
		t.Fatalf("first Generate error = %v", err)
	}
	secondOpts := base
	secondOpts.Tools = []Tool{{Name: "different"}}
	second, err := client.Generate(context.Background(), secondOpts)
	if err != nil {
		t.Fatalf("second Generate error = %v", err)
	}
	if first == second {
		t.Fatal("tool change reused cached result")
	}
	if underlying.callCount() != 2 {
		t.Fatalf("underlying calls = %d, want 2", underlying.callCount())
	}
}

func TestREQCACHE002_ErrorsNeverCached(t *testing.T) {
	wantErr := errors.New("transient")
	wantResult := &GenerateResult{FinishReason: FinishReason{Unified: FinishReasonStop}}
	underlying := &mockClient{errors: []error{wantErr, nil}, results: []*GenerateResult{nil, wantResult}}
	cache := newMemoryCache()
	client := WithCache(underlying, cache)

	_, err := client.Generate(context.Background(), GenerateOptions{System: "same"})
	if !errors.Is(err, wantErr) {
		t.Fatalf("first error = %v, want %v", err, wantErr)
	}
	if cache.len() != 0 {
		t.Fatalf("cache len = %d, want 0", cache.len())
	}
	result, err := client.Generate(context.Background(), GenerateOptions{System: "same"})
	if err != nil {
		t.Fatalf("second Generate error = %v", err)
	}
	if result != wantResult {
		t.Fatalf("result = %p, want %p", result, wantResult)
	}
	if underlying.callCount() != 2 {
		t.Fatalf("underlying calls = %d, want 2", underlying.callCount())
	}
}

func TestREQCACHE003_NoBuiltInBackend(t *testing.T) {
	var _ CacheBackend = newMemoryCache()
	defer func() {
		if recover() == nil {
			t.Fatal("WithCache(nil) did not panic")
		}
	}()
	_ = WithCache(&mockClient{}, nil)
}

func TestREQCACHE005_ReturnsSamePointerReadOnly(t *testing.T) {
	want := &GenerateResult{FinishReason: FinishReason{Unified: FinishReasonStop}}
	underlying := &mockClient{results: []*GenerateResult{want}}
	client := WithCache(underlying, newMemoryCache())

	first, err := client.Generate(context.Background(), GenerateOptions{System: "same"})
	if err != nil {
		t.Fatalf("first Generate error = %v", err)
	}
	second, err := client.Generate(context.Background(), GenerateOptions{System: "same"})
	if err != nil {
		t.Fatalf("second Generate error = %v", err)
	}
	if first != want || second != want {
		t.Fatalf("cached pointers first=%p second=%p want=%p", first, second, want)
	}
	if underlying.callCount() != 1 {
		t.Fatalf("underlying calls = %d, want 1", underlying.callCount())
	}
}

func TestCacheKeyIncludesReasoningAndProviderOptions(t *testing.T) {
	highBudget := 4096
	otherBudget := 8192
	base := GenerateOptions{System: "same"}
	high := base
	high.Reasoning = &ReasoningOptions{Effort: ReasoningHigh, BudgetTokens: &highBudget}
	highAgain := base
	highAgain.Reasoning = &ReasoningOptions{Effort: ReasoningHigh, BudgetTokens: &highBudget}
	none := base
	none.Reasoning = &ReasoningOptions{Effort: ReasoningNone}
	other := base
	other.Reasoning = &ReasoningOptions{Effort: ReasoningHigh, BudgetTokens: &otherBudget}
	provider := base
	provider.ProviderOptions = ProviderOptions{"openai": {"reasoningEffort": "high"}}

	highKey, err := cacheKey(high)
	if err != nil {
		t.Fatalf("cacheKey(high) error = %v", err)
	}
	highAgainKey, err := cacheKey(highAgain)
	if err != nil {
		t.Fatalf("cacheKey(highAgain) error = %v", err)
	}
	if highKey != highAgainKey {
		t.Fatalf("same reasoning produced different keys: %q != %q", highKey, highAgainKey)
	}

	for name, opts := range map[string]GenerateOptions{"nil": base, "none": none, "other_budget": other, "provider_options": provider} {
		key, err := cacheKey(opts)
		if err != nil {
			t.Fatalf("cacheKey(%s) error = %v", name, err)
		}
		if key == highKey {
			t.Fatalf("cacheKey(%s) = high key; reasoning/provider options not included", name)
		}
	}
}
