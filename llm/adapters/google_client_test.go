package adapters

import (
	"errors"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

func TestGoogleNeutralReasoningMapsToReasoning(t *testing.T) {
	sdkOpts, warnings, err := toGoogleOptions(GenerateOptions{Reasoning: &llm.ReasoningOptions{Effort: llm.ReasoningHigh}})
	if err != nil {
		t.Fatalf("toGoogleOptions error = %v", err)
	}
	if sdkOpts.Reasoning != "high" {
		t.Fatalf("Reasoning = %q, want high", sdkOpts.Reasoning)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings = %#v, want none", warnings)
	}
}

func TestGoogleNeutralReasoningBudgetUnsupported(t *testing.T) {
	budget := 100
	_, _, err := toGoogleOptions(GenerateOptions{Reasoning: &llm.ReasoningOptions{Effort: llm.ReasoningHigh, BudgetTokens: &budget}})
	var ufe *llm.UnsupportedFeatureError
	if !errors.As(err, &ufe) || ufe.Feature != "reasoning_budget" {
		t.Fatalf("error = %v, want UnsupportedFeatureError(reasoning_budget)", err)
	}
}

func TestGoogleNeutralReasoningBudgetWarnStillAppliesEffort(t *testing.T) {
	budget := 100
	sdkOpts, warnings, err := toGoogleOptions(GenerateOptions{
		Reasoning:                &llm.ReasoningOptions{Effort: llm.ReasoningHigh, BudgetTokens: &budget},
		UnsupportedFeaturePolicy: llm.UnsupportedFeaturePolicyWarn,
	})
	if err != nil {
		t.Fatalf("toGoogleOptions error = %v", err)
	}
	if sdkOpts.Reasoning != "high" {
		t.Fatalf("Reasoning = %q, want high", sdkOpts.Reasoning)
	}
	if len(warnings) != 1 || warnings[0].Code != "reasoning_budget_unsupported" {
		t.Fatalf("warnings = %#v, want budget unsupported warning", warnings)
	}
}
