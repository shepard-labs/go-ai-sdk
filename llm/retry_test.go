package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func rateLimitErr(retryAfter *time.Duration) *APIError {
	return &APIError{Provider: "test", StatusCode: 429, Code: "rate_limit", Message: "slow down", RetryAfter: retryAfter}
}

func TestRetry_SuccessFirstAttemptNoRetry(t *testing.T) {
	want := &GenerateResult{FinishReason: FinishReason{Unified: FinishReasonStop}}
	underlying := &mockClient{results: []*GenerateResult{want}}
	var retries int
	client := WithRetry(underlying, RetryConfig{
		InitialDelay: time.Millisecond,
		OnRetry:      func(context.Context, int, error, time.Duration) { retries++ },
	})

	got, err := client.Generate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if got != want {
		t.Fatalf("result = %p, want %p", got, want)
	}
	if underlying.callCount() != 1 {
		t.Fatalf("calls = %d, want 1", underlying.callCount())
	}
	if retries != 0 {
		t.Fatalf("retries = %d, want 0", retries)
	}
}

func TestRetry_RetriesRateLimitThenSucceeds(t *testing.T) {
	second := &GenerateResult{FinishReason: FinishReason{Unified: FinishReasonStop}}
	underlying := &mockClient{
		errors:  []error{rateLimitErr(nil), nil},
		results: []*GenerateResult{nil, second},
	}
	var retryAttempts []int
	client := WithRetry(underlying, RetryConfig{
		InitialDelay: time.Millisecond,
		Jitter:       false,
		OnRetry: func(_ context.Context, attempt int, _ error, _ time.Duration) {
			retryAttempts = append(retryAttempts, attempt)
		},
	})

	got, err := client.Generate(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if got != second {
		t.Fatalf("result = %p, want second attempt %p", got, second)
	}
	if underlying.callCount() != 2 {
		t.Fatalf("calls = %d, want 2", underlying.callCount())
	}
	if len(retryAttempts) != 1 || retryAttempts[0] != 2 {
		t.Fatalf("OnRetry attempts = %v, want [2]", retryAttempts)
	}
}

func TestRetry_AllAttemptsFailReturnsLastError(t *testing.T) {
	errs := []error{rateLimitErr(nil), rateLimitErr(nil), errors.New("final")}
	underlying := &mockClient{errors: errs}
	client := WithRetry(underlying, RetryConfig{
		MaxAttempts:  3,
		InitialDelay: time.Millisecond,
		ShouldRetry:  func(error) bool { return true },
	})

	_, err := client.Generate(context.Background(), GenerateOptions{})
	if err == nil || err.Error() != "final" {
		t.Fatalf("error = %v, want final", err)
	}
	if underlying.callCount() != 3 {
		t.Fatalf("calls = %d, want 3", underlying.callCount())
	}
}

func TestRetry_MaxElapsedStopsEarly(t *testing.T) {
	underlying := &mockClient{errors: []error{rateLimitErr(nil), rateLimitErr(nil), rateLimitErr(nil)}}
	client := WithRetry(underlying, RetryConfig{
		MaxAttempts:  5,
		InitialDelay: time.Hour, // backoff dwarfs the budget
		Jitter:       false,
		MaxElapsed:   10 * time.Millisecond,
	})

	_, err := client.Generate(context.Background(), GenerateOptions{})
	if !IsRateLimit(err) {
		t.Fatalf("error = %v, want rate-limit", err)
	}
	// First attempt fails, the projected delay exceeds MaxElapsed, so it stops.
	if underlying.callCount() != 1 {
		t.Fatalf("calls = %d, want 1", underlying.callCount())
	}
}

func TestRetry_NonRetryableDoesNotRetry(t *testing.T) {
	authErr := &APIError{Provider: "test", StatusCode: 401, Code: "auth", Message: "nope"}
	underlying := &mockClient{errors: []error{authErr}}
	client := WithRetry(underlying, RetryConfig{InitialDelay: time.Millisecond})

	_, err := client.Generate(context.Background(), GenerateOptions{})
	if !IsAuth(err) {
		t.Fatalf("error = %v, want auth", err)
	}
	if underlying.callCount() != 1 {
		t.Fatalf("calls = %d, want 1", underlying.callCount())
	}
}

func TestRetry_RespectsRetryAfter(t *testing.T) {
	ra := 40 * time.Millisecond
	second := &GenerateResult{FinishReason: FinishReason{Unified: FinishReasonStop}}
	underlying := &mockClient{
		errors:  []error{rateLimitErr(&ra), nil},
		results: []*GenerateResult{nil, second},
	}
	var gotDelay time.Duration
	client := WithRetry(underlying, RetryConfig{
		InitialDelay: time.Millisecond, // would be much smaller than RetryAfter
		Jitter:       false,
		OnRetry: func(_ context.Context, _ int, _ error, delay time.Duration) {
			gotDelay = delay
		},
	})

	start := time.Now()
	got, err := client.Generate(context.Background(), GenerateOptions{})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if got != second {
		t.Fatalf("result = %p, want %p", got, second)
	}
	if gotDelay != ra {
		t.Fatalf("retry delay = %v, want %v", gotDelay, ra)
	}
	if elapsed < ra {
		t.Fatalf("elapsed = %v, want >= %v", elapsed, ra)
	}
}
