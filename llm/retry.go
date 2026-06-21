package llm

import (
	"context"
	"math/rand"
	"time"
)

// RetryConfig controls retry behavior.
type RetryConfig struct {
	// MaxAttempts is the total number of attempts (1 = no retry).
	// Default: 3.
	MaxAttempts int

	// InitialDelay is the base backoff duration.
	// Default: 500ms.
	InitialDelay time.Duration

	// MaxDelay caps the backoff duration.
	// Default: 30s.
	MaxDelay time.Duration

	// Multiplier is the backoff growth factor.
	// Default: 2.0.
	Multiplier float64

	// Jitter adds random noise to avoid thundering herd.
	// Default: true.
	Jitter bool

	// MaxElapsed is the total time budget across all attempts.
	// Zero means no elapsed-time limit.
	MaxElapsed time.Duration

	// ShouldRetry classifies errors. If nil, defaults to IsRateLimit || IsTemporary.
	ShouldRetry func(err error) bool

	// OnRetry, if non-nil, is called before each retry sleep with the
	// upcoming attempt number, the error that triggered the retry, and the
	// delay before the next attempt.
	OnRetry func(ctx context.Context, attempt int, err error, delay time.Duration)
}

func (cfg RetryConfig) withDefaults() RetryConfig {
	if cfg.MaxAttempts == 0 {
		cfg.MaxAttempts = 3
	}
	if cfg.InitialDelay == 0 {
		cfg.InitialDelay = 500 * time.Millisecond
	}
	if cfg.MaxDelay == 0 {
		cfg.MaxDelay = 30 * time.Second
	}
	if cfg.Multiplier == 0 {
		cfg.Multiplier = 2.0
	}
	if cfg.ShouldRetry == nil {
		cfg.ShouldRetry = func(err error) bool {
			return IsRateLimit(err) || IsTemporary(err)
		}
	}
	return cfg
}

// WithRetry wraps client with retry-on-error behavior.
func WithRetry(client Client, cfg RetryConfig) Client {
	return retryClient{client: client, cfg: cfg.withDefaults()}
}

type retryClient struct {
	client Client
	cfg    RetryConfig
}

// backoff computes the delay before the given retry attempt (1-based: the
// delay before attempt 2 uses attempt=1's index).
func (c retryClient) backoff(retryIndex int) time.Duration {
	d := float64(c.cfg.InitialDelay)
	for range retryIndex {
		d *= c.cfg.Multiplier
	}
	if d > float64(c.cfg.MaxDelay) {
		d = float64(c.cfg.MaxDelay)
	}
	delay := time.Duration(d)
	if c.cfg.Jitter && delay > 0 {
		delay = time.Duration(rand.Int63n(int64(delay)))
	}
	return delay
}

func (c retryClient) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	start := time.Now()
	var lastErr error
	for attempt := 1; attempt <= c.cfg.MaxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result, err := c.client.Generate(ctx, opts)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if attempt == c.cfg.MaxAttempts || !c.cfg.ShouldRetry(err) {
			return nil, lastErr
		}

		delay := c.backoff(attempt)
		if ra, ok := RetryAfter(err); ok {
			delay = ra
		}

		if c.cfg.MaxElapsed > 0 && time.Since(start)+delay > c.cfg.MaxElapsed {
			return nil, lastErr
		}

		if c.cfg.OnRetry != nil {
			c.cfg.OnRetry(ctx, attempt+1, err, delay)
		}
		if err := sleepCtx(ctx, delay); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

// Stream retries only synchronous errors returned from Stream before any part
// is emitted. In-stream StreamError parts are forwarded unchanged and never
// retried. A retry calls the inner Stream again from scratch.
func (c retryClient) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	start := time.Now()
	var lastErr error
	for attempt := 1; attempt <= c.cfg.MaxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ch, err := c.client.Stream(ctx, opts)
		if err == nil {
			return ch, nil
		}
		lastErr = err
		if attempt == c.cfg.MaxAttempts || !c.cfg.ShouldRetry(err) {
			return nil, lastErr
		}

		delay := c.backoff(attempt)
		if ra, ok := RetryAfter(err); ok {
			delay = ra
		}

		if c.cfg.MaxElapsed > 0 && time.Since(start)+delay > c.cfg.MaxElapsed {
			return nil, lastErr
		}

		if c.cfg.OnRetry != nil {
			c.cfg.OnRetry(ctx, attempt+1, err, delay)
		}
		if err := sleepCtx(ctx, delay); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}
