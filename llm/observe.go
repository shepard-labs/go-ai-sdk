package llm

import (
	"context"
	"time"
)

// ObserveConfig contains optional hooks called during generation.
// Any hook field left nil is a no-op.
type ObserveConfig struct {
	// OnRequest is called before each Generate or Stream call.
	OnRequest func(ctx context.Context, opts GenerateOptions)

	// OnResult is called after a successful Generate call.
	OnResult func(ctx context.Context, opts GenerateOptions, result *GenerateResult, duration time.Duration)

	// OnError is called after a failed Generate or Stream call.
	OnError func(ctx context.Context, opts GenerateOptions, err error, duration time.Duration)

	// OnRetry is called before each retry attempt.
	OnRetry func(ctx context.Context, attempt int, err error, delay time.Duration)

	// OnStreamPart is called for each part emitted from a Stream.
	// Called in the streaming goroutine — must not block.
	OnStreamPart func(ctx context.Context, part StreamPart)
}

// WithObserver wraps client with observability hooks.
func WithObserver(client Client, cfg ObserveConfig) Client {
	return observerClient{client: client, cfg: cfg}
}

type observerClient struct {
	client Client
	cfg    ObserveConfig
}

// safe runs fn, recovering and ignoring any panic so a hook cannot crash the
// caller.
func safe(fn func()) {
	if fn == nil {
		return
	}
	defer func() { _ = recover() }()
	fn()
}

func (c observerClient) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	if c.cfg.OnRequest != nil {
		safe(func() { c.cfg.OnRequest(ctx, opts) })
	}
	start := time.Now()
	result, err := c.client.Generate(ctx, opts)
	duration := time.Since(start)
	if err != nil {
		if c.cfg.OnError != nil {
			safe(func() { c.cfg.OnError(ctx, opts, err, duration) })
		}
		return nil, err
	}
	if c.cfg.OnResult != nil {
		safe(func() { c.cfg.OnResult(ctx, opts, result, duration) })
	}
	return result, nil
}

func (c observerClient) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	if c.cfg.OnRequest != nil {
		safe(func() { c.cfg.OnRequest(ctx, opts) })
	}
	start := time.Now()
	ch, err := c.client.Stream(ctx, opts)
	if err != nil {
		if c.cfg.OnError != nil {
			safe(func() { c.cfg.OnError(ctx, opts, err, time.Since(start)) })
		}
		return nil, err
	}
	if c.cfg.OnStreamPart == nil {
		return ch, nil
	}
	out := make(chan StreamPart)
	go func() {
		defer close(out)
		for part := range ch {
			safe(func() { c.cfg.OnStreamPart(ctx, part) })
			out <- part
		}
	}()
	return out, nil
}

func (c observerClient) Identity() Identity {
	if p, ok := c.client.(IdentifiedClient); ok {
		return p.Identity()
	}
	return Identity{}
}

func (c observerClient) RequestIdentity(opts GenerateOptions) (Identity, error) {
	return clientIdentity(c.client, opts)
}
