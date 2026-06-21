package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

// streamMock is a Client whose Stream emits a fixed set of parts.
type streamMock struct {
	parts     []StreamPart
	streamErr error
	genResult *GenerateResult
	genErr    error
}

func (m *streamMock) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	return m.genResult, m.genErr
}

func (m *streamMock) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	if m.streamErr != nil {
		return nil, m.streamErr
	}
	ch := make(chan StreamPart)
	go func() {
		defer close(ch)
		for _, p := range m.parts {
			ch <- p
		}
	}()
	return ch, nil
}

func TestObserve_OnRequestFiresBeforeCall(t *testing.T) {
	var order []string
	underlying := GeneratorFuncShim(func(context.Context, GenerateOptions) (*GenerateResult, error) {
		order = append(order, "call")
		return &GenerateResult{}, nil
	})
	client := WithObserver(underlying, ObserveConfig{
		OnRequest: func(context.Context, GenerateOptions) { order = append(order, "request") },
	})

	if _, err := client.Generate(context.Background(), GenerateOptions{}); err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if len(order) != 2 || order[0] != "request" || order[1] != "call" {
		t.Fatalf("order = %v, want [request call]", order)
	}
}

func TestObserve_OnResultFiresWithDuration(t *testing.T) {
	underlying := GeneratorFuncShim(func(context.Context, GenerateOptions) (*GenerateResult, error) {
		time.Sleep(2 * time.Millisecond)
		return &GenerateResult{}, nil
	})
	var gotDuration time.Duration
	var fired bool
	client := WithObserver(underlying, ObserveConfig{
		OnResult: func(_ context.Context, _ GenerateOptions, _ *GenerateResult, d time.Duration) {
			fired = true
			gotDuration = d
		},
	})

	if _, err := client.Generate(context.Background(), GenerateOptions{}); err != nil {
		t.Fatalf("Generate error = %v", err)
	}
	if !fired {
		t.Fatal("OnResult did not fire")
	}
	if gotDuration <= 0 {
		t.Fatalf("duration = %v, want > 0", gotDuration)
	}
}

func TestObserve_OnErrorFiresOnFailure(t *testing.T) {
	wantErr := errors.New("boom")
	underlying := GeneratorFuncShim(func(context.Context, GenerateOptions) (*GenerateResult, error) {
		return nil, wantErr
	})
	var gotErr error
	client := WithObserver(underlying, ObserveConfig{
		OnError: func(_ context.Context, _ GenerateOptions, err error, _ time.Duration) {
			gotErr = err
		},
	})

	_, err := client.Generate(context.Background(), GenerateOptions{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
	if !errors.Is(gotErr, wantErr) {
		t.Fatalf("OnError err = %v, want %v", gotErr, wantErr)
	}
}

func TestObserve_OnStreamPartFiresForEachPart(t *testing.T) {
	parts := []StreamPart{
		StreamTextStart{},
		StreamTextDelta{Text: "a"},
		StreamTextDelta{Text: "b"},
		StreamFinish{},
	}
	underlying := &streamMock{parts: parts}
	var observed []StreamPart
	client := WithObserver(underlying, ObserveConfig{
		OnStreamPart: func(_ context.Context, p StreamPart) { observed = append(observed, p) },
	})

	ch, err := client.Stream(context.Background(), GenerateOptions{})
	if err != nil {
		t.Fatalf("Stream error = %v", err)
	}
	var forwarded []StreamPart
	for p := range ch {
		forwarded = append(forwarded, p)
	}
	if len(observed) != len(parts) {
		t.Fatalf("observed %d parts, want %d", len(observed), len(parts))
	}
	if len(forwarded) != len(parts) {
		t.Fatalf("forwarded %d parts, want %d", len(forwarded), len(parts))
	}
}

// GeneratorFuncShim adapts a function to Client for tests (Stream unsupported).
type GeneratorFuncShim func(ctx context.Context, opts GenerateOptions) (*GenerateResult, error)

func (f GeneratorFuncShim) Generate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	return f(ctx, opts)
}

func (GeneratorFuncShim) Stream(ctx context.Context, opts GenerateOptions) (<-chan StreamPart, error) {
	return nil, ErrStreamNotImplemented
}
