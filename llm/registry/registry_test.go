package registry

import (
	"context"
	"sync"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

type stubClient struct{ id string }

func (stubClient) Generate(ctx context.Context, opts llm.GenerateOptions) (*llm.GenerateResult, error) {
	return nil, nil
}

func (stubClient) Stream(ctx context.Context, opts llm.GenerateOptions) (<-chan llm.StreamPart, error) {
	return nil, llm.ErrStreamNotImplemented
}

func TestRegisterAndNewClient(t *testing.T) {
	resetRegistry()
	var gotModelID string
	var gotOpts ProviderOptions
	Register("test", func(modelID string, opts ProviderOptions) (llm.Client, error) {
		gotModelID = modelID
		gotOpts = opts
		return stubClient{id: modelID}, nil
	})
	client, err := NewClient("test:my-model", ProviderOptions{APIKey: "k", BaseURL: "http://x"})
	if err != nil {
		t.Fatalf("NewClient error = %v", err)
	}
	if gotModelID != "my-model" {
		t.Fatalf("modelID = %q, want my-model", gotModelID)
	}
	if gotOpts.APIKey != "k" || gotOpts.BaseURL != "http://x" {
		t.Fatalf("opts = %#v", gotOpts)
	}
	if sc, ok := client.(stubClient); !ok || sc.id != "my-model" {
		t.Fatalf("client = %#v", client)
	}
}

func TestNewClientModelIDWithColons(t *testing.T) {
	resetRegistry()
	var gotModelID string
	Register("openrouter", func(modelID string, opts ProviderOptions) (llm.Client, error) {
		gotModelID = modelID
		return stubClient{}, nil
	})
	if _, err := NewClient("openrouter:openai/gpt-4o:free", ProviderOptions{}); err != nil {
		t.Fatalf("NewClient error = %v", err)
	}
	if gotModelID != "openai/gpt-4o:free" {
		t.Fatalf("modelID = %q, want openai/gpt-4o:free", gotModelID)
	}
}

func TestNewClientUnregisteredProvider(t *testing.T) {
	resetRegistry()
	_, err := NewClient("ghost:model", ProviderOptions{})
	if err == nil {
		t.Fatal("expected error for unregistered provider")
	}
	if got := err.Error(); !contains(got, "ghost") || !contains(got, "blank-import") {
		t.Fatalf("error = %q, want mention of provider name and blank-import", got)
	}
}

func TestNewClientMalformedModel(t *testing.T) {
	resetRegistry()
	for _, model := range []string{"noseparator", "anthropic:", ":model", ""} {
		if _, err := NewClient(model, ProviderOptions{}); err == nil {
			t.Fatalf("expected error for malformed model %q", model)
		}
	}
}

func TestRegisterDuplicatePanics(t *testing.T) {
	resetRegistry()
	Register("dup", func(string, ProviderOptions) (llm.Client, error) { return stubClient{}, nil })
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on duplicate registration")
		}
	}()
	Register("dup", func(string, ProviderOptions) (llm.Client, error) { return stubClient{}, nil })
}

func TestRegisterConcurrent(t *testing.T) {
	resetRegistry()
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			Register(string(rune('a'+i%26))+string(rune('0'+i/26)), func(string, ProviderOptions) (llm.Client, error) {
				return stubClient{}, nil
			})
		}(i)
	}
	wg.Wait()
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// resetRegistry clears all registrations between tests.
func resetRegistry() {
	mu.Lock()
	defer mu.Unlock()
	factories = make(map[string]ProviderFactory)
}
