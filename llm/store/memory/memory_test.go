package memory

import (
	"context"
	"sync"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/store"
)

var _ store.RunStore = (*Store)(nil)

func TestLoadMissingReturnsNil(t *testing.T) {
	s := New()
	got, err := s.Load(context.Background(), "absent")
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	if got != nil {
		t.Fatalf("Load = %#v, want nil", got)
	}
}

func TestSaveLoadDelete(t *testing.T) {
	s := New()
	ctx := context.Background()
	state := &store.RunState{ID: "r1", Messages: []llm.Message{{Role: "user", Content: []llm.Content{llm.TextContent{Text: "hi"}}}}}
	if err := s.Save(ctx, state); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	got, err := s.Load(ctx, "r1")
	if err != nil || got == nil {
		t.Fatalf("Load = %#v, err = %v", got, err)
	}
	if got.Messages[0].Content[0].(llm.TextContent).Text != "hi" {
		t.Fatalf("loaded = %#v", got)
	}
	if err := s.Delete(ctx, "r1"); err != nil {
		t.Fatalf("Delete error = %v", err)
	}
	if got, _ := s.Load(ctx, "r1"); got != nil {
		t.Fatal("expected nil after delete")
	}
}

func TestStoredStateIsIsolated(t *testing.T) {
	s := New()
	ctx := context.Background()
	state := &store.RunState{ID: "r1", Messages: []llm.Message{{Role: "user", Content: []llm.Content{llm.TextContent{Text: "orig"}}}}}
	if err := s.Save(ctx, state); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	state.Messages[0].Role = "mutated" // mutate caller's copy
	got, _ := s.Load(ctx, "r1")
	if got.Messages[0].Role != "user" {
		t.Fatalf("stored state mutated by caller: role = %q", got.Messages[0].Role)
	}
}

func TestConcurrentAccess(t *testing.T) {
	s := New()
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := string(rune('a' + i%26))
			_ = s.Save(ctx, &store.RunState{ID: id, Messages: []llm.Message{{Role: "user"}}})
			_, _ = s.Load(ctx, id)
			if i%3 == 0 {
				_ = s.Delete(ctx, id)
			}
		}(i)
	}
	wg.Wait()
}
