package file

import (
	"context"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/store"
)

var _ store.RunStore = (*Store)(nil)

func TestSaveLoadDelete(t *testing.T) {
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New error = %v", err)
	}
	ctx := context.Background()

	if got, err := s.Load(ctx, "absent"); err != nil || got != nil {
		t.Fatalf("Load absent = %#v, err = %v", got, err)
	}

	state := &store.RunState{
		ID:       "run-1",
		Messages: []llm.Message{{Role: "assistant", Content: []llm.Content{llm.ToolUseContent{ID: "c", Name: "t", Input: []byte(`{}`)}}}},
		Metadata: map[string]string{"k": "v"},
	}
	if err := s.Save(ctx, state); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	got, err := s.Load(ctx, "run-1")
	if err != nil || got == nil {
		t.Fatalf("Load = %#v, err = %v", got, err)
	}
	if got.Metadata["k"] != "v" {
		t.Fatalf("metadata = %#v", got.Metadata)
	}
	if got.Messages[0].Content[0].(llm.ToolUseContent).Name != "t" {
		t.Fatalf("content = %#v", got.Messages[0].Content[0])
	}

	if err := s.Delete(ctx, "run-1"); err != nil {
		t.Fatalf("Delete error = %v", err)
	}
	if got, _ := s.Load(ctx, "run-1"); got != nil {
		t.Fatal("expected nil after delete")
	}
	if err := s.Delete(ctx, "run-1"); err != nil {
		t.Fatalf("Delete absent error = %v", err)
	}
}

func TestRejectsInvalidRunID(t *testing.T) {
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New error = %v", err)
	}
	ctx := context.Background()
	for _, id := range []string{"../escape", "a/b", "", "."} {
		if err := s.Save(ctx, &store.RunState{ID: id}); err == nil {
			t.Fatalf("Save with id %q = nil, want error", id)
		}
		if _, err := s.Load(ctx, id); err == nil {
			t.Fatalf("Load with id %q = nil, want error", id)
		}
	}
}
