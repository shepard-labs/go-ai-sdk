package gcs

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/store"
	clientstorage "github.com/shepard-labs/go-clients/storage"
)

// Compile-time check that *Store satisfies the RunStore interface. Behavioral
// tests use a fake storage backend.
var _ store.RunStore = (*Store)(nil)

type fakeStorage struct {
	objects      map[string][]byte
	contentTypes map[string]string
	deleted      []string
	err          error
}

func newFakeStorage() *fakeStorage {
	return &fakeStorage{
		objects:      make(map[string][]byte),
		contentTypes: make(map[string]string),
	}
}

func (f *fakeStorage) Upload(ctx context.Context, objectName string, content []byte, contentType string) error {
	if f.err != nil {
		return f.err
	}
	f.objects[objectName] = append([]byte(nil), content...)
	f.contentTypes[objectName] = contentType
	return nil
}

func (f *fakeStorage) UploadReader(ctx context.Context, objectName string, r io.Reader, contentType string, size int64) error {
	panic("not implemented")
}

func (f *fakeStorage) Download(ctx context.Context, objectName string) ([]byte, error) {
	if f.err != nil {
		return nil, f.err
	}
	content, ok := f.objects[objectName]
	if !ok {
		return nil, clientstorage.ErrObjectNotFound
	}
	return append([]byte(nil), content...), nil
}

func (f *fakeStorage) Delete(ctx context.Context, objectName string) error {
	if f.err != nil {
		return f.err
	}
	f.deleted = append(f.deleted, objectName)
	delete(f.objects, objectName)
	return nil
}

func (f *fakeStorage) Close() error { return nil }

func TestSaveUploadsRunState(t *testing.T) {
	backend := newFakeStorage()
	s := New(backend)
	state := sampleState()

	if err := s.Save(context.Background(), state); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	key := "runs/run-1.json"
	if _, ok := backend.objects[key]; !ok {
		t.Fatalf("expected object %q to be uploaded", key)
	}
	if got := backend.contentTypes[key]; got != "application/json" {
		t.Fatalf("content type = %q, want application/json", got)
	}
}

func TestLoadDownloadsRunState(t *testing.T) {
	backend := newFakeStorage()
	s := New(backend)
	want := sampleState()
	if err := s.Save(context.Background(), want); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	got, err := s.Load(context.Background(), want.ID)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Load = %#v, want %#v", got, want)
	}
}

func TestLoadMissingRunReturnsNil(t *testing.T) {
	s := New(newFakeStorage())

	got, err := s.Load(context.Background(), "missing")
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	if got != nil {
		t.Fatalf("Load = %#v, want nil", got)
	}
}

func TestDeleteUsesRunObjectKey(t *testing.T) {
	backend := newFakeStorage()
	s := New(backend)

	if err := s.Delete(context.Background(), "run-1"); err != nil {
		t.Fatalf("Delete returned error: %v", err)
	}
	if !reflect.DeepEqual(backend.deleted, []string{"runs/run-1.json"}) {
		t.Fatalf("deleted = %#v, want runs/run-1.json", backend.deleted)
	}
}

func TestDeleteIgnoresMissingRun(t *testing.T) {
	backend := newFakeStorage()
	backend.err = clientstorage.ErrObjectNotFound
	s := New(backend)

	if err := s.Delete(context.Background(), "missing"); err != nil {
		t.Fatalf("Delete returned error: %v", err)
	}
}

func TestErrorsWrapStorageFailures(t *testing.T) {
	backend := newFakeStorage()
	backend.err = errors.New("boom")
	s := New(backend)

	if _, err := s.Load(context.Background(), "run-1"); err == nil {
		t.Fatal("Load returned nil error")
	}
	if err := s.Save(context.Background(), sampleState()); err == nil {
		t.Fatal("Save returned nil error")
	}
	if err := s.Delete(context.Background(), "run-1"); err == nil {
		t.Fatal("Delete returned nil error")
	}
}

func sampleState() *store.RunState {
	return &store.RunState{
		ID: "run-1",
		Messages: []llm.Message{{
			Role:    "user",
			Content: []llm.Content{llm.TextContent{Text: "hello"}},
		}},
		Metadata: map[string]string{"k": "v"},
	}
}
