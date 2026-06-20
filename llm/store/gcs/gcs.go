// Package gcs provides a Google Cloud Storage-backed RunStore backed by
// go-clients storage. Each run is stored as a single object at runs/{id}.json.
package gcs

import (
	"context"
	"errors"
	"fmt"

	"github.com/shepard-labs/go-ai-sdk/llm/store"
	clientstorage "github.com/shepard-labs/go-clients/storage"
)

// Store is a GCS-backed RunStore.
type Store struct {
	storage clientstorage.Storage
}

// New returns a GCS store backed by the given storage client. The caller owns
// the client's lifecycle.
func New(storage clientstorage.Storage) *Store {
	return &Store{storage: storage}
}

func objectKey(runID string) string {
	return "runs/" + runID + ".json"
}

// Load reads the run object, returning (nil, nil) if it does not exist.
func (s *Store) Load(ctx context.Context, runID string) (*store.RunState, error) {
	data, err := s.storage.Download(ctx, objectKey(runID))
	if err != nil {
		if errors.Is(err, clientstorage.ErrObjectNotFound) {
			return nil, nil
		}
		return nil, fmt.Errorf("gcs store: download: %w", err)
	}
	return store.UnmarshalState(data)
}

// Save writes the run object.
func (s *Store) Save(ctx context.Context, state *store.RunState) error {
	data, err := store.MarshalState(state)
	if err != nil {
		return err
	}
	if err := s.storage.Upload(ctx, objectKey(state.ID), data, "application/json"); err != nil {
		return fmt.Errorf("gcs store: upload: %w", err)
	}
	return nil
}

// Delete removes the run object. Deleting an absent run is a no-op.
func (s *Store) Delete(ctx context.Context, runID string) error {
	if err := s.storage.Delete(ctx, objectKey(runID)); err != nil && !errors.Is(err, clientstorage.ErrObjectNotFound) {
		return fmt.Errorf("gcs store: delete: %w", err)
	}
	return nil
}
