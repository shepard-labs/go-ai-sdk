// Package r2 provides a Cloudflare R2 RunStore backed by go-clients storage.
// Each run is stored as an object at runs/{id}.json, matching the GCS backend's
// layout.
package r2

import (
	"context"
	"errors"
	"fmt"

	"github.com/shepard-labs/go-ai-sdk/llm/store"
	clientstorage "github.com/shepard-labs/go-clients/storage"
)

// Store is an R2-backed RunStore.
type Store struct {
	storage clientstorage.Storage
}

// New returns an R2 store backed by the given storage client. The caller owns
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
		return nil, fmt.Errorf("r2 store: download: %w", err)
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
		return fmt.Errorf("r2 store: upload: %w", err)
	}
	return nil
}

// Delete removes the run object. S3 DeleteObject is idempotent, so deleting an
// absent run is a no-op.
func (s *Store) Delete(ctx context.Context, runID string) error {
	if err := s.storage.Delete(ctx, objectKey(runID)); err != nil && !errors.Is(err, clientstorage.ErrObjectNotFound) {
		return fmt.Errorf("r2 store: delete: %w", err)
	}
	return nil
}
