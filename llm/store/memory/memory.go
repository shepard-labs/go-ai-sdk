// Package memory provides an in-memory RunStore, primarily for tests and
// single-process use. It is safe for concurrent use.
package memory

import (
	"context"
	"sync"

	"github.com/shepard-labs/go-ai-sdk/llm/store"
)

// Store is an in-memory RunStore backed by a map guarded by a mutex.
type Store struct {
	mu   sync.RWMutex
	runs map[string][]byte
}

// New returns an empty in-memory store.
func New() *Store {
	return &Store{runs: make(map[string][]byte)}
}

// Load returns the stored run state, or (nil, nil) if absent.
func (s *Store) Load(ctx context.Context, runID string) (*store.RunState, error) {
	s.mu.RLock()
	data, ok := s.runs[runID]
	s.mu.RUnlock()
	if !ok {
		return nil, nil
	}
	return store.UnmarshalState(data)
}

// Save persists the run state, replacing any existing state for the same ID.
func (s *Store) Save(ctx context.Context, state *store.RunState) error {
	data, err := store.MarshalState(state)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.runs[state.ID] = data
	s.mu.Unlock()
	return nil
}

// Delete removes the run state for the given ID. Deleting an absent run is a
// no-op.
func (s *Store) Delete(ctx context.Context, runID string) error {
	s.mu.Lock()
	delete(s.runs, runID)
	s.mu.Unlock()
	return nil
}
