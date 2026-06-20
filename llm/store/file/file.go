// Package file provides a RunStore that persists each run as a JSON file in a
// directory.
package file

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/shepard-labs/go-ai-sdk/llm/store"
)

// Store persists run state as one JSON file per run ID under Dir.
type Store struct {
	dir string
}

// New returns a file-backed store writing to dir, creating it if necessary.
func New(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("file store: create dir: %w", err)
	}
	return &Store{dir: dir}, nil
}

func (s *Store) path(runID string) (string, error) {
	// Guard against path traversal: run IDs must be a single path element.
	if runID == "" || runID != filepath.Base(runID) || runID == "." || runID == ".." {
		return "", fmt.Errorf("file store: invalid run ID %q", runID)
	}
	return filepath.Join(s.dir, runID+".json"), nil
}

// Load reads the run state file, returning (nil, nil) if it does not exist.
func (s *Store) Load(ctx context.Context, runID string) (*store.RunState, error) {
	path, err := s.path(runID)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("file store: read: %w", err)
	}
	return store.UnmarshalState(data)
}

// Save writes the run state atomically (write to temp file, then rename).
func (s *Store) Save(ctx context.Context, state *store.RunState) error {
	path, err := s.path(state.ID)
	if err != nil {
		return err
	}
	data, err := store.MarshalState(state)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.dir, state.ID+".*.tmp")
	if err != nil {
		return fmt.Errorf("file store: temp: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("file store: write: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("file store: close: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("file store: rename: %w", err)
	}
	return nil
}

// Delete removes the run state file. Deleting an absent run is a no-op.
func (s *Store) Delete(ctx context.Context, runID string) error {
	path, err := s.path(runID)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("file store: delete: %w", err)
	}
	return nil
}
