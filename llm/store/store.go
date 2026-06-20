// Package store defines the RunStore interface for persisting and resuming
// agent conversation state across process boundaries. Concrete backends live in
// subpackages (memory, file) and submodules (postgres, gcs, r2).
package store

import (
	"context"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// RunState is the persisted state of a single agent run.
type RunState struct {
	ID       string
	Messages []llm.Message
	Metadata map[string]string
}

// RunStore persists agent run state. Load returns (nil, nil) when no run exists
// for the given ID.
type RunStore interface {
	Load(ctx context.Context, runID string) (*RunState, error)
	Save(ctx context.Context, state *RunState) error
	Delete(ctx context.Context, runID string) error
}
