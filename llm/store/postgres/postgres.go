// Package postgres provides a Postgres-backed RunStore using pgx.
//
// Schema:
//
//	CREATE TABLE agent_runs (
//	    id          TEXT PRIMARY KEY,
//	    messages    JSONB NOT NULL,
//	    metadata    JSONB,
//	    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
//	);
//
// The messages column stores the full serialized run state (the store codec
// output), which round-trips llm.Content interface values. The metadata column
// stores the run metadata map for queryability.
package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/shepard-labs/go-ai-sdk/llm/store"
)

// CreateTableSQL is the DDL for the agent_runs table.
const CreateTableSQL = `CREATE TABLE IF NOT EXISTS agent_runs (
    id          TEXT PRIMARY KEY,
    messages    JSONB NOT NULL,
    metadata    JSONB,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);`

// Store is a Postgres-backed RunStore.
type Store struct {
	pool *pgxpool.Pool
}

// New returns a Postgres store backed by the given connection pool. The caller
// owns the pool's lifecycle.
func New(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool}
}

// Migrate creates the agent_runs table if it does not exist.
func (s *Store) Migrate(ctx context.Context) error {
	if _, err := s.pool.Exec(ctx, CreateTableSQL); err != nil {
		return fmt.Errorf("postgres store: migrate: %w", err)
	}
	return nil
}

// Load reads the run state, returning (nil, nil) if absent.
func (s *Store) Load(ctx context.Context, runID string) (*store.RunState, error) {
	var blob []byte
	err := s.pool.QueryRow(ctx, `SELECT messages FROM agent_runs WHERE id = $1`, runID).Scan(&blob)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("postgres store: load: %w", err)
	}
	return store.UnmarshalState(blob)
}

// Save upserts the run state.
func (s *Store) Save(ctx context.Context, state *store.RunState) error {
	blob, err := store.MarshalState(state)
	if err != nil {
		return err
	}
	metadata, err := json.Marshal(state.Metadata)
	if err != nil {
		return fmt.Errorf("postgres store: marshal metadata: %w", err)
	}
	_, err = s.pool.Exec(ctx,
		`INSERT INTO agent_runs (id, messages, metadata, updated_at)
         VALUES ($1, $2, $3, NOW())
         ON CONFLICT (id) DO UPDATE SET messages = EXCLUDED.messages, metadata = EXCLUDED.metadata, updated_at = NOW()`,
		state.ID, blob, metadata)
	if err != nil {
		return fmt.Errorf("postgres store: save: %w", err)
	}
	return nil
}

// Delete removes the run state. Deleting an absent run is a no-op.
func (s *Store) Delete(ctx context.Context, runID string) error {
	if _, err := s.pool.Exec(ctx, `DELETE FROM agent_runs WHERE id = $1`, runID); err != nil {
		return fmt.Errorf("postgres store: delete: %w", err)
	}
	return nil
}
