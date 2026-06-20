package postgres

import (
	"github.com/shepard-labs/go-ai-sdk/llm/store"
)

// Compile-time check that *Store satisfies the RunStore interface. Behavioral
// tests require a live Postgres instance and run as integration tests.
var _ store.RunStore = (*Store)(nil)
