package adapters

import "errors"

// ErrCohereNotImplemented is returned by the Cohere adapter. The
// adapter is a stub pending translation of the cohere.Provider's
// LanguageModel/DoGenerate signature into a RouterOptions adapter.
var ErrCohereNotImplemented = errors.New("ai: cohere adapter not yet implemented")
