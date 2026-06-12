package adapters

import "errors"

// ErrGoogleNotImplemented is returned by the Google adapter. The
// adapter is a stub pending translation of the google.Provider's
// LanguageModel/DoGenerate signature into a RouterOptions adapter.
var ErrGoogleNotImplemented = errors.New("ai: google adapter not yet implemented")
