package openai

import "github.com/shepard-labs/go-ai-sdk/openaicompatible"

// ErrMissingAPIKey is declared in openai.go; it is duplicated here as a
// documentation aid for the rest of the package.

// UnsupportedFunctionalityError is returned when the caller requests a
// capability the openai package does not support.
type UnsupportedFunctionalityError = openaicompatible.UnsupportedFunctionalityError

// InvalidPromptError is returned when the prompt structure is invalid for
// the requested model type.
type InvalidPromptError = openaicompatible.InvalidPromptError

// InvalidResponseDataError is returned when the provider returns a response
// that cannot be parsed.
type InvalidResponseDataError = openaicompatible.InvalidResponseDataError

// APIError is a parsed OpenAI API error object.
type APIError = openaicompatible.APIError

// APICallError is a structured OpenAI API call failure.
type APICallError = openaicompatible.APICallError

// TooManyEmbeddingValuesForCallError is returned when the caller supplies
// more embedding input values than the model's MaxEmbeddingsPerCall allows.
type TooManyEmbeddingValuesForCallError = openaicompatible.TooManyEmbeddingValuesForCallError
