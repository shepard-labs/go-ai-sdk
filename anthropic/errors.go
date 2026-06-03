package anthropic

import "fmt"

type APIError struct {
	Type    string
	Message string
}

func (e APIError) Error() string {
	if e.Type == "" {
		return e.Message
	}
	return e.Type + ": " + e.Message
}

type APICallError struct {
	Message   string
	Status    int
	Retryable bool
	Cause     error
}

func (e *APICallError) Error() string {
	if e == nil {
		return ""
	}
	if e.Status != 0 {
		return fmt.Sprintf("anthropic: API call failed with status %d: %s", e.Status, e.Message)
	}
	return "anthropic: API call failed: " + e.Message
}

func (e *APICallError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

type Logger interface {
	Debug(msg string, keysAndValues ...any)
	Info(msg string, keysAndValues ...any)
	Warn(msg string, keysAndValues ...any)
	Error(msg string, keysAndValues ...any)
}

type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
