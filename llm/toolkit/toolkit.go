// Package toolkit provides ready-made, scoped tool sets for common agent tasks:
// filesystem access, shell command execution, and read-only git inspection.
// Each toolkit implements llm.ToolDispatcher and exposes its tool schemas via
// Tools(). Toolkits use only the standard library.
package toolkit

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// Toolkit is a named set of tools with a dispatcher.
type Toolkit interface {
	llm.ToolDispatcher
	// Tools returns the tool schemas this toolkit handles.
	Tools() []llm.Tool
}

// Tools flattens the tool lists of several toolkits into one slice.
func Tools(toolkits ...Toolkit) []llm.Tool {
	var all []llm.Tool
	for _, tk := range toolkits {
		all = append(all, tk.Tools()...)
	}
	return all
}

// merged dispatches tool calls to whichever toolkit owns the named tool.
type merged struct {
	routes map[string]llm.ToolDispatcher
}

// Merge combines toolkits into a single dispatcher. It panics if two toolkits
// declare the same tool name.
func Merge(toolkits ...Toolkit) llm.ToolDispatcher {
	routes := make(map[string]llm.ToolDispatcher)
	for _, tk := range toolkits {
		for _, tool := range tk.Tools() {
			if _, exists := routes[tool.Name]; exists {
				panic(fmt.Sprintf("toolkit: duplicate tool name %q", tool.Name))
			}
			routes[tool.Name] = tk
		}
	}
	return &merged{routes: routes}
}

// Dispatch routes a call to the toolkit that declared the tool.
func (m *merged) Dispatch(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	dispatcher, ok := m.routes[name]
	if !ok {
		return nil, fmt.Errorf("toolkit: unknown tool %q", name)
	}
	return dispatcher.Dispatch(ctx, name, input)
}

// jsonResult marshals a value to a json.RawMessage tool result.
func jsonResult(v any) (json.RawMessage, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("toolkit: marshal result: %w", err)
	}
	return data, nil
}
