package internal

import anthropic "github.com/shepard-labs/go-ai-sdk/anthropic"

type MessagesRequest struct {
	Model                  string
	Messages               []anthropic.Message
	System                 any
	MaxTokens              int
	Metadata               *Metadata
	StopSequences          []string
	Stream                 bool
	Temperature            *float64
	TopK                   *int
	TopP                   *float64
	Tools                  []Tool
	ToolChoice             *ToolChoice
	Thinking               *ThinkingRequest
	Container              *anthropic.Container
	MCPServers             []anthropic.MCPServer
	Speed                  string
	InferenceGeo           string
	OutputConfig           *OutputConfig
	StructuredOutput       *StructuredOutput
	ContextManagement      *anthropic.ContextManagement
	DisableParallelToolUse bool
}

type OutputConfig = anthropic.OutputConfig
type StructuredOutput = anthropic.StructuredOutput

type ThinkingRequest struct {
	Type         string
	BudgetTokens int
}

type Metadata = anthropic.Metadata
type ToolChoice = anthropic.ToolChoice
type Tool = anthropic.Tool

type APIError struct {
	Type    string
	Message string
}
