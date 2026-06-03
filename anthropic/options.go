package anthropic

type ModelOptions struct {
	SendReasoning          bool
	StructuredOutputMode   StructuredOutputMode
	Thinking               *ThinkingConfig
	DisableParallelToolUse bool
	CacheControl           *CacheControl
	Metadata               *Metadata
	MCPServers             []MCPServer
	Container              *Container
	ToolStreaming          *bool
	Effort                 string
	TaskBudget             *int
	Speed                  string
	InferenceGeo           string
	AnthropicBeta          []string
	ContextManagement      *ContextManagement
	RequestTools           []Tool
}

type ToolOptions struct {
	DeferLoading        *bool
	AllowedCallers      []ToolCallCaller
	EagerInputStreaming *bool
}

type ContextManagement struct {
	Edits []ContextManagementEdit `json:"edits,omitempty"`
}

type ContextManagementEdit interface{ IsContextManagementEdit() }
type ContextManagementEditResponse interface{ IsContextManagementEditResponse() }

type ClearToolUsesEdit struct {
	Type            string                   `json:"type"`
	Trigger         ContextManagementTrigger `json:"trigger,omitempty"`
	Keep            ContextManagementCount   `json:"keep,omitempty"`
	ClearAtLeast    ContextManagementCount   `json:"clear_at_least,omitempty"`
	ClearToolInputs bool                     `json:"clear_tool_inputs,omitempty"`
	ExcludeTools    []string                 `json:"exclude_tools,omitempty"`
}

func (ClearToolUsesEdit) IsContextManagementEdit() {}

type ClearThinkingEdit struct {
	Type string                 `json:"type"`
	Keep ContextManagementCount `json:"keep,omitempty"`
}

func (ClearThinkingEdit) IsContextManagementEdit() {}

type CompactEdit struct {
	Type                 string                   `json:"type"`
	Trigger              ContextManagementTrigger `json:"trigger,omitempty"`
	PauseAfterCompaction bool                     `json:"pause_after_compaction,omitempty"`
	Instructions         string                   `json:"instructions,omitempty"`
}

func (CompactEdit) IsContextManagementEdit() {}

type ContextManagementTrigger interface{ IsContextManagementTrigger() }

type InputTokensTrigger struct {
	Value int `json:"input_tokens"`
}

func (InputTokensTrigger) IsContextManagementTrigger() {}

type ToolUsesTrigger struct {
	Value int `json:"tool_uses"`
}

func (ToolUsesTrigger) IsContextManagementTrigger() {}

type ContextManagementCount interface{ IsContextManagementCount() }

type ToolUsesCount struct {
	Value int `json:"tool_uses"`
}

func (ToolUsesCount) IsContextManagementCount() {}

type InputTokensCount struct {
	Value int `json:"input_tokens"`
}

func (InputTokensCount) IsContextManagementCount() {}

type ThinkingTurnsCount struct {
	Value int `json:"thinking_turns"`
}

func (ThinkingTurnsCount) IsContextManagementCount() {}

type ClearToolUsesResponse struct{ ClearedToolUses int }

func (ClearToolUsesResponse) IsContextManagementEditResponse() {}

type ClearThinkingResponse struct{ ClearedThinkingTurns int }

func (ClearThinkingResponse) IsContextManagementEditResponse() {}

type CompactResponse struct{ Compacted bool }

func (CompactResponse) IsContextManagementEditResponse() {}

type Container struct {
	ID     string  `json:"id,omitempty"`
	Skills []Skill `json:"skills,omitempty"`
}

type Skill struct {
	Type    string `json:"type,omitempty"`
	SkillID string `json:"skill_id,omitempty"`
	Version string `json:"version,omitempty"`
}

type SkillInfo struct {
	Type    string
	SkillID string
	Version string
}

type MCPServer struct {
	Type               string             `json:"type,omitempty"`
	Name               string             `json:"name,omitempty"`
	URL                string             `json:"url,omitempty"`
	AuthorizationToken string             `json:"authorization_token,omitempty"`
	ToolConfiguration  *ToolConfiguration `json:"tool_configuration,omitempty"`
}

type ToolConfiguration struct {
	Enabled      bool     `json:"enabled"`
	AllowedTools []string `json:"allowed_tools,omitempty"`
}
