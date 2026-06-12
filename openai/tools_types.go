package openai

// ApplyPatchArgs is the configuration for the apply_patch provider tool.
type ApplyPatchArgs struct{}

// CustomToolArgs is the configuration for the custom provider tool.
type CustomToolArgs struct {
	Description string            `json:"description,omitempty"`
	Format      *CustomToolFormat `json:"format,omitempty"`
	Name        string            `json:"name,omitempty"`
}

// CustomToolFormat specifies a custom tool's grammar/text format.
type CustomToolFormat struct {
	Type       string `json:"type,omitempty"`
	Syntax     string `json:"syntax,omitempty"`
	Definition string `json:"definition,omitempty"`
}

// CodeInterpreterArgs is the configuration for the code_interpreter provider tool.
type CodeInterpreterArgs struct {
	Container *CodeInterpreterContainer `json:"container,omitempty"`
}

// CodeInterpreterContainer identifies a container for code_interpreter.
type CodeInterpreterContainer struct {
	Type    string   `json:"type,omitempty"`
	FileIDs []string `json:"file_ids,omitempty"`
}

// FileSearchArgs is the configuration for the file_search provider tool.
type FileSearchArgs struct {
	VectorStoreIDs []string           `json:"vectorStoreIDs,omitempty"`
	MaxNumResults  *int               `json:"maxNumResults,omitempty"`
	Ranking        *FileSearchRanking `json:"ranking,omitempty"`
	Filters        any                `json:"filters,omitempty"`
}

// FileSearchRanking configures file_search ranking options.
type FileSearchRanking struct {
	Ranker         *string  `json:"ranker,omitempty"`
	ScoreThreshold *float64 `json:"scoreThreshold,omitempty"`
}

// ImageGenerationArgs is the configuration for the image_generation provider tool.
type ImageGenerationArgs struct {
	Background        *string            `json:"background,omitempty"`
	InputFidelity     *string            `json:"inputFidelity,omitempty"`
	InputImageMask    *ImageGenInputMask `json:"inputImageMask,omitempty"`
	Model             *string            `json:"model,omitempty"`
	Moderation        *string            `json:"moderation,omitempty"`
	OutputCompression *int               `json:"outputCompression,omitempty"`
	OutputFormat      *string            `json:"outputFormat,omitempty"`
	PartialImages     *int               `json:"partialImages,omitempty"`
	Quality           *string            `json:"quality,omitempty"`
	Size              *string            `json:"size,omitempty"`
}

// ImageGenInputMask is a mask for image_generation.
type ImageGenInputMask struct {
	FileID   *string `json:"fileId,omitempty"`
	ImageURL *string `json:"imageUrl,omitempty"`
}

// LocalShellArgs is the configuration for the local_shell provider tool.
type LocalShellArgs struct{}

// ShellArgs is the configuration for the shell provider tool.
type ShellArgs struct {
	Environment *ShellEnvironment `json:"environment,omitempty"`
}

// ShellEnvironment is the shell environment configuration.
type ShellEnvironment struct {
	Type          string              `json:"type,omitempty"`
	FileIDs       []string            `json:"fileIds,omitempty"`
	MemoryLimit   *string             `json:"memoryLimit,omitempty"`
	NetworkPolicy *ShellNetworkPolicy `json:"networkPolicy,omitempty"`
	Skills        []ShellSkill        `json:"skills,omitempty"`
	ContainerID   *string             `json:"containerId,omitempty"`
	LocalSkills   []ShellLocalSkill   `json:"localSkills,omitempty"`
}

// ShellNetworkPolicy is the network policy for a shell environment.
type ShellNetworkPolicy struct {
	Type           string   `json:"type,omitempty"`
	AllowedDomains []string `json:"allowedDomains,omitempty"`
	DeniedDomains  []string `json:"deniedDomains,omitempty"`
	BlockedDomains []string `json:"blockedDomains,omitempty"`
}

// ShellSkill is a reference to a remote skill available to a shell.
type ShellSkill struct {
	Type              string            `json:"type,omitempty"`
	ProviderReference ProviderReference `json:"providerReference,omitempty"`
	Name              string            `json:"name,omitempty"`
	Description       string            `json:"description,omitempty"`
}

// ShellLocalSkill is a local skill attached to a local shell environment.
type ShellLocalSkill struct {
	Type string `json:"type,omitempty"`
	Name string `json:"name,omitempty"`
}

// WebSearchArgs is the configuration for the web_search provider tool.
type WebSearchArgs struct {
	ExternalWebAccess *bool                  `json:"externalWebAccess,omitempty"`
	Filters           *WebSearchFilters      `json:"filters,omitempty"`
	SearchContextSize *string                `json:"searchContextSize,omitempty"`
	UserLocation      *WebSearchUserLocation `json:"userLocation,omitempty"`
}

// WebSearchFilters restricts web_search to specific allowed domains.
type WebSearchFilters struct {
	AllowedDomains []string `json:"allowedDomains,omitempty"`
}

// WebSearchUserLocation hints at the user's location for web_search.
type WebSearchUserLocation struct {
	Type     string `json:"type,omitempty"`
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// WebSearchPreviewArgs is the configuration for the web_search_preview provider tool.
type WebSearchPreviewArgs struct {
	SearchContextSize *string                `json:"searchContextSize,omitempty"`
	UserLocation      *WebSearchUserLocation `json:"userLocation,omitempty"`
}

// MCPArgs is the configuration for the MCP provider tool.
type MCPArgs struct {
	ServerLabel       string              `json:"serverLabel"`
	AllowedTools      any                 `json:"allowedTools,omitempty"`
	Authorization     *string             `json:"authorization,omitempty"`
	ConnectorID       *string             `json:"connectorId,omitempty"`
	Headers           map[string]string   `json:"headers,omitempty"`
	RequireApproval   *MCPRequireApproval `json:"requireApproval,omitempty"`
	ServerDescription *string             `json:"serverDescription,omitempty"`
	ServerURL         *string             `json:"serverUrl,omitempty"`
}

// MCPRequireApproval describes when the MCP tool should request approval.
type MCPRequireApproval struct {
	Always    *bool    `json:"always,omitempty"`
	Never     *bool    `json:"never,omitempty"`
	ToolNames []string `json:"tool_names,omitempty"`
}

// ToolSearchArgs is the configuration for the tool_search provider tool.
type ToolSearchArgs struct {
	Execution   *string `json:"execution,omitempty"`
	Description *string `json:"description,omitempty"`
	Parameters  any     `json:"parameters,omitempty"`
}
