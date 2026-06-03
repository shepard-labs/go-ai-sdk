package anthropic

import "encoding/json"

type apiMessage struct {
	Role    string            `json:"role"`
	Content []apiContentBlock `json:"content"`
}

type apiContentBlock struct {
	Type             string           `json:"type"`
	Text             string           `json:"text,omitempty"`
	Thinking         string           `json:"thinking,omitempty"`
	Signature        string           `json:"signature,omitempty"`
	Data             string           `json:"data,omitempty"`
	ID               string           `json:"id,omitempty"`
	Name             string           `json:"name,omitempty"`
	Input            any              `json:"input,omitempty"`
	Content          any              `json:"content,omitempty"`
	ToolUseID        string           `json:"tool_use_id,omitempty"`
	IsError          bool             `json:"is_error,omitempty"`
	ServerName       string           `json:"server_name,omitempty"`
	Source           *apiBlockSource  `json:"source,omitempty"`
	CacheControl     *CacheControl    `json:"cache_control,omitempty"`
	Citations        []Citation       `json:"citations,omitempty"`
	ErrorCode        string           `json:"error_code,omitempty"`
	RetrievedAt      string           `json:"retrieved_at,omitempty"`
	EncryptedContent string           `json:"encrypted_content,omitempty"`
	PageAge          string           `json:"page_age,omitempty"`
	Stdout           string           `json:"stdout,omitempty"`
	EncryptedStdout  string           `json:"encrypted_stdout,omitempty"`
	Stderr           string           `json:"stderr,omitempty"`
	ReturnCode       int              `json:"return_code,omitempty"`
	ToolReferences   []ToolReference  `json:"tool_references,omitempty"`
	URL              string           `json:"url,omitempty"`
	Title            string           `json:"title,omitempty"`
	MediaType        string           `json:"media_type,omitempty"`
	Filename         string           `json:"filename,omitempty"`
	ProviderExecuted bool             `json:"provider_executed,omitempty"`
	Dynamic          bool             `json:"dynamic,omitempty"`
	ProviderMetadata ProviderMetadata `json:"provider_metadata,omitempty"`
}

type apiBlockSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

func ConvertPrompt(messages []Message) []Message { return messages }

func convertPrompt(messages []Message) ([]apiContentBlock, []apiMessage) {
	var system []apiContentBlock
	var converted []apiMessage
	for _, message := range messages {
		switch msg := message.(type) {
		case SystemMessage:
			system = append(system, apiContentBlock{Type: "text", Text: msg.Content, CacheControl: msg.CacheControl})
		case UserMessage:
			converted = append(converted, apiMessage{Role: "user", Content: convertUserContent(msg.Content)})
		case AssistantMessage:
			converted = append(converted, apiMessage{Role: "assistant", Content: convertAssistantContent(msg.Content)})
		}
	}
	return system, converted
}

func convertUserContent(contents []UserContent) []apiContentBlock {
	blocks := make([]apiContentBlock, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case TextContent:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: c.Text})
		case CacheControlledTextContent:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: c.Text, CacheControl: c.CacheControl})
		case ImageContent:
			blocks = append(blocks, apiContentBlock{Type: "image", Source: imageSourceToAPI(c.Source)})
		case CacheControlledImageContent:
			blocks = append(blocks, apiContentBlock{Type: "image", Source: imageSourceToAPI(c.Source), CacheControl: c.CacheControl})
		case DocumentContent:
			blocks = append(blocks, apiContentBlock{Type: "document", Source: documentSourceToAPI(c.Source)})
		case CacheControlledDocumentContent:
			blocks = append(blocks, apiContentBlock{Type: "document", Source: documentSourceToAPI(c.Source), CacheControl: c.CacheControl})
		case ToolResultContent:
			blocks = append(blocks, apiContentBlock{Type: "tool_result", ToolUseID: c.ToolCallID, Content: convertToolResultParts(c.Result), IsError: c.IsError})
		}
	}
	return blocks
}

func convertAssistantContent(contents []AssistantContent) []apiContentBlock {
	blocks := make([]apiContentBlock, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case TextContent:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: c.Text})
		case CacheControlledTextContent:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: c.Text, CacheControl: c.CacheControl})
		case ThinkingContent:
			blocks = append(blocks, apiContentBlock{Type: "thinking", Thinking: c.Thinking, Signature: c.Signature})
		case RedactedThinkingContent:
			blocks = append(blocks, apiContentBlock{Type: "redacted_thinking", Data: c.Data})
		case CompactionContent:
			blocks = append(blocks, apiContentBlock{Type: "compaction", Text: c.Text})
		case ToolCallContent:
			blocks = append(blocks, apiContentBlock{Type: "tool_use", ID: c.ToolCallID, Name: c.ToolName, Input: rawJSONToAny(c.Input), ProviderMetadata: c.ProviderMetadata})
		case ServerToolUseContent:
			blocks = append(blocks, apiContentBlock{Type: "server_tool_use", ID: c.ID, Name: c.Name, Input: rawJSONToAny(c.Input), ProviderMetadata: c.ProviderMetadata})
		case MCPToolUseContent:
			blocks = append(blocks, apiContentBlock{Type: "mcp_tool_use", ID: c.ID, Name: c.Name, ServerName: c.ServerName, Input: rawJSONToAny(c.Input), ProviderMetadata: c.ProviderMetadata})
		}
	}
	return blocks
}

func convertToolResultParts(parts []ToolResultPart) []apiContentBlock {
	blocks := make([]apiContentBlock, 0, len(parts))
	for _, part := range parts {
		switch p := part.(type) {
		case ToolResultText:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: p.Text})
		case ToolResultImage:
			blocks = append(blocks, apiContentBlock{Type: "image", Source: imageSourceToAPI(p.Source)})
		case ToolResultDocument:
			blocks = append(blocks, apiContentBlock{Type: "document", Source: documentSourceToAPI(p.Source)})
		case ToolResultReference:
			blocks = append(blocks, apiContentBlock{Type: "content_reference", ID: p.ID})
		}
	}
	return blocks
}

func rawJSONToAny(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return string(raw)
	}
	return value
}

func imageSourceToAPI(source ImageSource) *apiBlockSource {
	return &apiBlockSource{Type: valueOrDefault(source.Type, "base64"), MediaType: source.MediaType, Data: source.Data, URL: source.URL}
}

func documentSourceToAPI(source DocumentSource) *apiBlockSource {
	return &apiBlockSource{Type: valueOrDefault(source.Type, "base64"), MediaType: source.MediaType, Data: source.Data, URL: source.URL}
}
