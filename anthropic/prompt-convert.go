package anthropic

import "encoding/json"

type apiMessage struct {
	Role    string            `json:"role"`
	Content []apiContentBlock `json:"content"`
}

type apiContentBlock struct {
	Type             string                     `json:"type"`
	Text             string                     `json:"text,omitempty"`
	Thinking         string                     `json:"thinking,omitempty"`
	Signature        string                     `json:"signature,omitempty"`
	Data             string                     `json:"data,omitempty"`
	ID               string                     `json:"id,omitempty"`
	Name             string                     `json:"name,omitempty"`
	Input            any                        `json:"input,omitempty"`
	Content          any                        `json:"content,omitempty"`
	ToolUseID        string                     `json:"tool_use_id,omitempty"`
	IsError          bool                       `json:"is_error,omitempty"`
	ServerName       string                     `json:"server_name,omitempty"`
	Source           *apiBlockSource            `json:"source,omitempty"`
	CacheControl     *CacheControl              `json:"cache_control,omitempty"`
	Citations        []Citation                 `json:"citations,omitempty"`
	ErrorCode        string                     `json:"error_code,omitempty"`
	RetrievedAt      string                     `json:"retrieved_at,omitempty"`
	EncryptedContent string                     `json:"encrypted_content,omitempty"`
	PageAge          string                     `json:"page_age,omitempty"`
	Stdout           string                     `json:"stdout,omitempty"`
	EncryptedStdout  string                     `json:"encrypted_stdout,omitempty"`
	Stderr           string                     `json:"stderr,omitempty"`
	ReturnCode       int                        `json:"return_code,omitempty"`
	ToolReferences   []ToolReference            `json:"tool_references,omitempty"`
	URL              string                     `json:"url,omitempty"`
	Title            string                     `json:"title,omitempty"`
	MediaType        string                     `json:"media_type,omitempty"`
	Filename         string                     `json:"filename,omitempty"`
	ProviderExecuted bool                       `json:"provider_executed,omitempty"`
	Dynamic          bool                       `json:"dynamic,omitempty"`
	Caller           *ToolCallCallerInfo        `json:"caller,omitempty"`
	Raw              map[string]json.RawMessage `json:"-"`
	ProviderMetadata ProviderMetadata           `json:"-"`
}

func (b apiContentBlock) MarshalJSON() ([]byte, error) {
	type plain apiContentBlock
	typed, err := json.Marshal(plain(b))
	if err != nil {
		return nil, err
	}
	var out map[string]json.RawMessage
	if err := json.Unmarshal(typed, &out); err != nil {
		return nil, err
	}
	for key, value := range b.Raw {
		if isReservedReplayKey(b.Type, key) {
			continue
		}
		if _, exists := out[key]; !exists {
			out[key] = value
		}
	}
	return json.Marshal(out)
}

func (b *apiContentBlock) UnmarshalJSON(data []byte) error {
	type plain apiContentBlock
	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*b = apiContentBlock(decoded)
	return json.Unmarshal(data, &b.Raw)
}

var commonReplayReservedKeys = map[string]struct{}{
	"type": {}, "provider_metadata": {},
}

var replayReservedKeysByType = map[string]map[string]struct{}{
	"text":                                   {"text": {}, "citations": {}, "cache_control": {}},
	"thinking":                               {"thinking": {}, "signature": {}},
	"redacted_thinking":                      {"data": {}},
	"compaction":                             {"text": {}},
	"image":                                  {"source": {}, "cache_control": {}},
	"document":                               {"source": {}, "citations": {}, "cache_control": {}},
	"content_reference":                      {"id": {}},
	"tool_use":                               {"id": {}, "name": {}, "input": {}, "dynamic": {}, "caller": {}},
	"server_tool_use":                        {"id": {}, "name": {}, "input": {}, "dynamic": {}, "caller": {}},
	"mcp_tool_use":                           {"id": {}, "name": {}, "server_name": {}, "input": {}, "dynamic": {}},
	"tool_result":                            {"tool_use_id": {}, "content": {}, "is_error": {}},
	"mcp_tool_result":                        {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"web_fetch_tool_result":                  {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"web_search_tool_result":                 {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"code_execution_tool_result":             {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"bash_code_execution_tool_result":        {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"text_editor_code_execution_tool_result": {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"tool_search_tool_result":                {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"advisor_tool_result":                    {"tool_use_id": {}, "content": {}, "is_error": {}, "dynamic": {}, "caller": {}},
	"web_fetch_error":                        {"error_code": {}},
	"web_fetch_result":                       {"url": {}, "retrieved_at": {}, "content": {}},
	"web_search_error":                       {"error_code": {}},
	"web_search_result":                      {"url": {}, "title": {}, "encrypted_content": {}, "page_age": {}},
	"code_execution_error":                   {"error_code": {}},
	"code_execution_result":                  {"stdout": {}, "stderr": {}, "return_code": {}, "content": {}},
	"encrypted_code_execution_result":        {"encrypted_stdout": {}, "stderr": {}, "return_code": {}, "content": {}},
	"bash_code_execution_error":              {"error_code": {}},
	"bash_code_execution_result":             {"stdout": {}, "stderr": {}, "return_code": {}, "content": {}},
	"tool_search_error":                      {"error_code": {}},
	"tool_search_result":                     {"tool_references": {}},
	"advisor_error":                          {"error_code": {}},
	"advisor_result":                         {"text": {}},
	"advisor_redacted_result":                {"encrypted_content": {}},
}

func isReservedReplayKey(blockType, key string) bool {
	if _, ok := commonReplayReservedKeys[key]; ok {
		return true
	}
	_, ok := replayReservedKeysByType[blockType][key]
	return ok
}

type apiBlockSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

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
			blocks = append(blocks, convertToolResultContent(c))
		}
	}
	return blocks
}

func convertAssistantContent(contents []AssistantContent) []apiContentBlock {
	blocks := make([]apiContentBlock, 0, len(contents))
	for _, content := range contents {
		switch c := content.(type) {
		case TextContent:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: c.Text, Citations: c.Citations, Raw: c.Raw})
		case CacheControlledTextContent:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: c.Text, CacheControl: c.CacheControl})
		case ThinkingContent:
			blocks = append(blocks, apiContentBlock{Type: "thinking", Thinking: c.Thinking, Signature: c.Signature})
		case RedactedThinkingContent:
			blocks = append(blocks, apiContentBlock{Type: "redacted_thinking", Data: c.Data})
		case CompactionContent:
			blocks = append(blocks, apiContentBlock{Type: "compaction", Text: c.Text})
		case ToolCallContent:
			blockType := "tool_use"
			if c.ProviderExecuted {
				blockType = "server_tool_use"
			}
			blocks = append(blocks, apiContentBlock{Type: blockType, ID: c.ToolCallID, Name: c.ToolName, Input: normalizedToolInput(c.Input), Dynamic: c.Dynamic, Caller: c.Caller, Raw: c.Raw})
		case ServerToolUseContent:
			blocks = append(blocks, apiContentBlock{Type: "server_tool_use", ID: c.ID, Name: c.Name, Input: normalizedToolInput(c.Input), Dynamic: c.Dynamic, Caller: c.Caller, Raw: c.Raw})
		case MCPToolUseContent:
			blocks = append(blocks, apiContentBlock{Type: "mcp_tool_use", ID: c.ID, Name: c.Name, ServerName: c.ServerName, Input: normalizedToolInput(c.Input), Dynamic: c.Dynamic, Raw: c.Raw})
		case ToolResultContent:
			blocks = append(blocks, convertToolResultContent(c))
		}
	}
	return orderMixedToolBlocks(blocks)
}

func orderMixedToolBlocks(blocks []apiContentBlock) []apiContentBlock {
	for start := 0; start < len(blocks); {
		if !isToolReplayBlock(blocks[start]) {
			start++
			continue
		}
		end := start
		for end < len(blocks) && isToolReplayBlock(blocks[end]) {
			end++
		}
		var provider, client []apiContentBlock
		for _, block := range blocks[start:end] {
			if isProviderReplayBlock(block.Type) {
				provider = append(provider, block)
			} else {
				client = append(client, block)
			}
		}
		copy(blocks[start:end], append(provider, client...))
		start = end
	}
	return blocks
}

func isToolReplayBlock(block apiContentBlock) bool {
	return block.Type == "tool_use" || block.Type == "tool_result" || isProviderReplayBlock(block.Type)
}

func isProviderReplayBlock(blockType string) bool {
	return blockType == "server_tool_use" || blockType == "mcp_tool_use" || (blockType != "tool_result" && len(blockType) > len("_tool_result") && blockType[len(blockType)-len("_tool_result"):] == "_tool_result")
}

func convertToolResultContent(c ToolResultContent) apiContentBlock {
	typed := convertToolResultValue(c.Result)
	content := typed
	if len(c.RawContent) != 0 {
		content = mergeReplayValue(rawJSONToAny(c.RawContent), typed)
	}
	return apiContentBlock{Type: valueOrDefault(c.Type, "tool_result"), ToolUseID: c.ToolCallID, Content: content, IsError: c.IsError, Dynamic: c.Dynamic, Caller: c.Caller, Raw: c.Raw}
}

func mergeReplayValue(raw, typed any) any {
	switch typedValue := typed.(type) {
	case []apiContentBlock:
		rawValues, ok := raw.([]any)
		if !ok {
			return raw
		}
		merged := make([]any, len(typedValue))
		for i, block := range typedValue {
			var rawValue any
			if i < len(rawValues) {
				rawValue = rawValues[i]
			}
			merged[i] = mergeReplayBlock(rawValue, block)
		}
		return merged
	default:
		return typed
	}
}

func mergeReplayBlock(raw any, typed apiContentBlock) any {
	typedJSON, _ := json.Marshal(typed)
	var typedMap map[string]any
	_ = json.Unmarshal(typedJSON, &typedMap)
	return mergeCanonicalReplayValue(raw, typedMap)
}

func mergeCanonicalReplayValue(raw, typed any) any {
	switch typedValue := typed.(type) {
	case map[string]any:
		rawMap, _ := raw.(map[string]any)
		blockType, _ := typedValue["type"].(string)
		out := make(map[string]any, len(rawMap)+len(typedValue))
		for key, value := range rawMap {
			if !isReservedReplayKey(blockType, key) {
				out[key] = value
			}
		}
		for key, value := range typedValue {
			out[key] = mergeCanonicalReplayValue(rawMap[key], value)
		}
		return out
	case []any:
		rawValues, _ := raw.([]any)
		out := make([]any, len(typedValue))
		for i, value := range typedValue {
			var rawValue any
			if i < len(rawValues) {
				rawValue = rawValues[i]
			}
			out[i] = mergeCanonicalReplayValue(rawValue, value)
		}
		return out
	default:
		return typed
	}
}

func convertToolResultParts(parts []ToolResultPart) []apiContentBlock {
	blocks := make([]apiContentBlock, 0, len(parts))
	for _, part := range parts {
		switch p := part.(type) {
		case ToolResultUnknown:
			blocks = append(blocks, apiContentBlock{Type: p.Type, Raw: p.Raw})
		case ToolResultText:
			blocks = append(blocks, apiContentBlock{Type: "text", Text: p.Text})
		case ToolResultImage:
			blocks = append(blocks, apiContentBlock{Type: "image", Source: imageSourceToAPI(p.Source)})
		case ToolResultDocument:
			blocks = append(blocks, apiContentBlock{Type: "document", Source: documentSourceToAPI(p.Source)})
		case ToolResultReference:
			blocks = append(blocks, apiContentBlock{Type: "content_reference", ID: p.ID})
		case WebFetchError:
			blocks = append(blocks, apiContentBlock{Type: p.Type, ErrorCode: p.ErrorCode})
		case WebFetchResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, URL: p.URL, RetrievedAt: p.RetrievedAt, Content: apiContentBlock{Type: "document", Source: documentSourceToAPI(p.Content.Source)}})
		case WebSearchError:
			blocks = append(blocks, apiContentBlock{Type: p.Type, ErrorCode: p.ErrorCode})
		case WebSearchResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, URL: p.URL, Title: p.Title, EncryptedContent: p.EncryptedContent, PageAge: p.PageAge})
		case CodeExecutionError:
			blocks = append(blocks, apiContentBlock{Type: p.Type, ErrorCode: p.ErrorCode})
		case CodeExecutionResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, Stdout: p.Stdout, Stderr: p.Stderr, ReturnCode: p.ReturnCode, Content: p.Content})
		case EncryptedCodeExecutionResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, EncryptedStdout: p.EncryptedStdout, Stderr: p.Stderr, ReturnCode: p.ReturnCode, Content: p.Content})
		case BashCodeExecutionError:
			blocks = append(blocks, apiContentBlock{Type: p.Type, ErrorCode: p.ErrorCode})
		case BashCodeExecutionResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, Stdout: p.Stdout, Stderr: p.Stderr, ReturnCode: p.ReturnCode, Content: p.Content})
		case ToolSearchError:
			blocks = append(blocks, apiContentBlock{Type: p.Type, ErrorCode: p.ErrorCode})
		case ToolSearchResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, ToolReferences: p.ToolReferences})
		case AdvisorError:
			blocks = append(blocks, apiContentBlock{Type: p.Type, ErrorCode: p.ErrorCode})
		case AdvisorResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, Text: p.Text})
		case AdvisorRedactedResult:
			blocks = append(blocks, apiContentBlock{Type: p.Type, EncryptedContent: p.EncryptedContent})
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

func normalizedToolInput(raw json.RawMessage) any {
	var object map[string]any
	if len(raw) == 0 || json.Unmarshal(raw, &object) != nil || object == nil {
		return map[string]any{}
	}
	return object
}

func convertToolResultValue(parts []ToolResultPart) any {
	if len(parts) == 1 {
		if text, ok := parts[0].(ToolResultText); ok {
			return []apiContentBlock{{Type: "text", Text: text.Text}}
		}
	}
	return convertToolResultParts(parts)
}

func imageSourceToAPI(source ImageSource) *apiBlockSource {
	return &apiBlockSource{Type: valueOrDefault(source.Type, "base64"), MediaType: source.MediaType, Data: source.Data, URL: source.URL}
}

func documentSourceToAPI(source DocumentSource) *apiBlockSource {
	return &apiBlockSource{Type: valueOrDefault(source.Type, "base64"), MediaType: source.MediaType, Data: source.Data, URL: source.URL}
}
