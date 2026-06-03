package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

type anthropicLanguageModel struct {
	provider *anthropicProvider
	modelID  string
	options  ModelOptions
}

func (m *anthropicLanguageModel) ModelID() string  { return m.modelID }
func (m *anthropicLanguageModel) Provider() string { return m.provider.name }

func (m *anthropicLanguageModel) SupportURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{
		"anthropic": {
			regexp.MustCompile(`https://docs\.anthropic\.com/.*`),
			regexp.MustCompile(`https://status\.anthropic\.com/.*`),
		},
	}
}

func (m *anthropicLanguageModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	m.provider.logger.Debug("anthropic generate request", "model", m.modelID)
	request, warnings, requestOptions := m.buildRequestWithOptions(opts, false)
	body, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	resp, err := m.execute(ctx, body, requestOptions)
	if err != nil {
		m.provider.logger.Error("anthropic generate request failed", "model", m.modelID, "error", err)
		return nil, err
	}
	if resp == nil {
		return nil, &APICallError{Message: "fetcher returned nil response and nil error", Retryable: true}
	}
	defer resp.Body.Close()
	limit := m.provider.maxResponseBodyBytes
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		limit = m.provider.maxErrorResponseBytes
	}
	respBody, truncated, err := readLimited(resp.Body, limit)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		err := apiCallError(resp, respBody, truncated, nil)
		m.provider.logger.Error("anthropic generate API error", "model", m.modelID, "status", resp.StatusCode, "error", err)
		return nil, err
	}
	if truncated {
		return nil, &APICallError{Message: "response body exceeded configured limit", Status: resp.StatusCode, Headers: cloneHeader(resp.Header), RequestID: responseRequestID(resp.Header), Body: respBody, Truncated: true, Retryable: false}
	}
	result, err := parseGenerateResponse(respBody)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(warnings, result.Warnings...)
	m.provider.logger.Debug("anthropic generate response", "model", m.modelID, "finishReason", result.FinishReason)
	return result, nil
}

func (m *anthropicLanguageModel) DoStream(ctx context.Context, opts StreamOptions) (*StreamResult, error) {
	m.provider.logger.Debug("anthropic stream request", "model", m.modelID)
	request, warnings, requestOptions := m.buildRequestWithOptions(GenerateOptions(opts), true)
	body, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	resp, err := m.execute(ctx, body, requestOptions)
	if err != nil {
		m.provider.logger.Error("anthropic stream request failed", "model", m.modelID, "error", err)
		return nil, err
	}
	if resp == nil {
		return nil, &APICallError{Message: "fetcher returned nil response and nil error", Retryable: true}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		respBody, truncated, readErr := readLimited(resp.Body, m.provider.maxErrorResponseBytes)
		err := apiCallError(resp, respBody, truncated, readErr)
		m.provider.logger.Error("anthropic stream API error", "model", m.modelID, "status", resp.StatusCode, "error", err)
		return nil, err
	}

	parts := make(chan StreamPart)
	result := &StreamResult{Stream: parts, Parts: parts, Request: body, Response: &StreamResponse{Headers: resp.Header, ModelID: m.modelID}}
	go func() {
		defer close(parts)
		defer resp.Body.Close()
		for _, warning := range warnings {
			parts <- StreamRaw{Event: warning}
		}
		parseSSEStream(resp.Body, parts, result.Response)
	}()
	return result, nil
}

type apiMessagesRequest struct {
	Model                  string             `json:"model"`
	Messages               []apiMessage       `json:"messages"`
	System                 []apiContentBlock  `json:"system,omitempty"`
	MaxTokens              int                `json:"max_tokens"`
	Metadata               *Metadata          `json:"metadata,omitempty"`
	StopSequences          []string           `json:"stop_sequences,omitempty"`
	Stream                 bool               `json:"stream,omitempty"`
	Temperature            *float64           `json:"temperature,omitempty"`
	TopK                   *int               `json:"top_k,omitempty"`
	TopP                   *float64           `json:"top_p,omitempty"`
	Tools                  []Tool             `json:"tools,omitempty"`
	ToolChoice             *ToolChoice        `json:"tool_choice,omitempty"`
	Effort                 string             `json:"effort,omitempty"`
	TaskBudget             *int               `json:"task_budget,omitempty"`
	SendReasoning          bool               `json:"send_reasoning,omitempty"`
	Container              *Container         `json:"container,omitempty"`
	MCPServers             []MCPServer        `json:"mcp_servers,omitempty"`
	Speed                  string             `json:"speed,omitempty"`
	InferenceGeo           string             `json:"inference_geo,omitempty"`
	ResponseFormat         *ResponseFormat    `json:"response_format,omitempty"`
	OutputConfig           *OutputConfig      `json:"output_config,omitempty"`
	Thinking               *apiThinking       `json:"thinking,omitempty"`
	ContextManagement      *ContextManagement `json:"context_management,omitempty"`
	DisableParallelToolUse bool               `json:"disable_parallel_tool_use,omitempty"`
}

type apiThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
	Display      string `json:"display,omitempty"`
}

func (m *anthropicLanguageModel) buildRequest(opts GenerateOptions, stream bool) (apiMessagesRequest, []Warning) {
	request, warnings, _ := m.buildRequestWithOptions(opts, stream)
	return request, warnings
}

func (m *anthropicLanguageModel) buildRequestWithOptions(opts GenerateOptions, stream bool) (apiMessagesRequest, []Warning, ModelOptions) {
	system, messages := convertPrompt(opts.Messages)
	temperature, warnings := normalizeTemperature(opts.Temperature)
	warnings = append(warnings, unsupportedWarnings(opts)...)
	preparedTools := prepareToolsForRequest(opts.Tools, opts.ToolOptions)
	modelOptions := m.options
	modelOptions.RequestTools = preparedTools
	toolChoice := opts.ToolChoice
	if toolChoice != nil && toolChoice.DisableParallelToolUse {
		modelOptions.DisableParallelToolUse = true
	}
	maxTokens := opts.MaxTokens
	thinking, thinkingWarnings := buildThinking(modelOptions.Thinking, &maxTokens, &temperature, &opts.TopK, &opts.TopP)
	warnings = append(warnings, thinkingWarnings...)
	outputConfig, structuredTools, structuredChoice, disableParallel := m.buildStructuredOutput(opts.StructuredOutput, opts.ResponseFormat, modelOptions)
	if len(structuredTools) > 0 {
		preparedTools = append(preparedTools, structuredTools...)
		modelOptions.RequestTools = preparedTools
	}
	if structuredChoice != nil {
		toolChoice = structuredChoice
	}
	if disableParallel {
		modelOptions.DisableParallelToolUse = true
	}
	return apiMessagesRequest{
		Model:                  m.modelID,
		Messages:               messages,
		System:                 system,
		MaxTokens:              maxTokens,
		Metadata:               modelOptions.Metadata,
		StopSequences:          opts.StopSequences,
		Stream:                 stream,
		Temperature:            temperature,
		TopK:                   opts.TopK,
		TopP:                   opts.TopP,
		Tools:                  preparedTools,
		ToolChoice:             toolChoice,
		Container:              modelOptions.Container,
		MCPServers:             modelOptions.MCPServers,
		Speed:                  modelOptions.Speed,
		InferenceGeo:           modelOptions.InferenceGeo,
		Effort:                 modelOptions.Effort,
		TaskBudget:             modelOptions.TaskBudget,
		SendReasoning:          modelOptions.SendReasoning,
		ResponseFormat:         opts.ResponseFormat,
		OutputConfig:           outputConfig,
		Thinking:               thinking,
		ContextManagement:      modelOptions.ContextManagement,
		DisableParallelToolUse: modelOptions.DisableParallelToolUse,
	}, warnings, modelOptions
}

func buildThinking(config *ThinkingConfig, maxTokens *int, temperature **float64, topK **int, topP **float64) (*apiThinking, []Warning) {
	if config == nil {
		return nil, nil
	}
	switch config.Type {
	case ThinkingTypeEnabled:
		budget := config.BudgetTokens
		if budget == 0 {
			budget = 1024
		}
		*maxTokens += budget
		var warnings []Warning
		if *temperature != nil {
			*temperature = nil
			warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "temperature is not supported when thinking is enabled"})
		}
		if *topK != nil {
			*topK = nil
			warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "topK is not supported when thinking is enabled"})
		}
		if *topP != nil {
			*topP = nil
			warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "topP is not supported when thinking is enabled"})
		}
		return &apiThinking{Type: "enabled", BudgetTokens: budget}, warnings
	case ThinkingTypeAdaptive:
		return &apiThinking{Type: "adaptive", Display: string(config.Display)}, nil
	case ThinkingTypeDisabled:
		return &apiThinking{Type: "disabled"}, nil
	default:
		return nil, nil
	}
}

func (m *anthropicLanguageModel) buildStructuredOutput(output *StructuredOutput, responseFormat *ResponseFormat, opts ModelOptions) (*OutputConfig, []Tool, *ToolChoice, bool) {
	if output == nil && responseFormat == nil {
		return nil, nil, nil, false
	}
	schema := any(nil)
	if output != nil {
		schema = output.Schema
	} else if responseFormat != nil {
		schema = responseFormat.Schema
	}
	mode := opts.StructuredOutputMode
	if mode == "" || mode == StructuredOutputModeAuto {
		if ModelCapabilitiesForID(m.modelID).StructuredOutput {
			mode = StructuredOutputModeOutputFormat
		} else {
			mode = StructuredOutputModeJSONTool
		}
	}
	sanitized := SanitizeSchema(schema)
	if mode == StructuredOutputModeJSONTool {
		return nil, []Tool{{Name: "json", Description: "Respond with JSON matching the requested schema.", InputSchema: sanitized}}, &ToolChoice{Type: "required", Name: "json", DisableParallelToolUse: true}, true
	}
	return &OutputConfig{Format: &ResponseFormat{Type: "json", Schema: sanitized}}, nil, nil, false
}

func prepareToolsForRequest(input []Tool, opts ToolOptions) []Tool {
	prepared := make([]Tool, len(input))
	for i, tool := range input {
		if opts.DeferLoading != nil {
			tool.DeferLoading = opts.DeferLoading
		}
		if opts.AllowedCallers != nil {
			tool.AllowedCallers = append([]ToolCallCaller(nil), opts.AllowedCallers...)
		}
		if opts.EagerInputStreaming != nil {
			tool.EagerInputStreaming = opts.EagerInputStreaming
		}
		prepared[i] = tool
	}
	return prepared
}

func normalizeTemperature(input *float64) (*float64, []Warning) {
	if input == nil {
		return nil, nil
	}
	value := *input
	if value > 1 {
		clamped := 1.0
		return &clamped, []Warning{{Type: "unsupported-setting", Message: "temperature was clamped to 1.0"}}
	}
	if value < 0 {
		clamped := 0.0
		return &clamped, []Warning{{Type: "unsupported-setting", Message: "temperature was clamped to 0"}}
	}
	return &value, nil
}

func unsupportedWarnings(opts GenerateOptions) []Warning {
	var warnings []Warning
	if opts.FrequencyPenalty != nil {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "frequencyPenalty is not supported by Anthropic"})
	}
	if opts.PresencePenalty != nil {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "presencePenalty is not supported by Anthropic"})
	}
	if opts.Seed != nil {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "seed is not supported by Anthropic"})
	}
	if opts.TopP != nil && opts.Temperature != nil {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "topP is ignored when temperature is set for Anthropic models"})
	}
	return warnings
}

func (m *anthropicLanguageModel) execute(ctx context.Context, body []byte, opts ModelOptions) (*http.Response, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	var lastErr error
	for attempt := 0; attempt <= m.provider.retry.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(m.provider.baseURL, "/")+"/messages", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if m.provider.generateID != nil {
			req.Header.Set("x-request-id", m.provider.generateID())
		}
		for k, values := range m.provider.headersForOptions(opts) {
			for _, v := range values {
				req.Header.Add(k, v)
			}
		}
		resp, err := m.provider.fetch.Do(req)
		if !shouldRetry(resp, err) || attempt == m.provider.retry.MaxRetries {
			return resp, err
		}
		lastErr = err
		if resp != nil && resp.Body != nil {
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1024))
			_ = resp.Body.Close()
		}
		delay := m.provider.retryDelay(attempt, resp)
		m.provider.logger.Warn("anthropic request retry", "model", m.modelID, "attempt", attempt+1, "delay", delay, "error", err)
		timer := timeNewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
	return nil, lastErr
}

func shouldRetry(resp *http.Response, err error) bool {
	if err != nil {
		return true
	}
	if resp == nil {
		return false
	}
	return resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
}

func timeNewTimer(delay time.Duration) *time.Timer { return time.NewTimer(delay) }

func ModelCapabilitiesForID(modelID string) ModelCapabilities {
	switch modelID {
	case "claude-opus-4-8", "claude-opus-4-7":
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, RejectsSampling: true}
	case "claude-sonnet-4-6", "claude-opus-4-6":
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true}
	case "claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5", "claude-haiku-4-5-20251001":
		return ModelCapabilities{MaxOutputTokens: 64000, StructuredOutput: true}
	case "claude-opus-4-1":
		return ModelCapabilities{MaxOutputTokens: 32000, StructuredOutput: true}
	case "claude-sonnet-4", "claude-sonnet-4-0":
		return ModelCapabilities{MaxOutputTokens: 64000}
	case "claude-opus-4", "claude-opus-4-0":
		return ModelCapabilities{MaxOutputTokens: 32000}
	case "claude-3-haiku", "claude-3-haiku-20240307":
		return ModelCapabilities{MaxOutputTokens: 4096}
	default:
		return ModelCapabilities{MaxOutputTokens: 4096}
	}
}

type apiMessagesResponse struct {
	ID                string                        `json:"id"`
	Model             string                        `json:"model"`
	Role              string                        `json:"role"`
	Content           []apiContentBlock             `json:"content"`
	StopReason        string                        `json:"stop_reason"`
	StopSequence      string                        `json:"stop_sequence"`
	Usage             apiUsage                      `json:"usage"`
	Container         *ContainerInfo                `json:"container"`
	ContextManagement *apiContextManagementResponse `json:"context_management"`
}

type apiContextManagementResponse struct {
	AppliedEdits []apiContextManagementEditResponse `json:"applied_edits"`
	Edits        []apiContextManagementEditResponse `json:"edits"`
}

type apiContextManagementEditResponse struct {
	Type                 string `json:"type"`
	ClearedToolUses      int    `json:"cleared_tool_uses"`
	ClearedThinkingTurns int    `json:"cleared_thinking_turns"`
	Compacted            bool   `json:"compacted"`
}

type apiUsage struct {
	InputTokens              int              `json:"input_tokens"`
	OutputTokens             int              `json:"output_tokens"`
	CacheCreationInputTokens int              `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int              `json:"cache_read_input_tokens"`
	ServiceTier              string           `json:"service_tier"`
	Iterations               []UsageIteration `json:"iterations"`
}

func parseGenerateResponse(body []byte) (*GenerateResult, error) {
	var response apiMessagesResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, err
	}
	usage := UsageFromResponseUsage(ResponseUsage{
		InputTokens:              response.Usage.InputTokens,
		OutputTokens:             response.Usage.OutputTokens,
		CacheCreationInputTokens: response.Usage.CacheCreationInputTokens,
		CacheReadInputTokens:     response.Usage.CacheReadInputTokens,
		ServiceTier:              response.Usage.ServiceTier,
		Iterations:               response.Usage.Iterations,
	})
	metadata := MessageMetadata{
		"id":             response.ID,
		"model":          response.Model,
		"stopSequence":   response.StopSequence,
		"anthropicUsage": response.Usage,
	}
	if response.Container != nil {
		metadata["container"] = response.Container
	}
	if response.ContextManagement != nil {
		metadata["contextManagement"] = parseContextManagementResponse(response.ContextManagement)
	}
	content := convertJSONToolCallToText(parseContentBlocks(response.Content))
	return &GenerateResult{
		Content:          content,
		FinishReason:     FinishReasonFromStopReason(response.StopReason),
		Usage:            usage,
		ProviderMetadata: ProviderMetadata{"anthropic": response},
		MessageMetadata:  metadata,
	}, nil
}

func convertJSONToolCallToText(content []Content) []Content {
	converted := make([]Content, len(content))
	copy(converted, content)
	for i, item := range converted {
		call, ok := item.(ToolCallContent)
		if !ok || call.ToolName != "json" {
			continue
		}
		converted[i] = TextContent{Text: string(call.Input)}
	}
	return converted
}

func parseContextManagementResponse(response *apiContextManagementResponse) ContextManagementResponse {
	if response == nil {
		return ContextManagementResponse{}
	}
	return ContextManagementResponse{Edits: parseContextManagementEditResponses(response.Edits), AppliedEdits: parseContextManagementEditResponses(response.AppliedEdits)}
}

func parseContextManagementEditResponses(edits []apiContextManagementEditResponse) []ContextManagementEditResponse {
	parsed := make([]ContextManagementEditResponse, 0, len(edits))
	for _, edit := range edits {
		switch edit.Type {
		case "clear_tool_uses":
			parsed = append(parsed, ClearToolUsesResponse{ClearedToolUses: edit.ClearedToolUses})
		case "clear_thinking":
			parsed = append(parsed, ClearThinkingResponse{ClearedThinkingTurns: edit.ClearedThinkingTurns})
		case "compact":
			parsed = append(parsed, CompactResponse{Compacted: edit.Compacted})
		}
	}
	return parsed
}

func parseContentBlocks(blocks []apiContentBlock) []Content {
	content := make([]Content, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			content = append(content, TextContent{Text: block.Text, Citations: normalizeCitations(block.Citations)})
		case "thinking":
			content = append(content, ReasoningContent{Type: "reasoning", Text: block.Thinking, Signature: block.Signature})
		case "redacted_thinking":
			content = append(content, ReasoningContent{Type: "redacted_reasoning", Text: block.Data})
		case "compaction":
			content = append(content, CompactionContent{Text: block.Text})
		case "source", "web_search_result", "web_fetch_result":
			content = append(content, SourceContent{SourceType: block.Type, ID: block.ID, URL: block.URL, Title: block.Title, MediaType: block.MediaType, Filename: block.Filename})
		case "tool_use":
			content = append(content, ToolCallContent{Type: "tool-call", ToolCallID: block.ID, ToolName: block.Name, Input: anyToRawJSON(block.Input), ProviderExecuted: block.ProviderExecuted, Dynamic: block.Dynamic, ProviderMetadata: block.ProviderMetadata})
		case "server_tool_use":
			content = append(content, ToolCallContent{Type: "server-tool-call", ToolCallID: block.ID, ToolName: block.Name, Input: anyToRawJSON(block.Input), ProviderExecuted: true, Dynamic: block.Dynamic, ProviderMetadata: block.ProviderMetadata})
		case "mcp_tool_use":
			content = append(content, ToolCallContent{Type: "mcp-tool-call", ToolCallID: block.ID, ToolName: block.Name, Input: anyToRawJSON(block.Input), ProviderExecuted: true, Dynamic: block.Dynamic, ProviderMetadata: block.ProviderMetadata})
		case "mcp_tool_result", "web_fetch_tool_result", "web_search_tool_result", "code_execution_tool_result", "bash_code_execution_tool_result", "text_editor_code_execution_tool_result", "tool_search_tool_result", "advisor_tool_result", "tool_result":
			content = append(content, ToolResultContent{ToolCallID: block.ToolUseID, Result: parseToolResultParts(block.Content), IsError: block.IsError, ProviderExecuted: block.ProviderExecuted, Dynamic: block.Dynamic, ProviderMetadata: block.ProviderMetadata})
		}
	}
	return content
}

func anyToRawJSON(value any) json.RawMessage {
	if value == nil {
		return nil
	}
	data, _ := json.Marshal(value)
	return data
}

func parseToolResultParts(value any) []ToolResultPart {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	var blocks []apiContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return []ToolResultPart{ToolResultText{Text: string(raw)}}
	}
	parts := make([]ToolResultPart, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, ToolResultText{Text: block.Text})
		case "image":
			if block.Source != nil {
				parts = append(parts, ToolResultImage{Source: ImageSource{Type: block.Source.Type, MediaType: block.Source.MediaType, Data: block.Source.Data, URL: block.Source.URL}})
			}
		case "document":
			if block.Source != nil {
				parts = append(parts, ToolResultDocument{Source: DocumentSource{Type: block.Source.Type, MediaType: block.Source.MediaType, Data: block.Source.Data, URL: block.Source.URL}})
			}
		case "content_reference":
			parts = append(parts, ToolResultReference{ID: block.ID})
		case "web_fetch_error":
			parts = append(parts, WebFetchError{Type: block.Type, ErrorCode: block.ErrorCode})
		case "web_fetch_result":
			parts = append(parts, WebFetchResult{Type: block.Type, URL: block.URL, RetrievedAt: block.RetrievedAt, Content: DocumentContent{Source: DocumentSource{Type: valueOrDefault(block.Type, "document"), MediaType: block.MediaType, Data: block.Data, URL: block.URL}}})
		case "web_search_error":
			parts = append(parts, WebSearchError{Type: block.Type, ErrorCode: block.ErrorCode})
		case "web_search_result":
			parts = append(parts, WebSearchResult{Type: block.Type, URL: block.URL, Title: block.Title, EncryptedContent: block.EncryptedContent, PageAge: block.PageAge})
		case "code_execution_error":
			parts = append(parts, CodeExecutionError{Type: block.Type, ErrorCode: block.ErrorCode})
		case "code_execution_result":
			parts = append(parts, CodeExecutionResult{Type: block.Type, Stdout: block.Stdout, Stderr: block.Stderr, ReturnCode: block.ReturnCode, Content: parseCodeExecutionOutput(block.Content)})
		case "encrypted_code_execution_result":
			parts = append(parts, EncryptedCodeExecutionResult{Type: block.Type, EncryptedStdout: block.EncryptedStdout, Stderr: block.Stderr, ReturnCode: block.ReturnCode, Content: parseCodeExecutionOutput(block.Content)})
		case "bash_code_execution_error":
			parts = append(parts, BashCodeExecutionError{Type: block.Type, ErrorCode: block.ErrorCode})
		case "bash_code_execution_result":
			parts = append(parts, BashCodeExecutionResult{Type: block.Type, Content: parseBashCodeExecutionOutput(block.Content), Stdout: block.Stdout, Stderr: block.Stderr, ReturnCode: block.ReturnCode})
		case "tool_search_error":
			parts = append(parts, ToolSearchError{Type: block.Type, ErrorCode: block.ErrorCode})
		case "tool_search_result":
			parts = append(parts, ToolSearchResult{Type: block.Type, ToolReferences: block.ToolReferences})
		case "advisor_error":
			parts = append(parts, AdvisorError{Type: block.Type, ErrorCode: block.ErrorCode})
		case "advisor_result":
			parts = append(parts, AdvisorResult{Type: block.Type, Text: block.Text})
		case "advisor_redacted_result":
			parts = append(parts, AdvisorRedactedResult{Type: block.Type, EncryptedContent: block.EncryptedContent})
		}
	}
	return parts
}

func parseCodeExecutionOutput(value any) []CodeExecutionOutput {
	raw, err := json.Marshal(value)
	if err != nil || string(raw) == "null" {
		return nil
	}
	var out []CodeExecutionOutput
	_ = json.Unmarshal(raw, &out)
	return out
}

func parseBashCodeExecutionOutput(value any) []BashCodeExecutionOutput {
	raw, err := json.Marshal(value)
	if err != nil || string(raw) == "null" {
		return nil
	}
	var out []BashCodeExecutionOutput
	_ = json.Unmarshal(raw, &out)
	return out
}

func normalizeCitations(citations []Citation) []Citation {
	for i := range citations {
		if citations[i].CitedText == "" {
			citations[i].CitedText = citations[i].Text
		}
		if citations[i].StartCharIndex == 0 {
			citations[i].StartCharIndex = citations[i].StartChar
		}
		if citations[i].EndCharIndex == 0 {
			citations[i].EndCharIndex = citations[i].EndChar
		}
	}
	return citations
}

func readLimited(reader io.Reader, limit int64) ([]byte, bool, error) {
	if limit <= 0 {
		limit = defaultMaxResponseBodyBytes
	}
	limited := io.LimitReader(reader, limit+1)
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) > limit {
		return body[:limit], true, nil
	}
	return body, false, nil
}

func apiCallError(resp *http.Response, body []byte, truncated bool, cause error) error {
	status := 0
	headers := http.Header(nil)
	requestID := ""
	if resp != nil {
		status = resp.StatusCode
		headers = cloneHeader(resp.Header)
		requestID = responseRequestID(resp.Header)
	}
	message := strings.TrimSpace(string(body))
	var decoded struct {
		Error APIError `json:"error"`
	}
	errorType := ""
	if err := json.Unmarshal(body, &decoded); err == nil && decoded.Error.Message != "" {
		message = decoded.Error.Message
		errorType = decoded.Error.Type
	}
	if truncated && message != "" {
		message += " [truncated]"
	} else if truncated {
		message = "response body exceeded configured limit"
	}
	return &APICallError{Status: status, Message: message, Type: errorType, Retryable: status == http.StatusTooManyRequests || status >= 500, Headers: headers, RequestID: requestID, Body: body, Truncated: truncated, Cause: cause}
}

func cloneHeader(header http.Header) http.Header {
	if header == nil {
		return nil
	}
	cloned := make(http.Header, len(header))
	for k, values := range header {
		cloned[k] = append([]string(nil), values...)
	}
	return cloned
}

func responseRequestID(header http.Header) string {
	if header == nil {
		return ""
	}
	for _, name := range []string{"x-request-id", "request-id", "x-amzn-requestid"} {
		if value := header.Get(name); value != "" {
			return value
		}
	}
	return ""
}

func parseSSEStream(reader io.Reader, out chan<- StreamPart, response *StreamResponse) {
	buffered := bufio.NewReader(reader)
	var eventType string
	var data strings.Builder
	blockTypes := map[string]string{}
	for {
		line, err := buffered.ReadString('\n')
		if err != nil && len(line) == 0 {
			if err != io.EOF {
				out <- StreamError{Err: err}
			} else if eventType != "" || data.Len() > 0 {
				emitSSEEvent(eventType, data.String(), out, blockTypes, response)
			}
			return
		}
		line = strings.TrimSuffix(line, "\n")
		line = strings.TrimSuffix(line, "\r")
		if line == "" {
			emitSSEEvent(eventType, data.String(), out, blockTypes, response)
			eventType = ""
			data.Reset()
		} else if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event: "))
		} else if strings.HasPrefix(line, "data: ") {
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimPrefix(line, "data: "))
		}
		if err == io.EOF {
			if eventType != "" || data.Len() > 0 {
				emitSSEEvent(eventType, data.String(), out, blockTypes, response)
			}
			return
		}
	}
}

func emitSSEEvent(eventType, data string, out chan<- StreamPart, blockTypes map[string]string, response *StreamResponse) {
	if eventType == "" || eventType == "ping" {
		return
	}
	if eventType == "error" {
		out <- StreamError{Err: streamingAPIError(data)}
		return
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(data), &payload); err != nil {
		out <- StreamError{Err: err}
		return
	}
	switch eventType {
	case "message_start":
		var message struct {
			Message struct{ ID, Model string } `json:"message"`
		}
		_ = json.Unmarshal([]byte(data), &message)
		response.ID = message.Message.ID
		response.ModelID = message.Message.Model
		out <- StreamStart{}
	case "content_block_start":
		var event struct {
			Index        int             `json:"index"`
			ContentBlock apiContentBlock `json:"content_block"`
		}
		_ = json.Unmarshal([]byte(data), &event)
		id := fmt.Sprint(event.Index)
		blockTypes[id] = event.ContentBlock.Type
		switch event.ContentBlock.Type {
		case "thinking", "redacted_thinking":
			out <- StreamReasoningStart{ID: id}
		case "input_json":
			out <- StreamToolInputStart{ID: id, ToolName: event.ContentBlock.Name, ProviderExecuted: event.ContentBlock.ProviderExecuted, Dynamic: event.ContentBlock.Dynamic}
		case "tool_use", "server_tool_use":
			out <- StreamToolCall{ToolCallContent: ToolCallContent{Type: streamToolCallType(event.ContentBlock.Type), ToolCallID: event.ContentBlock.ID, ToolName: event.ContentBlock.Name, Input: anyToRawJSON(event.ContentBlock.Input), ProviderExecuted: event.ContentBlock.ProviderExecuted || event.ContentBlock.Type != "tool_use", Dynamic: event.ContentBlock.Dynamic, ProviderMetadata: event.ContentBlock.ProviderMetadata}}
		case "mcp_tool_use":
			out <- StreamToolCall{ToolCallContent: ToolCallContent{Type: "mcp-tool-call", ToolCallID: event.ContentBlock.ID, ToolName: event.ContentBlock.Name, Input: anyToRawJSON(event.ContentBlock.Input), ProviderExecuted: true, Dynamic: event.ContentBlock.Dynamic, ProviderMetadata: event.ContentBlock.ProviderMetadata}}
		case "source", "web_search_result", "web_fetch_result":
			out <- StreamSource{SourceContent: SourceContent{SourceType: event.ContentBlock.Type, ID: event.ContentBlock.ID, URL: event.ContentBlock.URL, Title: event.ContentBlock.Title, MediaType: event.ContentBlock.MediaType, Filename: event.ContentBlock.Filename}, ProviderMetadata: event.ContentBlock.ProviderMetadata}
		case "tool_result", "mcp_tool_result", "web_fetch_tool_result", "web_search_tool_result", "code_execution_tool_result", "bash_code_execution_tool_result", "text_editor_code_execution_tool_result", "tool_search_tool_result", "advisor_tool_result":
			out <- StreamToolResult{ToolResultContent: ToolResultContent{ToolCallID: event.ContentBlock.ToolUseID, Result: parseToolResultParts(event.ContentBlock.Content), IsError: event.ContentBlock.IsError, ProviderExecuted: event.ContentBlock.ProviderExecuted, Dynamic: event.ContentBlock.Dynamic, ProviderMetadata: event.ContentBlock.ProviderMetadata}}
		default:
			out <- StreamTextStart{ID: id}
		}
	case "content_block_delta":
		var event struct {
			Index int `json:"index"`
			Delta struct {
				Type        string     `json:"type"`
				Text        string     `json:"text"`
				Thinking    string     `json:"thinking"`
				Signature   string     `json:"signature"`
				PartialJSON string     `json:"partial_json"`
				Citations   []Citation `json:"citations"`
			} `json:"delta"`
		}
		_ = json.Unmarshal([]byte(data), &event)
		id := fmt.Sprint(event.Index)
		switch event.Delta.Type {
		case "text_delta":
			out <- StreamTextDelta{ID: id, Text: event.Delta.Text, Citations: normalizeCitations(event.Delta.Citations)}
		case "citations_delta":
			out <- StreamTextDelta{ID: id, Citations: normalizeCitations(event.Delta.Citations)}
		case "thinking_delta":
			out <- StreamReasoningDelta{ID: id, Delta: ThinkingDelta{Thinking: event.Delta.Thinking}}
		case "signature_delta":
			out <- StreamReasoningDelta{ID: id, Delta: SignatureDelta{Signature: event.Delta.Signature}}
		case "input_json_delta":
			out <- StreamToolInputDelta{ID: id, Delta: InputJSONDelta{PartialJSON: event.Delta.PartialJSON}}
		case "mcp_tool_use_delta":
			out <- StreamToolInputDelta{ID: id, Delta: InputJSONDelta{PartialJSON: event.Delta.PartialJSON}}
		}
	case "content_block_stop":
		var event struct {
			Index int `json:"index"`
		}
		_ = json.Unmarshal([]byte(data), &event)
		id := fmt.Sprint(event.Index)
		if blockTypes[id] == "thinking" || blockTypes[id] == "redacted_thinking" {
			out <- StreamReasoningEnd{ID: id}
		} else if blockTypes[id] == "input_json" || blockTypes[id] == "mcp_tool_use" {
			out <- StreamToolInputEnd{ID: id}
		} else {
			out <- StreamTextEnd{ID: id}
		}
	case "message_delta":
		var event struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage apiUsage `json:"usage"`
		}
		_ = json.Unmarshal([]byte(data), &event)
		out <- StreamFinish{FinishReason: FinishReasonFromStopReason(event.Delta.StopReason), Usage: UsageFromResponseUsage(ResponseUsage{InputTokens: event.Usage.InputTokens, OutputTokens: event.Usage.OutputTokens, CacheCreationInputTokens: event.Usage.CacheCreationInputTokens, CacheReadInputTokens: event.Usage.CacheReadInputTokens})}
	case "message_stop":
		out <- StreamFinish{FinishReason: FinishReasonStop}
	}
}

func streamToolCallType(apiType string) string {
	switch apiType {
	case "server_tool_use":
		return "server-tool-call"
	case "mcp_tool_use":
		return "mcp-tool-call"
	default:
		return "tool-call"
	}
}

func streamingAPIError(data string) error {
	var event struct {
		Error APIError `json:"error"`
	}
	_ = json.Unmarshal([]byte(data), &event)
	status := http.StatusInternalServerError
	if event.Error.Type == "overloaded_error" {
		status = 529
	}
	return &APICallError{Status: status, Message: event.Error.Message, Retryable: status == 529 || status >= 500}
}
