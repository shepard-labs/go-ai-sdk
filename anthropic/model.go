package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"reflect"
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
	request, warnings, requestOptions, replaySafe, err := m.buildRequestWithOptions(opts, false)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	resp, err := m.execute(ctx, body, requestOptions, replaySafe)
	if err != nil {
		m.provider.logger.Error("anthropic generate request failed", "model", m.modelID, "error", err)
		return nil, err
	}
	if resp == nil {
		return nil, &APICallError{Message: "fetcher returned nil response and nil error", Retryable: replaySafe}
	}
	if resp.Body == nil {
		return nil, nilResponseBodyError(resp, replaySafe)
	}
	defer resp.Body.Close()
	limit := m.provider.maxResponseBodyBytes
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		limit = m.provider.maxErrorResponseBytes
	}
	respBody, truncated, err := readLimited(resp.Body, limit)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		apiErr := apiCallError(resp, respBody, truncated, err, replaySafe)
		m.provider.logger.Error("anthropic generate API error", "model", m.modelID, "status", resp.StatusCode, "error", apiErr)
		return nil, apiErr
	}
	if err != nil {
		return nil, &APICallError{
			Message:   "failed to read successful response body",
			Status:    resp.StatusCode,
			Headers:   cloneHeader(resp.Header),
			RequestID: responseRequestID(resp.Header),
			Body:      append([]byte(nil), respBody...),
			Truncated: truncated || int64(len(respBody)) >= limit,
			Retryable: retryableIncompleteResponse(resp.StatusCode, replaySafe),
			Cause:     err,
		}
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
	streamCtx, cancel := context.WithCancel(ctx)
	request, warnings, requestOptions, replaySafe, err := m.buildRequestWithOptions(GenerateOptions(opts), true)
	if err != nil {
		cancel()
		return nil, err
	}
	body, err := json.Marshal(request)
	if err != nil {
		cancel()
		return nil, err
	}
	resp, err := m.execute(streamCtx, body, requestOptions, replaySafe)
	if err != nil {
		cancel()
		m.provider.logger.Error("anthropic stream request failed", "model", m.modelID, "error", err)
		return nil, err
	}
	if resp == nil {
		cancel()
		return nil, &APICallError{Message: "fetcher returned nil response and nil error", Retryable: replaySafe}
	}
	if resp.Body == nil {
		cancel()
		return nil, nilResponseBodyError(resp, replaySafe)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer cancel()
		defer resp.Body.Close()
		respBody, truncated, readErr := readLimited(resp.Body, m.provider.maxErrorResponseBytes)
		err := apiCallError(resp, respBody, truncated, readErr, replaySafe)
		m.provider.logger.Error("anthropic stream API error", "model", m.modelID, "status", resp.StatusCode, "error", err)
		return nil, err
	}

	// Cancellation may discard buffered nonterminal events to make room for its
	// terminal error; normal completion preserves all events and backpressure.
	parts := make(chan StreamPart, 17)
	parsedParts := make(chan StreamPart)
	responseHeaders := cloneHeader(resp.Header)
	responseRequestID := responseRequestID(responseHeaders)
	cancelStream := func() {
		cancel()
		_ = resp.Body.Close()
	}
	result := &StreamResult{Stream: parts, Parts: parts, Request: body, Response: &StreamResponse{Headers: responseHeaders, ModelID: m.modelID}, Cancel: cancelStream}
	go func() {
		defer close(parsedParts)
		for _, warning := range warnings {
			if !sendStreamPart(streamCtx, parsedParts, StreamRaw{Event: warning}) {
				return
			}
		}
		parseSSEStreamWithMetadata(streamCtx, resp.Body, parsedParts, result.Response, m.provider.maxErrorResponseBytes, streamErrorMetadata{headers: responseHeaders, requestID: responseRequestID})
	}()
	go func() {
		defer close(parts)
		defer cancelStream()
		sendCancellation := func(err error) {
			terminal := StreamError{Err: err}
			for {
				select {
				case parts <- terminal:
					return
				case <-parts:
				}
			}
		}
		for {
			select {
			case <-streamCtx.Done():
				sendCancellation(streamCtx.Err())
				return
			case part, open := <-parsedParts:
				if !open {
					if err := streamCtx.Err(); err != nil {
						sendCancellation(err)
					}
					return
				}
				if isTerminalStreamPart(part) {
					select {
					case parts <- part:
					case <-streamCtx.Done():
						sendCancellation(streamCtx.Err())
					}
					return
				}
				select {
				case <-streamCtx.Done():
					sendCancellation(streamCtx.Err())
					return
				case parts <- part:
				}
			}
		}
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
	Tools                  []apiTool          `json:"tools,omitempty"`
	ToolChoice             *ToolChoice        `json:"tool_choice,omitempty"`
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
	Type         string                   `json:"type,omitempty"`
	BudgetTokens int                      `json:"budget_tokens,omitempty"`
	Display      string                   `json:"display,omitempty"`
	BlockBinding *apiThinkingBlockBinding `json:"block_binding,omitempty"`
}

type apiThinkingBlockBinding struct {
	PrefixMismatchBehavior string `json:"prefix_mismatch_behavior,omitempty"`
}

const minimumThinkingBudgetTokens = 1024

type apiCachingConfig struct {
	Type string `json:"type"`
	TTL  string `json:"ttl"`
}

// apiTool is the request-wire declaration and intentionally excludes SDK-only
// identity, execution, dynamic, and schema metadata from Tool.
type apiTool struct {
	ID                  string                   `json:"-"`
	Name                string                   `json:"name,omitempty"`
	Description         string                   `json:"description,omitempty"`
	InputSchema         any                      `json:"input_schema,omitempty"`
	CacheControl        *CacheControl            `json:"cache_control,omitempty"`
	EagerInputStreaming *bool                    `json:"eager_input_streaming,omitempty"`
	Strict              *bool                    `json:"strict,omitempty"`
	DeferLoading        *bool                    `json:"defer_loading,omitempty"`
	AllowedCallers      []ToolCallCaller         `json:"allowed_callers,omitempty"`
	InputExamples       []any                    `json:"input_examples,omitempty"`
	Type                string                   `json:"type,omitempty"`
	DisplayWidthPx      *int                     `json:"display_width_px,omitempty"`
	DisplayHeightPx     *int                     `json:"display_height_px,omitempty"`
	DisplayNumber       *int                     `json:"display_number,omitempty"`
	EnableZoom          *bool                    `json:"enable_zoom,omitempty"`
	MaxCharacters       *int                     `json:"max_characters,omitempty"`
	MaxUses             *int                     `json:"max_uses,omitempty"`
	AllowedDomains      []string                 `json:"allowed_domains,omitempty"`
	BlockedDomains      []string                 `json:"blocked_domains,omitempty"`
	Citations           *CitationsConfig         `json:"citations,omitempty"`
	MaxContentTokens    *int                     `json:"max_content_tokens,omitempty"`
	UserLocation        *UserLocation            `json:"user_location,omitempty"`
	Model               string                   `json:"model,omitempty"`
	Caching             *apiCachingConfig        `json:"caching,omitempty"`
	MCPServerName       string                   `json:"mcp_server_name,omitempty"`
	DefaultConfig       *MCPToolConfig           `json:"default_config,omitempty"`
	Configs             map[string]MCPToolConfig `json:"configs,omitempty"`
	UseCache            *bool                    `json:"use_cache,omitempty"`
	ResponseInclusion   string                   `json:"response_inclusion,omitempty"`
}

// Mappings: @ai-sdk/anthropic@3.0.115 src/anthropic-prepare-tools.ts:135-376;
// declarations are unchanged in @ai-sdk/anthropic@4.0.46:132-375.
func toolsToWire(tools []Tool) ([]apiTool, bool, error) {
	wire := make([]apiTool, 0, len(tools))
	replaySafe := true
	for i, tool := range tools {
		if tool.ID == "" {
			wire = append(wire, apiTool{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema, CacheControl: tool.CacheControl, EagerInputStreaming: tool.EagerInputStreaming, Strict: tool.Strict, DeferLoading: tool.DeferLoading, AllowedCallers: append([]ToolCallCaller(nil), tool.AllowedCallers...), InputExamples: append([]any(nil), tool.InputExamples...)})
			continue
		}
		replaySafe = false
		var provider apiTool
		switch tool.ID {
		case "anthropic.bash_20241022":
			provider.Type, provider.Name = "bash_20241022", "bash"
		case "anthropic.bash_20250124":
			provider.Type, provider.Name = "bash_20250124", "bash"
		case "anthropic.code_execution_20250522":
			provider.Type, provider.Name = "code_execution_20250522", "code_execution"
		case "anthropic.code_execution_20250825":
			provider.Type, provider.Name = "code_execution_20250825", "code_execution"
		case "anthropic.code_execution_20260120":
			provider.Type, provider.Name = "code_execution_20260120", "code_execution"
		case "anthropic.computer_20251124":
			if tool.DisplayWidthPx == nil || tool.DisplayHeightPx == nil || tool.DisplayNumber == nil || tool.EnableZoom == nil || *tool.DisplayWidthPx <= 0 || *tool.DisplayHeightPx <= 0 || *tool.DisplayNumber < 0 {
				return nil, false, invalidProviderTool(i, tool.ID, "valid display fields and enable zoom are required")
			}
			provider = apiTool{Type: "computer_20251124", Name: "computer", DisplayWidthPx: tool.DisplayWidthPx, DisplayHeightPx: tool.DisplayHeightPx, DisplayNumber: tool.DisplayNumber, EnableZoom: tool.EnableZoom}
		case "anthropic.computer_toolset_20260801":
			provider = apiTool{Type: "computer_toolset_20260801", Configs: tool.Configs}
		case "anthropic.text_editor_20250728":
			if tool.MaxCharacters != nil && *tool.MaxCharacters <= 0 {
				return nil, false, invalidProviderTool(i, tool.ID, "max characters must be positive")
			}
			provider = apiTool{Type: "text_editor_20250728", Name: "str_replace_based_edit_tool", MaxCharacters: tool.MaxCharacters}
		case "anthropic.memory_20250818":
			provider.Type, provider.Name = "memory_20250818", "memory"
		case "anthropic.web_fetch_20250910", "anthropic.web_fetch_20260209":
			if err := validateUses(i, tool.ID, tool.MaxUses, tool.MaxContentTokens); err != nil {
				return nil, false, err
			}
			provider = apiTool{Type: strings.TrimPrefix(tool.ID, "anthropic."), Name: "web_fetch", MaxUses: tool.MaxUses, AllowedDomains: tool.AllowedDomains, BlockedDomains: tool.BlockedDomains, Citations: tool.Citations, MaxContentTokens: tool.MaxContentTokens}
		case "anthropic.web_fetch_20260318":
			if err := validateUses(i, tool.ID, tool.MaxUses, tool.MaxContentTokens); err != nil {
				return nil, false, err
			}
			provider = apiTool{Type: strings.TrimPrefix(tool.ID, "anthropic."), Name: "web_fetch", MaxUses: tool.MaxUses, AllowedDomains: tool.AllowedDomains, BlockedDomains: tool.BlockedDomains, Citations: tool.Citations, MaxContentTokens: tool.MaxContentTokens, UseCache: tool.UseCache, ResponseInclusion: tool.ResponseInclusion}
		case "anthropic.web_search_20250305", "anthropic.web_search_20260209":
			if err := validateUses(i, tool.ID, tool.MaxUses, nil); err != nil {
				return nil, false, err
			}
			if tool.UserLocation != nil && tool.UserLocation.Type != "approximate" {
				return nil, false, invalidProviderTool(i, tool.ID, "user location type must be approximate")
			}
			provider = apiTool{Type: strings.TrimPrefix(tool.ID, "anthropic."), Name: "web_search", MaxUses: tool.MaxUses, AllowedDomains: tool.AllowedDomains, BlockedDomains: tool.BlockedDomains, UserLocation: tool.UserLocation}
		case "anthropic.web_search_20260318":
			if err := validateUses(i, tool.ID, tool.MaxUses, nil); err != nil {
				return nil, false, err
			}
			if tool.UserLocation != nil && tool.UserLocation.Type != "approximate" {
				return nil, false, invalidProviderTool(i, tool.ID, "user location type must be approximate")
			}
			provider = apiTool{Type: strings.TrimPrefix(tool.ID, "anthropic."), Name: "web_search", MaxUses: tool.MaxUses, AllowedDomains: tool.AllowedDomains, BlockedDomains: tool.BlockedDomains, UserLocation: tool.UserLocation, ResponseInclusion: tool.ResponseInclusion}
		case "anthropic.tool_search_regex_20251119":
			provider.Type, provider.Name = "tool_search_tool_regex_20251119", "tool_search_tool_regex"
		case "anthropic.tool_search_bm25_20251119":
			provider.Type, provider.Name = "tool_search_tool_bm25_20251119", "tool_search_tool_bm25"
		case "anthropic.advisor_20260301":
			if tool.AdvisorModel == "" {
				return nil, false, invalidProviderTool(i, tool.ID, "model is required")
			}
			if tool.AdvisorMaxUses != nil && *tool.AdvisorMaxUses <= 0 {
				return nil, false, invalidProviderTool(i, tool.ID, "max uses must be positive")
			}
			var caching *apiCachingConfig
			if tool.AdvisorCaching != nil && tool.AdvisorCaching.Enabled {
				caching = &apiCachingConfig{Type: "ephemeral", TTL: "5m"}
			}
			provider = apiTool{Type: "advisor_20260301", Name: "advisor", Model: tool.AdvisorModel, MaxUses: tool.AdvisorMaxUses, Caching: caching}
		default:
			return nil, false, invalidProviderTool(i, tool.ID, "unknown provider tool ID")
		}
		provider.ID = tool.ID
		wire = append(wire, provider)
	}
	return wire, replaySafe, nil
}

func validateUses(index int, id string, maxUses, maxContentTokens *int) error {
	if maxUses != nil && *maxUses <= 0 {
		return invalidProviderTool(index, id, "max uses must be positive")
	}
	if maxContentTokens != nil && *maxContentTokens <= 0 {
		return invalidProviderTool(index, id, "max content tokens must be positive")
	}
	return nil
}

func invalidProviderTool(index int, id, message string) error {
	return &RequestConstructionError{ToolIndex: index, ToolID: id, Message: message}
}

func (m *anthropicLanguageModel) buildRequest(opts GenerateOptions, stream bool) (apiMessagesRequest, []Warning) {
	request, warnings, _, _, _ := m.buildRequestWithOptions(opts, stream)
	return request, warnings
}

func (m *anthropicLanguageModel) buildRequestWithOptions(opts GenerateOptions, stream bool) (apiMessagesRequest, []Warning, ModelOptions, bool, error) {
	system, messages := convertPrompt(opts.Messages)
	temperature, warnings := normalizeTemperature(opts.Temperature)
	warnings = append(warnings, unsupportedWarnings(opts, ModelCapabilitiesForID(m.modelID))...)
	preparedTools := prepareToolsForRequest(opts.Tools, opts.ToolOptions)
	wireTools, replaySafe, err := toolsToWire(preparedTools)
	if err != nil {
		return apiMessagesRequest{}, warnings, ModelOptions{}, false, err
	}
	modelOptions := m.options
	if opts.Thinking != nil {
		modelOptions.Thinking = opts.Thinking
	}
	modelOptions.RequestTools = preparedTools
	toolChoice := normalizeToolChoice(opts.ToolChoice)
	if toolChoice != nil && toolChoice.DisableParallelToolUse {
		modelOptions.DisableParallelToolUse = true
	}
	capabilities, knownModel := modelCapabilities(m.modelID)
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = capabilities.MaxOutputTokens
		if !knownModel {
			warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("The model %q is unknown. The max output tokens have been limited to %d. Set maxTokens explicitly to override this limit.", m.modelID, maxTokens)})
		}
	}
	if modelOptions.Thinking != nil && modelOptions.Thinking.Type == ThinkingTypeEnabled && modelOptions.Thinking.BudgetTokens != 0 && modelOptions.Thinking.BudgetTokens < minimumThinkingBudgetTokens {
		return apiMessagesRequest{}, warnings, ModelOptions{}, false, &RequestConstructionError{
			ToolIndex: -1,
			Message:   fmt.Sprintf("thinking budget must be zero to use the default or at least %d tokens, got %d", minimumThinkingBudgetTokens, modelOptions.Thinking.BudgetTokens),
		}
	}
	thinking, thinkingWarnings := buildThinking(modelOptions.Thinking, &maxTokens, &temperature, &opts.TopK, &opts.TopP)
	warnings = append(warnings, thinkingWarnings...)
	thinking, thinkingWarnings = applyThinkingCapabilityFallbacks(thinking, capabilities, &modelOptions, m.modelID)
	warnings = append(warnings, thinkingWarnings...)
	modelOptions.effectiveThinking = thinking
	if knownModel && thinking != nil && thinking.Type == "enabled" {
		outputTokens := maxTokens - thinking.BudgetTokens
		if thinking.BudgetTokens >= capabilities.MaxOutputTokens {
			return apiMessagesRequest{}, warnings, ModelOptions{}, false, &ThinkingTokenLimitError{
				ModelID:         m.modelID,
				BudgetTokens:    thinking.BudgetTokens,
				OutputTokens:    outputTokens,
				MaxOutputTokens: capabilities.MaxOutputTokens,
			}
		}
	}
	topK := opts.TopK
	topP := opts.TopP
	if capabilities.RejectsSampling {
		if temperature != nil && *temperature != 1.0 {
			warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "temperature is not supported by this Anthropic model"})
		}
		if topK != nil {
			warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "topK is not supported by this Anthropic model"})
		}
		if topP != nil {
			warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "topP is not supported by this Anthropic model"})
		}
		temperature = nil
		topK = nil
		topP = nil
	}
	if isClaudeModel(m.modelID) && temperature != nil && topP != nil {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "topP is ignored when temperature is set for Anthropic models"})
		topP = nil
	}
	if knownModel && maxTokens > capabilities.MaxOutputTokens {
		if opts.MaxTokens > 0 {
			warnings = append(warnings, Warning{Type: "unsupported-setting", Message: fmt.Sprintf("%d (maxTokens + thinking budget) is greater than %s %d max output tokens. The max output tokens have been limited to %d.", maxTokens, m.modelID, capabilities.MaxOutputTokens, capabilities.MaxOutputTokens)})
		}
		maxTokens = capabilities.MaxOutputTokens
	}
	outputConfig, structuredTools, structuredChoice, disableParallel, structuredWarnings := m.buildStructuredOutput(opts.StructuredOutput, opts.ResponseFormat, modelOptions)
	warnings = append(warnings, structuredWarnings...)
	if outputConfig == nil && (modelOptions.Effort != "" || modelOptions.TaskBudget != nil) {
		outputConfig = &OutputConfig{}
	}
	if outputConfig != nil {
		outputConfig.Effort = modelOptions.Effort
		if modelOptions.TaskBudget != nil {
			outputConfig.TaskBudget = &TokenTaskBudget{Type: "tokens", Total: *modelOptions.TaskBudget}
		}
	}
	preparedTools = append(preparedTools, prepareMCPToolsets(modelOptions.MCPServers)...)
	if len(modelOptions.MCPServers) > 0 {
		replaySafe = false
	}
	if len(structuredTools) > 0 {
		preparedTools = append(preparedTools, structuredTools...)
		structuredWire, _, conversionErr := toolsToWire(structuredTools)
		if conversionErr != nil {
			return apiMessagesRequest{}, warnings, ModelOptions{}, false, conversionErr
		}
		wireTools = append(wireTools, structuredWire...)
		modelOptions.RequestTools = preparedTools
	}
	for _, toolset := range prepareMCPToolsets(modelOptions.MCPServers) {
		wireTools = append(wireTools, apiTool{Type: toolset.Type, MCPServerName: toolset.MCPServerName, DefaultConfig: toolset.DefaultConfig, Configs: toolset.Configs})
	}
	if structuredChoice != nil {
		toolChoice = normalizeToolChoice(structuredChoice)
	}
	toolChoice, preparedTools, wireTools, forcedUseWarnings := applyForcedToolUseFallback(toolChoice, capabilities, preparedTools, wireTools, m.modelID)
	warnings = append(warnings, forcedUseWarnings...)
	modelOptions.RequestTools = preparedTools
	if disableParallel {
		modelOptions.DisableParallelToolUse = true
	}
	if modelOptions.DisableParallelToolUse && len(preparedTools) > 0 {
		if toolChoice == nil {
			toolChoice = &ToolChoice{Type: "auto"}
		}
		toolChoice.DisableParallelToolUse = true
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
		TopK:                   topK,
		TopP:                   topP,
		Tools:                  wireTools,
		ToolChoice:             toolChoice,
		Container:              modelOptions.Container,
		MCPServers:             modelOptions.MCPServers,
		Speed:                  modelOptions.Speed,
		InferenceGeo:           modelOptions.InferenceGeo,
		ResponseFormat:         opts.ResponseFormat,
		OutputConfig:           outputConfig,
		Thinking:               thinking,
		ContextManagement:      modelOptions.ContextManagement,
		DisableParallelToolUse: modelOptions.DisableParallelToolUse,
	}, warnings, modelOptions, replaySafe, nil
}

func normalizeToolChoice(choice *ToolChoice) *ToolChoice {
	if choice == nil {
		return nil
	}
	normalized := *choice
	switch normalized.Type {
	case "required":
		if normalized.Name != "" {
			normalized.Type = "tool"
		} else {
			normalized.Type = "any"
		}
	}
	return &normalized
}

// applyThinkingCapabilityFallbacks mirrors the rejects*/effort handling in
// @ai-sdk/anthropic@4.0.79 prepareRequest: models that reject a thinking mode
// get a warned fallback instead of an API error.
func applyThinkingCapabilityFallbacks(thinking *apiThinking, capabilities ModelCapabilities, modelOptions *ModelOptions, modelID string) (*apiThinking, []Warning) {
	var warnings []Warning
	// Effort is clamped before thinking-type conversion so a disabled request
	// with xhigh/max effort still clamps even when disabled itself falls back
	// to between_tools or is omitted below.
	if thinking != nil && thinking.Type == "disabled" && capabilities.RejectsThinkingDisabledAboveHighEffort && isHighEffort(modelOptions.Effort) {
		warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("effort %q is not supported with thinking disabled on model %q; using %q", modelOptions.Effort, modelID, "high")})
		modelOptions.Effort = "high"
	}
	if thinking != nil && thinking.Type == "between_tools" && isHighEffort(modelOptions.Effort) {
		warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("effort %q is not supported with between_tools thinking on model %q; using %q", modelOptions.Effort, modelID, "high")})
		modelOptions.Effort = "high"
	}
	if thinking != nil && thinking.Type == "disabled" && capabilities.RejectsThinkingDisabled {
		if capabilities.SupportsBetweenToolsThinking {
			warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("thinking type %q is not supported by model %q; falling back to %q", "disabled", modelID, "between_tools")})
			thinking = &apiThinking{Type: "between_tools", BlockBinding: thinking.BlockBinding}
		} else {
			warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("thinking type %q is not supported by model %q; omitting thinking configuration", "disabled", modelID)})
			thinking = nil
		}
	}
	if thinking != nil && thinking.Type == "enabled" && capabilities.RejectsBudgetThinking {
		warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("thinking type %q (budget-based) is not supported by model %q; falling back to %q", "enabled", modelID, "adaptive")})
		thinking = &apiThinking{Type: "adaptive", BlockBinding: thinking.BlockBinding}
	}
	return thinking, warnings
}

func isHighEffort(effort string) bool {
	return effort == "xhigh" || effort == "max"
}

// applyForcedToolUseFallback mirrors @ai-sdk/anthropic@4.0.79 prepareRequest:
// models that reject forced tool use fall back to auto. A named tool choice
// keeps only that tool on the wire.
func applyForcedToolUseFallback(toolChoice *ToolChoice, capabilities ModelCapabilities, prepared []Tool, wire []apiTool, modelID string) (*ToolChoice, []Tool, []apiTool, []Warning) {
	if toolChoice == nil || (toolChoice.Type != "required" && toolChoice.Type != "tool" && toolChoice.Type != "any") || !capabilities.RejectsForcedToolUse {
		return toolChoice, prepared, wire, nil
	}
	warning := Warning{Type: "compatibility", Message: fmt.Sprintf("forced tool use (tool_choice %q) is not supported by model %q; falling back to %q", toolChoice.Type, modelID, "auto")}
	fallback := &ToolChoice{Type: "auto", DisableParallelToolUse: toolChoice.DisableParallelToolUse}
	if toolChoice.Type == "tool" && toolChoice.Name != "" {
		keptPrepared := make([]Tool, 0, len(prepared))
		for _, tool := range prepared {
			if tool.Name == toolChoice.Name {
				keptPrepared = append(keptPrepared, tool)
			}
		}
		keptWire := make([]apiTool, 0, len(wire))
		for _, tool := range wire {
			if tool.Name == toolChoice.Name {
				keptWire = append(keptWire, tool)
			}
		}
		if len(keptPrepared) == 0 {
			// Named tool matched nothing: keep the original tool lists so the
			// request still carries usable tools alongside the warning.
			return fallback, prepared, wire, []Warning{warning}
		}
		return fallback, keptPrepared, keptWire, []Warning{warning}
	}
	return fallback, prepared, wire, []Warning{warning}
}

func buildThinking(config *ThinkingConfig, maxTokens *int, temperature **float64, topK **int, topP **float64) (*apiThinking, []Warning) {
	if config == nil {
		return nil, nil
	}
	var warnings []Warning
	var blockBinding *apiThinkingBlockBinding
	if config.BlockBinding != nil {
		switch config.BlockBinding.PrefixMismatchBehavior {
		case "", ThinkingBlockBindingError, ThinkingBlockBindingDropBlock:
			blockBinding = &apiThinkingBlockBinding{PrefixMismatchBehavior: config.BlockBinding.PrefixMismatchBehavior}
		default:
			warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("thinking block binding prefix_mismatch_behavior %q is not supported; omitting block binding", config.BlockBinding.PrefixMismatchBehavior)})
		}
	}
	switch config.Type {
	case ThinkingTypeEnabled:
		budget := config.BudgetTokens
		if budget == 0 {
			budget = minimumThinkingBudgetTokens
			warnings = append(warnings, Warning{Type: "compatibility", Message: "thinking budget is required when thinking is enabled. using default budget of 1024 tokens."})
		}
		*maxTokens += budget
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
		return &apiThinking{Type: "enabled", BudgetTokens: budget, BlockBinding: blockBinding}, warnings
	case ThinkingTypeAdaptive:
		return &apiThinking{Type: "adaptive", Display: string(config.Display), BlockBinding: blockBinding}, nil
	case ThinkingTypeBetweenTools:
		return &apiThinking{Type: "between_tools", BlockBinding: blockBinding}, nil
	case ThinkingTypeDisabled:
		return &apiThinking{Type: "disabled", BlockBinding: blockBinding}, nil
	default:
		// Binding-only recovery: a block binding without a thinking type
		// still sends a thinking object carrying block_binding.
		if blockBinding != nil {
			return &apiThinking{BlockBinding: blockBinding}, nil
		}
		return nil, nil
	}
}

func (m *anthropicLanguageModel) buildStructuredOutput(output *StructuredOutput, responseFormat *ResponseFormat, opts ModelOptions) (*OutputConfig, []Tool, *ToolChoice, bool, []Warning) {
	if output == nil && responseFormat == nil {
		return nil, nil, nil, false, nil
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
	var warnings []Warning
	if mode == StructuredOutputModeJSONTool && schema != nil {
		capabilities := ModelCapabilitiesForID(m.modelID)
		if capabilities.RejectsForcedToolUse && capabilities.StructuredOutput {
			warnings = append(warnings, Warning{Type: "compatibility", Message: fmt.Sprintf("forced tool use (jsonTool structured output) is not supported by model %q; using output format instead", m.modelID)})
			mode = StructuredOutputModeOutputFormat
		}
	}
	sanitized := SanitizeSchema(schema)
	if mode == StructuredOutputModeJSONTool {
		return nil, []Tool{{Name: "json", Description: "Respond with JSON matching the requested schema.", InputSchema: sanitized}}, &ToolChoice{Type: "required", Name: "json", DisableParallelToolUse: true}, true, warnings
	}
	return &OutputConfig{Format: &ResponseFormat{Type: "json_schema", Schema: sanitized}}, nil, nil, false, warnings
}

func prepareMCPToolsets(servers []MCPServer) []Tool {
	if len(servers) == 0 {
		return nil
	}
	toolsets := make([]Tool, 0, len(servers))
	for _, server := range servers {
		if server.Name == "" {
			continue
		}
		toolset := Tool{Type: "mcp_toolset", MCPServerName: server.Name}
		if server.ToolConfiguration != nil {
			config := server.ToolConfiguration
			if len(config.AllowedTools) > 0 {
				toolset.DefaultConfig = &MCPToolConfig{Enabled: boolPtr(false)}
				toolset.Configs = make(map[string]MCPToolConfig, len(config.AllowedTools))
				for _, name := range config.AllowedTools {
					toolset.Configs[name] = MCPToolConfig{Enabled: boolPtr(true)}
				}
			} else if !config.Enabled {
				toolset.DefaultConfig = &MCPToolConfig{Enabled: boolPtr(false)}
			}
		}
		toolsets = append(toolsets, toolset)
	}
	return toolsets
}

func boolPtr(value bool) *bool { return &value }

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

func unsupportedWarnings(opts GenerateOptions, capabilities ModelCapabilities) []Warning {
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
	return warnings
}

func (m *anthropicLanguageModel) execute(ctx context.Context, body []byte, opts ModelOptions, replaySafe bool) (*http.Response, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	var lastErr error
	endpoint, err := messagesEndpoint(m.provider.baseURL)
	if err != nil {
		return nil, err
	}
	for attempt := 0; attempt <= m.provider.retry.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
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
		if !shouldRetryRequest(resp, err, replaySafe) || attempt == m.provider.retry.MaxRetries {
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

func messagesEndpoint(baseURL string) (string, error) {
	u, err := url.ParseRequestURI(baseURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		if err == nil {
			err = errors.New("base URL must include a scheme and host")
		}
		return "", &InvalidBaseURLError{URL: baseURL, Cause: err}
	}
	u = u.JoinPath("messages")
	return u.String(), nil
}

func shouldRetryRequest(resp *http.Response, err error, replaySafe bool) bool {
	if !replaySafe {
		return false
	}
	return shouldRetry(resp, err)
}

func shouldRetry(resp *http.Response, err error) bool {
	if err != nil {
		return isPreSendTransportError(err)
	}
	if resp == nil {
		return false
	}
	return resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
}

func isTerminalStreamPart(part StreamPart) bool {
	switch part.(type) {
	case StreamFinish, StreamError:
		return true
	default:
		return false
	}
}

func timeNewTimer(delay time.Duration) *time.Timer { return time.NewTimer(delay) }

func ModelCapabilitiesForID(modelID string) ModelCapabilities {
	capabilities, _ := modelCapabilities(modelID)
	return capabilities
}

// matchesFamily anchors a versioned model family to the model ID: an exact
// match or a dated/aliased variant (family followed by "-" or "@"). Unlike
// strings.Contains, a custom prefixed ID such as "my-claude-sonnet-4-5" does
// not match family "claude-sonnet-4-5".
func matchesFamily(modelID, family string) bool {
	return modelID == family ||
		strings.HasPrefix(modelID, family+"-") ||
		strings.HasPrefix(modelID, family+"@")
}

// Mirrors getModelCapabilities() in @ai-sdk/anthropic@4.0.79. Check order is
// load-bearing: family names overlap (e.g. "claude-opus-5-5" starts with
// "claude-opus-5-", "claude-fable-5-1" starts with "claude-fable-5-"), so the
// more specific families must match first.
func modelCapabilities(modelID string) (ModelCapabilities, bool) {
	switch {
	case matchesFamily(modelID, "claude-sonnet-5-5"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true, RejectsThinkingDisabled: true, RejectsBudgetThinking: true, RejectsForcedToolUse: true, SupportsBetweenToolsThinking: true}, true
	case matchesFamily(modelID, "claude-opus-5-5"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true, RejectsThinkingDisabled: true, RejectsBudgetThinking: true, RejectsForcedToolUse: true}, true
	case matchesFamily(modelID, "claude-haiku-5-5"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true, RejectsBudgetThinking: true}, true
	case matchesFamily(modelID, "claude-opus-5"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true}, true
	case matchesFamily(modelID, "claude-fable-5-1"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true, RejectsThinkingDisabled: true, RejectsBudgetThinking: true, RejectsForcedToolUse: true}, true
	case matchesFamily(modelID, "claude-fable-5"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true, RejectsThinkingDisabled: true, RejectsBudgetThinking: true}, true
	case matchesFamily(modelID, "claude-opus-4-8"), matchesFamily(modelID, "claude-opus-4-7"), matchesFamily(modelID, "claude-sonnet-5"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true}, true
	case matchesFamily(modelID, "claude-sonnet-4-6"), matchesFamily(modelID, "claude-opus-4-6"):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true}, true
	case matchesFamily(modelID, "claude-sonnet-4-5"), matchesFamily(modelID, "claude-opus-4-5"), matchesFamily(modelID, "claude-haiku-4-5"):
		return ModelCapabilities{MaxOutputTokens: 64000, StructuredOutput: true}, true
	case matchesFamily(modelID, "claude-opus-4-1"):
		return ModelCapabilities{MaxOutputTokens: 32000, StructuredOutput: true}, true
	case sonnet4BaseModelPattern.MatchString(modelID):
		return ModelCapabilities{MaxOutputTokens: 64000}, true
	case opus4BaseModelPattern.MatchString(modelID):
		return ModelCapabilities{MaxOutputTokens: 32000}, true
	case matchesFamily(modelID, "claude-3-haiku"):
		return ModelCapabilities{MaxOutputTokens: 4096}, true
	case legacyClaudeModelPattern.MatchString(modelID):
		return ModelCapabilities{MaxOutputTokens: 4096}, false
	case isClaudeModel(modelID):
		return ModelCapabilities{MaxOutputTokens: 128000, StructuredOutput: true, SupportsAdaptiveThinking: true, RejectsSampling: true, SupportsXhighEffort: true, RejectsThinkingDisabledAboveHighEffort: true}, false
	default:
		return ModelCapabilities{MaxOutputTokens: 4096}, false
	}
}

var (
	sonnet4BaseModelPattern  = regexp.MustCompile(`^claude-sonnet-4(?:-|@)`)
	opus4BaseModelPattern    = regexp.MustCompile(`^claude-opus-4(?:-|@)`)
	legacyClaudeModelPattern = regexp.MustCompile(`^claude-(?:instant(?:-|$)|v?2(?:$|[-.:])|3(?:$|[-.]))`)
)

func isClaudeModel(modelID string) bool {
	return strings.HasPrefix(modelID, "claude-")
}

type apiMessagesResponse struct {
	ID                   string                        `json:"id"`
	Model                string                        `json:"model"`
	Role                 string                        `json:"role"`
	Content              []apiContentBlock             `json:"content"`
	StopReason           string                        `json:"stop_reason"`
	StopSequence         string                        `json:"stop_sequence"`
	Usage                apiUsage                      `json:"usage"`
	Container            *ContainerInfo                `json:"container"`
	ContextManagement    *apiContextManagementResponse `json:"context_management"`
	InputTransformations []InputTransformation         `json:"input_transformations,omitempty"`
	SafeguardResults     []SafeguardResult             `json:"safeguard_results,omitempty"`
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
	InputTokens              *uint64                    `json:"input_tokens"`
	OutputTokens             *uint64                    `json:"output_tokens"`
	CacheCreationInputTokens *uint64                    `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *uint64                    `json:"cache_read_input_tokens"`
	ServiceTier              *string                    `json:"service_tier"`
	InferenceGeo             *string                    `json:"inference_geo"`
	CacheCreation            *CacheCreationUsage        `json:"cache_creation"`
	ServerToolUse            *ServerToolUsage           `json:"server_tool_use"`
	OutputTokensDetails      *OutputTokensDetails       `json:"output_tokens_details"`
	Iterations               []apiUsageIteration        `json:"iterations"`
	Raw                      map[string]json.RawMessage `json:"-"`
	rawBytes                 json.RawMessage
}

type apiUsageIteration struct {
	Type                     string `json:"type"`
	Model                    string `json:"model,omitempty"`
	InputTokens              uint64 `json:"input_tokens"`
	OutputTokens             uint64 `json:"output_tokens"`
	CacheCreationInputTokens uint64 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     uint64 `json:"cache_read_input_tokens,omitempty"`
}

func (u *apiUsage) UnmarshalJSON(data []byte) error {
	type plain apiUsage
	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*u = apiUsage(decoded)
	u.rawBytes = append(u.rawBytes[:0], data...)
	return json.Unmarshal(data, &u.Raw)
}

func parseGenerateResponse(body []byte) (*GenerateResult, error) {
	var response apiMessagesResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, err
	}
	usage, err := convertAPIUsage(response.Usage)
	if err != nil {
		return nil, err
	}
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
	if response.InputTransformations != nil {
		metadata["input_transformations"] = response.InputTransformations
	}
	if response.SafeguardResults != nil {
		metadata["safeguard_results"] = response.SafeguardResults
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

func convertAPIUsage(usage apiUsage) (Usage, error) {
	raw := append(json.RawMessage(nil), usage.rawBytes...)
	if raw == nil {
		var err error
		raw, err = json.Marshal(usage.Raw)
		if err != nil {
			return Usage{}, err
		}
	}
	return usageFromAPIUsage(apiResponseUsage{
		InputTokens: uint64Value(usage.InputTokens), OutputTokens: uint64Value(usage.OutputTokens),
		CacheCreationInputTokens: uint64Value(usage.CacheCreationInputTokens), CacheReadInputTokens: uint64Value(usage.CacheReadInputTokens),
		ServiceTier: stringValue(usage.ServiceTier), InferenceGeo: stringValue(usage.InferenceGeo), CacheCreation: usage.CacheCreation,
		ServerToolUse: usage.ServerToolUse, OutputTokensDetails: usage.OutputTokensDetails, Iterations: usage.Iterations, Raw: raw,
	})
}

func uint64Value(value *uint64) uint64 {
	if value == nil {
		return 0
	}
	return *value
}
func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
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
	skipWebSources := hasWebSearchResults(blocks)
	for _, block := range blocks {
		switch block.Type {
		case "text":
			content = append(content, TextContent{Text: block.Text, Citations: normalizeCitations(block.Citations), Raw: block.Raw})
			for _, citation := range block.Citations {
				if citation.Type == "web_search_result_location" && skipWebSources {
					continue
				}
				if source, ok := citationToSource(citation); ok {
					content = append(content, source)
				}
			}
		case "thinking":
			content = append(content, ReasoningContent{Type: "reasoning", Text: block.Thinking, Signature: block.Signature})
		case "redacted_thinking":
			content = append(content, ReasoningContent{Type: "redacted_reasoning", Text: block.Data})
		case "compaction":
			text := compactionBlockText(block)
			if text == "" {
				continue
			}
			content = append(content, CompactionContent{Text: text, Signature: block.Signature})
		case "fallback":
			content = append(content, FallbackContent{FromModel: fallbackModel(block.From), ToModel: fallbackModel(block.To)})
		case "source", "web_search_result", "web_fetch_result":
			content = append(content, SourceContent{SourceType: block.Type, ID: block.ID, URL: block.URL, Title: block.Title, MediaType: block.MediaType, Filename: block.Filename})
		case "tool_use":
			if block.ToolsetName != "" {
				content = append(content, ToolCallContent{Type: "tool-call", ToolCallID: block.ID, ToolName: block.Name, Input: toToolsetMemberInput(block.Name, block.Input), ToolsetName: block.ToolsetName, ProviderExecuted: block.ProviderExecuted, Dynamic: block.Dynamic, Caller: block.Caller, Raw: block.Raw})
				continue
			}
			content = append(content, ToolCallContent{Type: "tool-call", ToolCallID: block.ID, ToolName: block.Name, Input: anyToRawJSON(block.Input), ProviderExecuted: block.ProviderExecuted, Dynamic: block.Dynamic, Caller: block.Caller, Raw: block.Raw})
		case "server_tool_use":
			content = append(content, ServerToolUseContent{ID: block.ID, Name: block.Name, Input: anyToRawJSON(block.Input), Dynamic: block.Dynamic, Caller: block.Caller, Raw: block.Raw})
		case "mcp_tool_use":
			content = append(content, MCPToolUseContent{ID: block.ID, Name: block.Name, ServerName: block.ServerName, Input: anyToRawJSON(block.Input), Dynamic: block.Dynamic, Raw: block.Raw})
		case "mcp_tool_result", "web_fetch_tool_result", "web_search_tool_result", "code_execution_tool_result", "bash_code_execution_tool_result", "text_editor_code_execution_tool_result", "tool_search_tool_result", "advisor_tool_result", "tool_result":
			content = append(content, ToolResultContent{Type: block.Type, ToolCallID: block.ToolUseID, Result: parseToolResultParts(block.Content), IsError: block.IsError, ProviderExecuted: block.Type != "tool_result" || block.ProviderExecuted, Dynamic: block.Dynamic, ToolsetName: block.ToolsetName, Caller: block.Caller, Raw: block.Raw, RawContent: block.Raw["content"]})
		}
	}
	return content
}

// isWebSearchResultsBlock reports whether a response block carries web search
// results whose presence makes web_search_result_location citations redundant.
// Only web_search_tool_result blocks with non-empty content suppress
// citations; bare web_search_result source blocks do not. This is the upstream
// generate pre-scan semantic, shared by generate and stream for parity.
func isWebSearchResultsBlock(block apiContentBlock) bool {
	return block.Type == "web_search_tool_result" && blockHasArrayContent(block)
}

// hasWebSearchResults mirrors the doGenerate pre-scan: web_search citations
// are redundant when the response already carries web_search results.
func hasWebSearchResults(blocks []apiContentBlock) bool {
	for _, block := range blocks {
		if isWebSearchResultsBlock(block) {
			return true
		}
	}
	return false
}

func blockHasArrayContent(block apiContentBlock) bool {
	raw, err := json.Marshal(block.Content)
	if err != nil {
		return false
	}
	var items []json.RawMessage
	return json.Unmarshal(raw, &items) == nil && len(items) > 0
}

// compactionBlockText accepts the response `content` string as well as the
// replay `text` field; an empty compaction block carries nothing to surface.
func compactionBlockText(block apiContentBlock) string {
	if block.Text != "" {
		return block.Text
	}
	if text, ok := block.Content.(string); ok {
		return text
	}
	return ""
}

func fallbackModel(endpoint *apiFallbackEndpoint) string {
	if endpoint == nil {
		return ""
	}
	return endpoint.Model
}

// citationToSource maps a citation to a SourceContent part: URL-based for web
// and fetch search locations, document-based for document locations. Unknown
// citation shapes (including empty pass-through citations) yield no source.
func citationToSource(citation Citation) (SourceContent, bool) {
	switch citation.Type {
	case "web_search_result_location":
		if citation.URL == "" {
			return SourceContent{}, false
		}
		return SourceContent{SourceType: "url", URL: citation.URL, Title: citation.Title}, true
	case "search_result_location":
		if citation.Source == "" {
			return SourceContent{}, false
		}
		return SourceContent{SourceType: "url", URL: citation.Source, Title: citation.Title}, true
	case "page_location", "char_location", "content_block_location":
		if citation.FileID == "" && citation.DocumentTitle == "" {
			return SourceContent{}, false
		}
		return SourceContent{SourceType: "document", ID: citation.FileID, Title: citation.DocumentTitle}, true
	default:
		return SourceContent{}, false
	}
}

// toToolsetMemberInput exposes a toolset member call on the toolset tool by
// injecting the member name as the `action` property of the member input.
// Object input gains `action` only when absent (a caller-supplied action is
// kept as-is); non-object input is forwarded unchanged.
func toToolsetMemberInput(memberName string, input any) json.RawMessage {
	actionOnly, _ := json.Marshal(map[string]any{"action": memberName})
	if input == nil {
		return actionOnly
	}
	raw, err := json.Marshal(input)
	if err != nil {
		return actionOnly
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return raw
	}
	if _, exists := fields["action"]; exists {
		return raw
	}
	fields["action"] = memberName
	merged, err := json.Marshal(fields)
	if err != nil {
		return raw
	}
	return merged
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
			var document apiContentBlock
			if content, err := json.Marshal(block.Content); err == nil {
				_ = json.Unmarshal(content, &document)
			}
			source := DocumentSource{}
			if document.Source != nil {
				source = DocumentSource{Type: document.Source.Type, MediaType: document.Source.MediaType, Data: document.Source.Data, URL: document.Source.URL}
			}
			parts = append(parts, WebFetchResult{Type: block.Type, URL: block.URL, RetrievedAt: block.RetrievedAt, Content: DocumentContent{Source: source}})
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
		default:
			parts = append(parts, ToolResultUnknown{Type: block.Type, Raw: block.Raw})
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
		if int64(len(body)) > limit {
			return body[:limit], true, err
		}
		return body, false, err
	}
	if int64(len(body)) > limit {
		return body[:limit], true, nil
	}
	return body, false, nil
}

func apiCallError(resp *http.Response, body []byte, truncated bool, cause error, replaySafe bool) error {
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
	return &APICallError{Status: status, Message: message, Type: errorType, Retryable: replaySafe && (status == http.StatusTooManyRequests || status >= 500), Headers: headers, RequestID: requestID, Body: body, Truncated: truncated, Cause: cause}
}

func nilResponseBodyError(resp *http.Response, replaySafe bool) error {
	return &APICallError{
		Status:    resp.StatusCode,
		Message:   "HTTP response has nil body",
		Retryable: retryableIncompleteResponse(resp.StatusCode, replaySafe),
		Headers:   cloneHeader(resp.Header),
		RequestID: responseRequestID(resp.Header),
	}
}

func retryableIncompleteResponse(status int, replaySafe bool) bool {
	return replaySafe && ((status >= 200 && status < 300) || status == http.StatusTooManyRequests || status >= 500)
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
		for key, values := range header {
			if strings.EqualFold(key, name) && len(values) > 0 && values[0] != "" {
				return values[0]
			}
		}
	}
	return ""
}

func sendStreamPart(ctx context.Context, out chan<- StreamPart, part StreamPart) bool {
	select {
	case out <- part:
		return true
	case <-ctx.Done():
		return false
	}
}

// parseSSEStream reads complete SSE records without Scanner's token limit. It
// honors CRLF, comments, optional single-space field separation, repeated data
// fields, unknown fields, and an unterminated final record.
func parseSSEStream(ctx context.Context, reader io.Reader, out chan<- StreamPart, response *StreamResponse) {
	parseSSEStreamWithLimit(ctx, reader, out, response, defaultMaxResponseBodyBytes)
}

func parseSSEStreamWithLimit(ctx context.Context, reader io.Reader, out chan<- StreamPart, response *StreamResponse, diagnosticLimit int64) {
	parseSSEStreamWithMetadata(ctx, reader, out, response, diagnosticLimit, streamErrorMetadata{})
}

type streamErrorMetadata struct {
	headers   http.Header
	requestID string
}

func parseSSEStreamWithMetadata(ctx context.Context, reader io.Reader, out chan<- StreamPart, response *StreamResponse, diagnosticLimit int64, metadata streamErrorMetadata) {
	if diagnosticLimit <= 0 {
		diagnosticLimit = defaultMaxResponseBodyBytes
	}
	buffered := bufio.NewReader(reader)
	var eventType string
	var data []string
	var rawEvent []byte
	var rawTruncated bool
	appendRaw := func(raw string) {
		if rawTruncated {
			return
		}
		remaining := diagnosticLimit - int64(len(rawEvent))
		if remaining <= 0 {
			rawTruncated = true
			return
		}
		if int64(len(raw)) > remaining {
			rawEvent = append(rawEvent, raw[:remaining]...)
			rawTruncated = true
			return
		}
		rawEvent = append(rawEvent, raw...)
	}
	state := streamParserState{blocks: make(map[int]*streamBlockState), finish: StreamFinish{FinishReason: FinishReasonUnknown}}
	dispatch := func() bool {
		if eventType == "" && len(data) == 0 {
			return true
		}
		if state.stopped {
			sendStreamPart(ctx, out, StreamError{Err: localStreamError("event after terminal message_stop", rawEvent, rawTruncated, nil, metadata)})
			return false
		}
		parts, err := state.handle(eventType, []byte(strings.Join(data, "\n")), rawEvent, rawTruncated, metadata)
		if err != nil {
			var apiErr *APICallError
			if !errors.As(err, &apiErr) {
				err = localStreamError(err.Error(), rawEvent, rawTruncated, err, metadata)
			}
			sendStreamPart(ctx, out, StreamError{Err: err})
			return false
		}
		for _, part := range parts {
			if metadata, ok := part.(StreamResponseMetadata); ok && response != nil {
				response.ID = metadata.ID
				response.ModelID = metadata.ModelID
			}
			if !sendStreamPart(ctx, out, part) {
				return false
			}
		}
		eventType, data, rawEvent, rawTruncated = "", nil, nil, false
		return !state.stopped
	}
	for {
		line, err := buffered.ReadString('\n')
		if err != nil && err != io.EOF {
			if len(line) > 0 {
				appendRaw(line)
			}
			sendStreamPart(ctx, out, StreamError{Err: localStreamError(err.Error(), rawEvent, rawTruncated, err, metadata)})
			return
		}
		if len(line) > 0 {
			appendRaw(line)
			line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
			if line == "" {
				if !dispatch() {
					return
				}
			} else if line[0] != ':' {
				field, value, found := strings.Cut(line, ":")
				if !found {
					value = ""
				} else if strings.HasPrefix(value, " ") {
					value = value[1:]
				}
				switch field {
				case "event":
					eventType = value
				case "data":
					data = append(data, value)
				}
			}
		}
		if err != nil {
			if !dispatch() {
				return
			}
			if !state.stopped {
				sendStreamPart(ctx, out, StreamError{Err: localStreamError("anthropic stream ended before message_stop", rawEvent, rawTruncated, io.ErrUnexpectedEOF, metadata)})
			}
			return
		}
	}
}

type streamBlockState struct {
	typ, id, name             string
	providerExecuted, dynamic bool
	providerMetadata          ProviderMetadata
	toolsetName, memberName   string
	initialInput              []byte
	hasInputDelta             bool
	input                     strings.Builder
}

type streamParserState struct {
	started, messageDeltaSeen, stopped bool
	messageID, model                   string
	initialUsage                       apiUsage
	blocks                             map[int]*streamBlockState
	finish                             StreamFinish
	hasWebSearchResults                bool
	fallbackWebCitations               []Citation
	inputTransformations               []InputTransformation
	safeguardResults                   []SafeguardResult
}

func (s *streamParserState) handle(eventType string, data, raw []byte, truncated bool, metadata streamErrorMetadata) ([]StreamPart, error) {
	if eventType == "" || eventType == "ping" {
		return nil, nil
	}
	if s.messageDeltaSeen && eventType != "message_stop" {
		return nil, localStreamError(fmt.Sprintf("%s after message_delta", eventType), raw, truncated, nil, metadata)
	}
	if eventType == "error" {
		return nil, streamingAPIError(data, raw, truncated, metadata)
	}
	if !s.started && eventType != "message_start" {
		return nil, localStreamError(fmt.Sprintf("%s before message_start", eventType), raw, truncated, nil, metadata)
	}
	switch eventType {
	case "message_start":
		var message struct {
			Message struct {
				ID, Model            string
				Usage                apiUsage              `json:"usage"`
				Content              []apiContentBlock     `json:"content,omitempty"`
				InputTransformations []InputTransformation `json:"input_transformations,omitempty"`
			} `json:"message"`
		}
		if err := json.Unmarshal(data, &message); err != nil {
			return nil, localStreamError("decode message_start", raw, truncated, err, metadata)
		}
		if message.Message.ID == "" || message.Message.Model == "" {
			return nil, fmt.Errorf("message_start missing id or model")
		}
		if s.started {
			if s.messageID == message.Message.ID && s.model == message.Message.Model && reflect.DeepEqual(s.initialUsage, message.Message.Usage) {
				return nil, nil
			}
			return nil, fmt.Errorf("conflicting duplicate message_start")
		}
		s.started, s.messageID, s.model, s.initialUsage = true, message.Message.ID, message.Message.Model, message.Message.Usage
		if message.Message.InputTransformations != nil {
			s.inputTransformations = message.Message.InputTransformations
		}
		parts := []StreamPart{StreamStart{}, StreamResponseMetadata{ID: s.messageID, ModelID: s.model}}
		for _, block := range message.Message.Content {
			if block.Type != "fallback" {
				continue
			}
			parts = append(parts, StreamRaw{Event: FallbackContent{FromModel: fallbackModel(block.From), ToModel: fallbackModel(block.To)}})
		}
		return parts, nil
	case "content_block_start":
		var event struct {
			Index        *int            `json:"index"`
			ContentBlock apiContentBlock `json:"content_block"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, err
		}
		if event.Index == nil || *event.Index < 0 {
			return nil, fmt.Errorf("content_block_start missing or invalid index")
		}
		if _, exists := s.blocks[*event.Index]; exists {
			return nil, fmt.Errorf("duplicate content block index %d", *event.Index)
		}
		id := fmt.Sprint(*event.Index)
		block := &streamBlockState{typ: event.ContentBlock.Type, id: event.ContentBlock.ID, name: event.ContentBlock.Name, providerExecuted: event.ContentBlock.ProviderExecuted, dynamic: event.ContentBlock.Dynamic, providerMetadata: event.ContentBlock.ProviderMetadata, toolsetName: event.ContentBlock.ToolsetName, memberName: event.ContentBlock.Name}
		s.blocks[*event.Index] = block
		switch event.ContentBlock.Type {
		case "thinking", "redacted_thinking":
			return []StreamPart{StreamReasoningStart{ID: id}}, nil
		case "fallback":
			return []StreamPart{StreamRaw{Event: FallbackContent{FromModel: fallbackModel(event.ContentBlock.From), ToModel: fallbackModel(event.ContentBlock.To)}}}, nil
		case "compaction":
			parts := []StreamPart{StreamTextStart{ID: id}}
			if text := compactionBlockText(event.ContentBlock); text != "" {
				parts = append(parts, StreamTextDelta{ID: id, Text: text})
			}
			return parts, nil
		case "input_json", "tool_use", "server_tool_use", "mcp_tool_use":
			if event.ContentBlock.Input != nil {
				block.initialInput = append([]byte(nil), anyToRawJSON(event.ContentBlock.Input)...)
			}
			return []StreamPart{StreamToolInputStart{ID: id, ToolName: block.name, ProviderExecuted: block.providerExecuted || block.typ == "server_tool_use" || block.typ == "mcp_tool_use", Dynamic: block.dynamic}}, nil
		case "source", "web_search_result", "web_fetch_result":
			return []StreamPart{StreamSource{SourceContent: SourceContent{SourceType: event.ContentBlock.Type, ID: event.ContentBlock.ID, URL: event.ContentBlock.URL, Title: event.ContentBlock.Title, MediaType: event.ContentBlock.MediaType, Filename: event.ContentBlock.Filename}, ProviderMetadata: event.ContentBlock.ProviderMetadata}}, nil
		case "tool_result", "mcp_tool_result", "web_fetch_tool_result", "web_search_tool_result", "code_execution_tool_result", "bash_code_execution_tool_result", "text_editor_code_execution_tool_result", "tool_search_tool_result", "advisor_tool_result":
			if isWebSearchResultsBlock(event.ContentBlock) {
				s.hasWebSearchResults = true
			}
			return []StreamPart{StreamToolResult{ToolResultContent: ToolResultContent{Type: event.ContentBlock.Type, ToolCallID: event.ContentBlock.ToolUseID, Result: parseToolResultParts(event.ContentBlock.Content), IsError: event.ContentBlock.IsError, ProviderExecuted: event.ContentBlock.Type != "tool_result" || event.ContentBlock.ProviderExecuted, Dynamic: event.ContentBlock.Dynamic, ToolsetName: event.ContentBlock.ToolsetName, ProviderMetadata: event.ContentBlock.ProviderMetadata}}}, nil
		case "text":
			return []StreamPart{StreamTextStart{ID: id}}, nil
		default:
			delete(s.blocks, *event.Index)
			return nil, fmt.Errorf("unsupported content block type %q", event.ContentBlock.Type)
		}
	case "content_block_delta":
		var event struct {
			Index *int `json:"index"`
			Delta struct {
				Type        string     `json:"type"`
				Text        string     `json:"text"`
				Thinking    string     `json:"thinking"`
				Signature   string     `json:"signature"`
				PartialJSON string     `json:"partial_json"`
				Content     *string    `json:"content"`
				Citations   []Citation `json:"citations"`
				Citation    *Citation  `json:"citation"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, err
		}
		if event.Index == nil || *event.Index < 0 {
			return nil, fmt.Errorf("content_block_delta missing or invalid index")
		}
		block, ok := s.blocks[*event.Index]
		if !ok {
			return nil, fmt.Errorf("delta for inactive content block %d", *event.Index)
		}
		id := fmt.Sprint(*event.Index)
		switch event.Delta.Type {
		case "text_delta":
			if block.typ != "text" {
				return nil, fmt.Errorf("text delta for %s block", block.typ)
			}
			return []StreamPart{StreamTextDelta{ID: id, Text: event.Delta.Text, Citations: normalizeCitations(event.Delta.Citations)}}, nil
		case "citations_delta":
			if block.typ != "text" {
				return nil, fmt.Errorf("citation delta for %s block", block.typ)
			}
			citations := event.Delta.Citations
			if event.Delta.Citation != nil {
				citations = append(citations, *event.Delta.Citation)
			}
			citations = normalizeCitations(citations)
			parts := []StreamPart{StreamTextDelta{ID: id, Citations: citations}}
			for _, citation := range citations {
				if citation.Type == "web_search_result_location" {
					s.fallbackWebCitations = append(s.fallbackWebCitations, citation)
					continue
				}
				if source, ok := citationToSource(citation); ok {
					parts = append(parts, StreamSource{SourceContent: source})
				}
			}
			return parts, nil
		case "thinking_delta":
			if block.typ != "thinking" && block.typ != "redacted_thinking" {
				return nil, fmt.Errorf("thinking delta for %s block", block.typ)
			}
			return []StreamPart{StreamReasoningDelta{ID: id, Delta: ThinkingDelta{Thinking: event.Delta.Thinking}}}, nil
		case "signature_delta":
			if block.typ != "thinking" && block.typ != "redacted_thinking" {
				return nil, fmt.Errorf("signature delta for %s block", block.typ)
			}
			return []StreamPart{StreamReasoningDelta{ID: id, Delta: SignatureDelta{Signature: event.Delta.Signature}}}, nil
		case "input_json_delta":
			if block.typ != "input_json" && block.typ != "tool_use" && block.typ != "server_tool_use" {
				return nil, fmt.Errorf("input delta for %s block", block.typ)
			}
			block.hasInputDelta = true
			block.input.WriteString(event.Delta.PartialJSON)
			return []StreamPart{StreamToolInputDelta{ID: id, Delta: InputJSONDelta{PartialJSON: event.Delta.PartialJSON}}}, nil
		case "mcp_tool_use_delta":
			if block.typ != "mcp_tool_use" {
				return nil, fmt.Errorf("mcp input delta for %s block", block.typ)
			}
			block.hasInputDelta = true
			block.input.WriteString(event.Delta.PartialJSON)
			return []StreamPart{StreamToolInputDelta{ID: id, Delta: InputJSONDelta{PartialJSON: event.Delta.PartialJSON}}}, nil
		case "compaction_delta":
			if block.typ != "compaction" {
				return nil, fmt.Errorf("compaction delta for %s block", block.typ)
			}
			if event.Delta.Content == nil {
				return nil, nil
			}
			return []StreamPart{StreamTextDelta{ID: id, Text: *event.Delta.Content}}, nil
		default:
			return nil, fmt.Errorf("unsupported delta type %q", event.Delta.Type)
		}
	case "content_block_stop":
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(data, &payload); err != nil || payload == nil {
			return nil, fmt.Errorf("content_block_stop payload is not an object")
		}
		var event struct {
			Index *int `json:"index"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, err
		}
		if event.Index == nil || *event.Index < 0 {
			return nil, fmt.Errorf("content_block_stop missing or invalid index")
		}
		block, ok := s.blocks[*event.Index]
		if !ok {
			return nil, fmt.Errorf("stop for inactive content block %d", *event.Index)
		}
		delete(s.blocks, *event.Index)
		id := fmt.Sprint(*event.Index)
		switch block.typ {
		case "thinking", "redacted_thinking":
			return []StreamPart{StreamReasoningEnd{ID: id}}, nil
		case "text":
			return []StreamPart{StreamTextEnd{ID: id}}, nil
		case "fallback":
			return nil, nil
		case "compaction":
			return []StreamPart{StreamTextEnd{ID: id}}, nil
		case "input_json", "tool_use", "server_tool_use", "mcp_tool_use":
			raw := block.initialInput
			if block.hasInputDelta {
				raw = []byte(block.input.String())
			}
			if block.toolsetName != "" && block.typ == "tool_use" {
				var decoded any
				if len(raw) > 0 {
					if err := json.Unmarshal(raw, &decoded); err != nil {
						return nil, fmt.Errorf("tool input for block %d is not a valid JSON object", *event.Index)
					}
				}
				raw = toToolsetMemberInput(block.memberName, decoded)
			}
			if len(raw) == 0 {
				raw = []byte("{}")
			}
			var object map[string]json.RawMessage
			if json.Unmarshal(raw, &object) != nil || object == nil {
				return nil, fmt.Errorf("tool input for block %d is not a valid JSON object", *event.Index)
			}
			parts := []StreamPart{StreamToolInputEnd{ID: id}}
			if block.typ != "input_json" {
				call := ToolCallContent{Type: streamToolCallType(block.typ), ToolCallID: block.id, ToolName: block.name, Input: append(json.RawMessage(nil), raw...), ProviderExecuted: block.providerExecuted || block.typ != "tool_use", Dynamic: block.dynamic, ProviderMetadata: block.providerMetadata}
				if block.toolsetName != "" && block.typ == "tool_use" {
					call.ToolsetName = block.toolsetName
				}
				parts = append(parts, StreamToolCall{ToolCallContent: call})
			}
			return parts, nil
		case "source", "web_search_result", "web_fetch_result", "tool_result", "mcp_tool_result", "web_fetch_tool_result", "web_search_tool_result", "code_execution_tool_result", "bash_code_execution_tool_result", "text_editor_code_execution_tool_result", "tool_search_tool_result", "advisor_tool_result":
			return nil, nil
		}
	case "message_delta":
		var event struct {
			Delta struct {
				StopReason       string            `json:"stop_reason"`
				SafeguardResults []SafeguardResult `json:"safeguard_results,omitempty"`
			} `json:"delta"`
			Usage                apiUsage              `json:"usage"`
			InputTransformations []InputTransformation `json:"input_transformations,omitempty"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, err
		}
		if len(s.blocks) != 0 {
			return nil, fmt.Errorf("message_delta with active content blocks")
		}
		s.messageDeltaSeen = true
		if event.InputTransformations != nil {
			s.inputTransformations = event.InputTransformations
		}
		if event.Delta.SafeguardResults != nil {
			s.safeguardResults = event.Delta.SafeguardResults
		}
		usage := mergeAPIUsage(s.initialUsage, event.Usage)
		converted, err := convertAPIUsage(usage)
		if err != nil {
			return nil, err
		}
		s.finish = StreamFinish{FinishReason: FinishReasonFromStopReason(event.Delta.StopReason), Usage: converted}
		return nil, nil
	case "message_stop":
		if len(s.blocks) != 0 {
			return nil, fmt.Errorf("message_stop with active content blocks")
		}
		s.stopped = true
		var parts []StreamPart
		if !s.hasWebSearchResults {
			for _, citation := range s.fallbackWebCitations {
				if source, ok := citationToSource(citation); ok {
					parts = append(parts, StreamSource{SourceContent: source})
				}
			}
		}
		finish := s.finish
		if s.inputTransformations != nil || s.safeguardResults != nil {
			meta := ProviderMetadata{}
			if s.inputTransformations != nil {
				meta["inputTransformations"] = s.inputTransformations
			}
			if s.safeguardResults != nil {
				meta["safeguardResults"] = s.safeguardResults
			}
			finish.ProviderMetadata = meta
		}
		return append(parts, finish), nil
	default:
		return nil, fmt.Errorf("unsupported event type %q", eventType)
	}
	return nil, nil
}

func mergeAPIUsage(previous, later apiUsage) apiUsage {
	merged := previous
	if later.InputTokens != nil {
		merged.InputTokens = later.InputTokens
	}
	if later.OutputTokens != nil {
		merged.OutputTokens = later.OutputTokens
	}
	if later.CacheCreationInputTokens != nil {
		merged.CacheCreationInputTokens = later.CacheCreationInputTokens
	}
	if later.CacheReadInputTokens != nil {
		merged.CacheReadInputTokens = later.CacheReadInputTokens
	}
	if later.ServiceTier != nil {
		merged.ServiceTier = later.ServiceTier
	}
	if later.InferenceGeo != nil {
		merged.InferenceGeo = later.InferenceGeo
	}
	if later.CacheCreation != nil {
		merged.CacheCreation = later.CacheCreation
	}
	if later.ServerToolUse != nil {
		merged.ServerToolUse = later.ServerToolUse
	}
	if later.OutputTokensDetails != nil {
		merged.OutputTokensDetails = later.OutputTokensDetails
	}
	if later.Iterations != nil {
		merged.Iterations = later.Iterations
	}
	merged.Raw = make(map[string]json.RawMessage, len(previous.Raw)+len(later.Raw))
	for key, value := range previous.Raw {
		merged.Raw[key] = append([]byte(nil), value...)
	}
	for key, value := range later.Raw {
		merged.Raw[key] = append([]byte(nil), value...)
	}
	// A merged stream has no single original usage object, so Raw is a
	// canonical serialization of the combined fields.
	merged.rawBytes = nil
	return merged
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

func streamingAPIError(data, raw []byte, truncated bool, metadata streamErrorMetadata) error {
	var event struct {
		Error APIError `json:"error"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return localStreamError("decode error event", raw, truncated, err, metadata)
	}
	if event.Error.Type == "" || event.Error.Message == "" {
		return localStreamError("invalid error event payload", raw, truncated, nil, metadata)
	}
	status, known := anthropicErrorTypeStatus(event.Error.Type)
	return &APICallError{Status: status, SyntheticStatus: known, Type: event.Error.Type, Message: event.Error.Message, Retryable: known && (status == http.StatusTooManyRequests || status >= 500), Headers: cloneHeader(metadata.headers), RequestID: metadata.requestID, Body: append([]byte(nil), raw...), Truncated: truncated}
}

func anthropicErrorTypeStatus(errorType string) (int, bool) {
	switch errorType {
	case "invalid_request_error":
		return http.StatusBadRequest, true
	case "authentication_error":
		return http.StatusUnauthorized, true
	case "permission_error":
		return http.StatusForbidden, true
	case "not_found_error":
		return http.StatusNotFound, true
	case "request_too_large":
		return http.StatusRequestEntityTooLarge, true
	case "rate_limit_error":
		return http.StatusTooManyRequests, true
	case "api_error":
		return http.StatusInternalServerError, true
	case "overloaded_error":
		return 529, true
	default:
		return 0, false
	}
}

func localStreamError(message string, raw []byte, truncated bool, cause error, metadata streamErrorMetadata) *APICallError {
	return &APICallError{Message: message, Headers: cloneHeader(metadata.headers), RequestID: metadata.requestID, Body: append([]byte(nil), raw...), Truncated: truncated, Cause: cause}
}
