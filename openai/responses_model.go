package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// openaiResponsesModel implements the OpenAI Responses API.
type openaiResponsesModel struct {
	provider *openaiProvider
	modelID  string
	// store is the value of Store from the most recent DoStream/DoGenerate
	// call, captured for use by the streaming goroutine. nil means Store
	// was not set (server default).
	store *bool
}

func newResponsesModel(p *openaiProvider, modelID string) ResponsesModel {
	return &openaiResponsesModel{provider: p, modelID: modelID}
}

func (m *openaiResponsesModel) ModelID() string  { return m.modelID }
func (m *openaiResponsesModel) Provider() string { return "openai.responses" }
func (m *openaiResponsesModel) SupportURLs() map[string][]*regexp.Regexp { return nil }

// DoGenerate performs a non-streaming Responses call.
func (m *openaiResponsesModel) DoGenerate(ctx context.Context, opts ResponsesGenerateOptions) (*ResponsesGenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	body, warnings, err := m.buildResponsesRequest(opts)
	if err != nil {
		return nil, err
	}
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointResponses, encoded, opts.Headers)
	if err != nil {
		return nil, err
	}
	result, err := m.parseResponsesResponse(resp.Body, encoded)
	if err != nil {
		return nil, err
	}
	result.Warnings = append(warnings, result.Warnings...)
	return result, nil
}

// DoStream performs a streaming Responses call.
func (m *openaiResponsesModel) DoStream(ctx context.Context, opts ResponsesStreamOptions) (*ResponsesStreamResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	body, warnings, err := m.buildResponsesRequest(opts.ResponsesGenerateOptions)
	if err != nil {
		return nil, err
	}
	body["stream"] = true
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	reqBody := append([]byte(nil), encoded...)
	resp, err := m.provider.executeStream(ctx, endpointResponses, encoded, opts.Headers)
	if err != nil {
		return nil, err
	}
	parts := make(chan StreamPart)
	sresp := &StreamResponse{Headers: resp.Headers.Clone()}
	result := &ResponsesStreamResult{Stream: parts, Parts: parts, Request: RequestMetadata{Body: reqBody}, Response: sresp}
	m.store = opts.Store
	go m.runResponsesStream(ctx, resp, parts, sresp, warnings, opts)
	return result, nil
}

// buildResponsesRequest assembles the OpenAI Responses API request body.
func (m *openaiResponsesModel) buildResponsesRequest(opts ResponsesGenerateOptions) (map[string]any, []Warning, error) {
	caps := ModelCapabilitiesForID(m.modelID)
	chatOptions, deprecatedWarnings := mergeResponsesOpenAIOptions(opts.ProviderOptions)
	warnings := append([]Warning(nil), deprecatedWarnings...)
	body := map[string]any{"model": m.modelID}

	// Instructions / system message.
	if opts.Instructions != "" {
		body["instructions"] = opts.Instructions
	}
	for _, msg := range opts.Messages {
		if sm, ok := msg.(SystemMessage); ok {
			mode := "system"
			if v, ok := chatOptions["systemMessageMode"].(string); ok {
				mode = v
			}
			switch mode {
			case "remove":
				warnings = append(warnings, Warning{Type: "other", Message: "system messages are dropped under systemMessageMode=remove"})
			case "developer":
				body["instructions"] = sm.Content
			default:
				// Append to input items.
			}
		}
	}

	// Conversation / previous response.
	if opts.Conversation != nil && opts.PreviousResponseID != nil {
		warnings = append(warnings, Warning{Type: "other", Message: "Conversation and PreviousResponseID are mutually exclusive; using PreviousResponseID"})
		body["previous_response_id"] = *opts.PreviousResponseID
	} else if opts.PreviousResponseID != nil {
		body["previous_response_id"] = *opts.PreviousResponseID
	} else if opts.Conversation != nil {
		body["conversation"] = *opts.Conversation
	}
	if opts.Store != nil {
		body["store"] = *opts.Store
	} else {
		body["store"] = true
	}

	// Sampling.
	stripTemp := caps.IsReasoningModel && !caps.SupportsNonReasoningParameters
	stripTopP := caps.IsReasoningModel && !caps.SupportsNonReasoningParameters
	if !stripTemp && opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if !stripTopP && opts.TopP != nil {
		body["top_p"] = *opts.TopP
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if opts.MaxOutputTokens != nil {
		body["max_output_tokens"] = *opts.MaxOutputTokens
	}
	if len(opts.StopSequences) > 0 {
		body["stop"] = append([]string(nil), opts.StopSequences...)
	}

	// Reasoning.
	reasoning := map[string]any{}
	if opts.Reasoning != nil {
		if opts.Reasoning.Effort != nil {
			reasoning["effort"] = *opts.Reasoning.Effort
		}
		if opts.Reasoning.Summary != nil {
			reasoning["summary"] = *opts.Reasoning.Summary
		}
	}
	if v, ok := chatOptions["reasoningEffort"].(string); ok && v != "" {
		reasoning["effort"] = v
	}
	if v, ok := chatOptions["reasoningSummary"].(string); ok && v != "" {
		reasoning["summary"] = v
	}
	if len(reasoning) > 0 {
		body["reasoning"] = reasoning
	}

	// Text format.
	textCfg := map[string]any{}
	if opts.StructuredOutput != nil {
		textCfg["format"] = buildResponsesStructuredFormat(opts.StructuredOutput)
	} else if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json" {
		textCfg["format"] = buildResponsesJSONFormat(opts.ResponseFormat)
	}
	if v, ok := chatOptions["textVerbosity"].(string); ok && v != "" {
		textCfg["verbosity"] = v
	}
	if len(textCfg) > 0 {
		body["text"] = textCfg
	}

	// Service tier.
	if v, ok := chatOptions["serviceTier"].(string); ok && v != "" {
		if v == "flex" && !caps.SupportsFlexProcessing {
			warnings = append(warnings, Warning{Type: "other", Message: "service_tier \"flex\" is not supported on this model"})
		} else if v == "priority" && !caps.SupportsPriorityProcessing {
			warnings = append(warnings, Warning{Type: "other", Message: "service_tier \"priority\" is not supported on this model"})
		} else {
			body["service_tier"] = v
		}
	}
	if v, ok := chatOptions["metadata"]; ok {
		body["metadata"] = v
	}
	if v, ok := chatOptions["user"].(string); ok && v != "" {
		body["user"] = v
	}
	if v, ok := chatOptions["promptCacheKey"].(string); ok && v != "" {
		body["prompt_cache_key"] = v
	}
	if v, ok := chatOptions["promptCacheRetention"].(string); ok && v != "" {
		body["prompt_cache_retention"] = v
	}
	if v, ok := chatOptions["safetyIdentifier"].(string); ok && v != "" {
		body["safety_identifier"] = v
	}
	if v, ok := chatOptions["truncation"].(string); ok && v != "" {
		body["truncation"] = v
	}
	if v, ok := chatOptions["maxToolCalls"]; ok {
		body["max_tool_calls"] = v
	}
	if v, ok := chatOptions["parallelToolCalls"]; ok {
		body["parallel_tool_calls"] = v
	}
	if v, ok := chatOptions["topLogprobs"]; ok {
		body["top_logprobs"] = v
	}
	if v, ok := chatOptions["logprobs"].(bool); ok && v {
		body["logprobs"] = true
	}
	if v, ok := chatOptions["contextManagement"]; ok {
		body["context_management"] = v
	}
	if v, ok := chatOptions["include"]; ok {
		body["include"] = v
	}

	// Build input items from messages.
	input, inputWarnings, err := m.convertResponsesInput(opts.Messages, chatOptions, body)
	if err != nil {
		return nil, warnings, err
	}
	warnings = append(warnings, inputWarnings...)

	// Post-processing: if store=false, drop any reasoning items that lack
	// encrypted_content (they can't be round-tripped without storage).
	if opts.Store != nil && !*opts.Store {
		filtered, dropped := dropUnencryptedReasoning(input)
		if dropped {
			warnings = append(warnings, Warning{
				Type:    "other",
				Message: "Reasoning items without encrypted_content were dropped because store=false.",
			})
		}
		input = filtered
	}

	body["input"] = input

	// Tools.
	toolsIn := make([]Tool, 0, len(opts.Tools))
	for _, t := range opts.Tools {
		toolsIn = append(toolsIn, Tool{
			Type:        t.Type,
			ID:          t.ID,
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
			Strict:      t.Strict,
			Args:        t.Args,
		})
	}
	toolsPayload, toolWarnings, err := m.convertResponsesTools(toolsIn)
	if err != nil {
		return nil, warnings, err
	}
	warnings = append(warnings, toolWarnings...)
	if len(toolsPayload) > 0 {
		body["tools"] = toolsPayload
	}

	// Auto-add SDK include values based on the request context (after tools
	// are known so we can check for web_search and code_interpreter).
	if autoInclude := buildAutoInclude(chatOptions, toolsIn, opts.Store); len(autoInclude) > 0 {
		existing, _ := body["include"].([]string)
		body["include"] = mergeIncludes(existing, autoInclude)
	}
	if opts.ToolChoice != nil {
		choice, err := m.convertResponsesToolChoice(*opts.ToolChoice, toolsIn)
		if err != nil {
			return nil, warnings, err
		}
		if choice != nil {
			body["tool_choice"] = choice
		}
	}
	return body, warnings, nil
}

// mergeResponsesOpenAIOptions merges the openai / openai-compatible /
// openaiCompatible provider option bags into a single map.
func mergeResponsesOpenAIOptions(opts ProviderOptions) (map[string]any, []Warning) {
	return mergeChatOpenAIOptions(opts)
}

func buildResponsesStructuredFormat(s *StructuredOutput) map[string]any {
	name := s.Name
	if name == "" {
		name = "response"
	}
	schema := map[string]any{
		"type":    "json_schema",
		"strict":  true,
		"name":    name,
		"schema":  s.Schema,
	}
	if s.Description != "" {
		schema["description"] = s.Description
	}
	return schema
}

func buildResponsesJSONFormat(r *ResponseFormat) map[string]any {
	if r.Schema == nil {
		return map[string]any{"type": "json_object"}
	}
	name := r.Name
	if name == "" {
		name = "response"
	}
	schema := map[string]any{
		"type":   "json_schema",
		"strict": true,
		"name":   name,
		"schema": r.Schema,
	}
	if r.Description != "" {
		schema["description"] = r.Description
	}
	return schema
}

func (m *openaiResponsesModel) convertResponsesToolChoice(choice ToolChoice, tools []Tool) (any, error) {
	switch choice.Type {
	case "auto", "none", "required":
		return choice.Type, nil
	case "tool":
		// Find the matching tool to determine the wire type.
		for _, t := range tools {
			if t.Name == choice.ToolName {
				if t.Type == "provider" && len(t.ID) > len("openai.") && t.ID[:len("openai.")] == "openai." {
					kind := t.ID[len("openai."):]
					switch kind {
					case "webSearch", "webSearchPreview":
						return "web_search", nil
					case "codeInterpreter":
						return "code_interpreter", nil
					case "mcp":
						return "mcp", nil
					case "fileSearch":
						return "file_search", nil
					case "imageGeneration":
						return "image_generation", nil
					case "shell":
						return "shell", nil
					case "localShell":
						return "local_shell", nil
					case "applyPatch":
						return "apply_patch", nil
					case "toolSearch":
						return "tool_search", nil
					case "custom":
						return "custom", nil
					}
				}
				if t.Type == "function" {
					return map[string]any{"type": "function", "name": t.Name}, nil
				}
			}
		}
		return map[string]any{"type": "function", "name": choice.ToolName}, nil
	}
	return nil, nil
}

// parseResponsesResponse turns an OpenAI Responses API response body into
// a ResponsesGenerateResult.
func (m *openaiResponsesModel) parseResponsesResponse(body, requestBody []byte) (*ResponsesGenerateResult, error) {
	var raw map[string]any
	if err := jsonUnmarshalStrict(body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse response: " + err.Error()}
	}
	result := &ResponsesGenerateResult{
		Request:  RequestMetadata{Body: requestBody},
		Response: ResponseMetadata{Body: body, ModelID: m.modelID},
	}
	if id, ok := raw["id"].(string); ok {
		result.Response.ID = id
	}
	if model, ok := raw["model"].(string); ok && model != "" {
		result.Response.ModelID = model
	}
	now := time.Now()
	result.Response.Timestamp = &now
	if v, ok := raw["created_at"].(float64); ok {
		ts := time.Unix(int64(v), 0)
		result.Response.Timestamp = &ts
	}
	if usage, ok := raw["usage"].(map[string]any); ok {
		result.Usage = parseResponsesUsage(usage)
	}
	if fr, ok := raw["status"].(string); ok {
		hasToolCall := false
		if outputs, ok := raw["output"].([]any); ok {
			for _, o := range outputs {
				if m, ok := o.(map[string]any); ok {
					t, _ := m["type"].(string)
					if t == "function_call" || t == "custom_tool_call" {
						hasToolCall = true
					}
				}
			}
		}
		result.FinishReason = mapResponsesFinishReason(fr, hasToolCall)
	}
	// Spec-mandated response-level provider metadata.
	if result.ProviderMetadata == nil {
		result.ProviderMetadata = ProviderMetadata{}
	}
	openaiPM := map[string]any{}
	if id, ok := raw["id"].(string); ok {
		openaiPM["responseId"] = id
	}
	if st, ok := raw["service_tier"].(string); ok && st != "" {
		openaiPM["serviceTier"] = st
	}
	if len(openaiPM) > 0 {
		result.ProviderMetadata["openai"] = openaiPM
	}
	if outputs, ok := raw["output"].([]any); ok {
		content, pm, warns := m.parseResponsesOutputs(outputs)
		result.Content = append(result.Content, content...)
		result.Warnings = append(result.Warnings, warns...)
		for k, v := range pm {
			result.ProviderMetadata[k] = v
		}
	}
	return result, nil
}

func parseResponsesUsage(usage map[string]any) Usage {
	u := Usage{Raw: mustJSON(usage)}
	if v, ok := usage["input_tokens"].(float64); ok {
		t := int(v)
		u.InputTokens.Total = &t
	}
	if v, ok := usage["output_tokens"].(float64); ok {
		t := int(v)
		u.OutputTokens.Total = &t
	}
	if v, ok := usage["total_tokens"].(float64); ok {
		t := int(v)
		if u.InputTokens.Total == nil {
			u.InputTokens.Total = &t
		}
	}
	if details, ok := usage["output_tokens_details"].(map[string]any); ok {
		if v, ok := details["reasoning_tokens"].(float64); ok {
			t := int(v)
			u.OutputTokens.Reasoning = &t
		}
	}
	if details, ok := usage["input_tokens_details"].(map[string]any); ok {
		if v, ok := details["cached_tokens"].(float64); ok {
			t := int(v)
			u.InputTokens.CacheRead = &t
		}
	}
	return u
}

func mapResponsesFinishReason(reason string, hasToolCall bool) FinishReason {
	switch reason {
	case "completed":
		if hasToolCall {
			return FinishReason{Unified: "tool-calls", Raw: reason}
		}
		return FinishReason{Unified: "stop", Raw: reason}
	case "incomplete":
		return FinishReason{Unified: "length", Raw: reason}
	case "failed":
		return FinishReason{Unified: "error", Raw: reason}
	case "cancelled", "canceled":
		return FinishReason{Unified: "other", Raw: reason}
	case "queued", "in_progress":
		return FinishReason{Unified: "other", Raw: reason}
	}
	if hasToolCall {
		return FinishReason{Unified: "tool-calls", Raw: reason}
	}
	return FinishReason{Unified: "other", Raw: reason}
}

// parseResponsesOutputs converts the output array into Content parts.
func (m *openaiResponsesModel) parseResponsesOutputs(outputs []any) ([]Content, ProviderMetadata, []Warning) {
	var out []Content
	pm := ProviderMetadata{}
	var warns []Warning
	for _, o := range outputs {
		item, ok := o.(map[string]any)
		if !ok {
			continue
		}
		t, _ := item["type"].(string)
		switch t {
		case "message":
			itemID, _ := item["id"].(string)
			phase, _ := item["phase"].(string)
			annotationsAny, _ := item["annotations"].([]any)
			annotations := make([]map[string]any, 0, len(annotationsAny))
			for _, a := range annotationsAny {
				if am, ok := a.(map[string]any); ok {
					annotations = append(annotations, am)
				}
			}
			if content, ok := item["content"].([]any); ok {
				for _, c := range content {
					cp, ok := c.(map[string]any)
					if !ok {
						continue
					}
					ct, _ := cp["type"].(string)
					switch ct {
					case "output_text":
						text, _ := cp["text"].(string)
						tc := TextContent{Text: text}
						if itemID != "" || phase != "" || len(annotations) > 0 {
							pm := ProviderMetadata{"openai": map[string]any{}}
							if itemID != "" {
								pm["openai"].(map[string]any)["itemId"] = itemID
							}
							if phase != "" {
								pm["openai"].(map[string]any)["phase"] = phase
							}
							if len(annotations) > 0 {
								pm["openai"].(map[string]any)["annotations"] = annotations
							}
							tc.ProviderOptions = pm
						}
						out = append(out, tc)
					}
				}
			}
		case "function_call":
			name, _ := item["name"].(string)
			args, _ := item["arguments"].(string)
			callID, _ := item["call_id"].(string)
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   name,
					Input:      json.RawMessage(args),
				},
			})
		case "custom_tool_call":
			name, _ := item["name"].(string)
			input, _ := item["input"].(string)
			callID, _ := item["call_id"].(string)
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   name,
					Input:      json.RawMessage(input),
				},
			})
		case "reasoning":
			var summary strings.Builder
			if sm, ok := item["summary"].([]any); ok {
				for _, s := range sm {
					if sm, ok := s.(map[string]any); ok {
						if text, ok := sm["text"].(string); ok {
							summary.WriteString(text)
						}
					}
				}
			}
			id, _ := item["id"].(string)
			encrypted, _ := item["encrypted_content"].(string)
			out = append(out, ReasoningContent{
				Text:             summary.String(),
				ItemID:           id,
				EncryptedContent: encrypted,
			})
		case "web_search_call":
			callID, _ := item["id"].(string)
			actionBytes, _ := json.Marshal(item)
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "web_search",
					Input:      json.RawMessage("{}"),
				},
				ProviderExecuted: true,
			})
			out = append(out, ToolResultContent{
				ToolResultContent: openaicompatible.ToolResultContent{
					ToolCallID: callID,
					Output:     ToolResultOutput{Type: "json", Value: actionBytes},
				},
			})
		case "file_search_call":
			callID, _ := item["id"].(string)
			queries, _ := item["queries"].([]any)
			results, _ := item["results"].([]any)
			combined := map[string]any{"queries": queries, "results": results}
			bytes, _ := json.Marshal(combined)
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "file_search",
					Input:      json.RawMessage("{}"),
				},
				ProviderExecuted: true,
			})
			out = append(out, ToolResultContent{
				ToolResultContent: openaicompatible.ToolResultContent{
					ToolCallID: callID,
					Output:     ToolResultOutput{Type: "json", Value: bytes},
				},
			})
		case "code_interpreter_call":
			callID, _ := item["id"].(string)
			code, _ := item["code"].(string)
			containerID, _ := item["container_id"].(string)
			inputBytes, _ := json.Marshal(map[string]any{"code": code, "container_id": containerID})
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "code_interpreter",
					Input:      inputBytes,
				},
				ProviderExecuted: true,
			})
			if outputs, ok := item["outputs"].([]any); ok {
				outputBytes, _ := json.Marshal(outputs)
				out = append(out, ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: callID,
						Output:     ToolResultOutput{Type: "json", Value: outputBytes},
					},
				})
			}
		case "image_generation_call":
			callID, _ := item["id"].(string)
			result, _ := item["result"].(string)
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "image_generation",
					Input:      json.RawMessage("{}"),
				},
				ProviderExecuted: true,
			})
			// Per spec: ToolResultContent{Output.Value: <base64 string>}
			out = append(out, ToolResultContent{
				ToolResultContent: openaicompatible.ToolResultContent{
					ToolCallID: callID,
					Output:     ToolResultOutput{Type: "json", Value: result},
				},
			})
		case "mcp_call":
			callID, _ := item["id"].(string)
			name, _ := item["name"].(string)
			serverLabel, _ := item["server_label"].(string)
			args, _ := item["arguments"].(string)
			inputBytes, _ := json.Marshal(map[string]any{
				"arguments":    args,
				"name":         name,
				"server_label": serverLabel,
			})
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "mcp",
					Input:      inputBytes,
				},
				ProviderExecuted: true,
				Dynamic:          true,
			})
			if output, ok := item["output"]; ok {
				outputBytes, _ := json.Marshal(output)
				out = append(out, ToolResultContent{
					ToolResultContent: openaicompatible.ToolResultContent{
						ToolCallID: callID,
						Output:     ToolResultOutput{Type: "json", Value: outputBytes},
					},
				})
			}
		case "mcp_approval_request":
			id, _ := item["id"].(string)
			name, _ := item["name"].(string)
			serverLabel, _ := item["server_label"].(string)
			args, _ := item["arguments"].(string)
			inputBytes, _ := json.Marshal(map[string]any{
				"arguments":    args,
				"name":         name,
				"server_label": serverLabel,
			})
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: id,
					ToolName:   "mcp",
					Input:      inputBytes,
				},
				ProviderExecuted: true,
				Dynamic:          true,
			})
		case "local_shell_call":
			callID, _ := item["call_id"].(string)
			action, _ := json.Marshal(item["action"])
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "local_shell",
					Input:      action,
				},
			})
		case "shell_call":
			id, _ := item["id"].(string)
			callID, _ := item["call_id"].(string)
			action, _ := json.Marshal(item["action"])
			providerExecuted := false
			if env, ok := item["action"].(map[string]any); ok {
				if _, ok := env["container_id"]; ok {
					providerExecuted = true
				}
			}
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "shell",
					Input:      action,
					ProviderMetadata: ProviderMetadata{
						"openai": map[string]any{"itemId": id},
					},
				},
				ProviderExecuted: providerExecuted,
			})
		case "apply_patch_call":
			callID, _ := item["call_id"].(string)
			op, _ := json.Marshal(item["operation"])
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "apply_patch",
					Input:      op,
				},
			})
		case "tool_search_call":
			callID, _ := item["call_id"].(string)
			execution, _ := item["execution"].(string)
			args, _ := item["arguments"].(string)
			out = append(out, ToolCallContent{
				ToolCallContentEmbed: ToolCallContentEmbed{
					ToolCallID: callID,
					ToolName:   "tool_search",
					Input:      json.RawMessage(args),
				},
				ProviderExecuted: execution == "server",
			})
		case "compaction":
			id, _ := item["id"].(string)
			encrypted, _ := item["encrypted_content"].(string)
			out = append(out, CompactionContent{ItemID: id, EncryptedContent: encrypted})
		default:
			warns = append(warns, Warning{Type: "other", Message: "unknown output type: " + t})
		}
	}
	return out, pm, warns
}

// mustJSON marshals v to a json.RawMessage (panics on marshal error - only
// used for trusted map[string]any inputs).
func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}

// convertResponsesInput converts a chat-style []Message into a Responses
// `input` array. It applies the rules in the spec.
func (m *openaiResponsesModel) convertResponsesInput(messages []Message, chatOptions map[string]any, body map[string]any) ([]any, []Warning, error) {
	mode := "system"
	if v, ok := chatOptions["systemMessageMode"].(string); ok && v != "" {
		mode = v
	}
	var out []any
	var warnings []Warning
	for _, msg := range messages {
		switch part := msg.(type) {
		case SystemMessage:
			if mode == "remove" {
				continue
			}
			if mode == "developer" {
				if existing, ok := body["instructions"].(string); ok {
					body["instructions"] = existing + "\n" + part.Content
				} else {
					body["instructions"] = part.Content
				}
				continue
			}
			out = append(out, map[string]any{
				"role":    "system",
				"content": part.Content,
			})
		case UserMessage:
			converted, err := m.convertResponsesUserMessage(part)
			if err != nil {
				return nil, warnings, err
			}
			out = append(out, converted)
		case AssistantMessage:
			converted, warns, err := m.convertResponsesAssistantMessage(part)
			warnings = append(warnings, warns...)
			if err != nil {
				return nil, warnings, err
			}
			out = append(out, converted...)
		case ToolMessage:
			converted, err := m.convertResponsesToolMessage(part)
			if err != nil {
				return nil, warnings, err
			}
			out = append(out, converted...)
		default:
			return nil, warnings, InvalidPromptError{Message: fmt.Sprintf("unsupported message type %T", msg)}
		}
	}
	return out, warnings, nil
}

func (m *openaiResponsesModel) convertResponsesUserMessage(msg UserMessage) (map[string]any, error) {
	if len(msg.Content) == 1 {
		if text, ok := msg.Content[0].(TextContent); ok {
			return map[string]any{
				"role":    "user",
				"content": []map[string]any{{"type": "input_text", "text": text.Text}},
			}, nil
		}
	}
	parts := make([]map[string]any, 0, len(msg.Content))
	for _, content := range msg.Content {
		part, err := m.convertResponsesUserContent(content)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	return map[string]any{"role": "user", "content": parts}, nil
}

func (m *openaiResponsesModel) convertResponsesUserContent(content UserContent) (map[string]any, error) {
	switch part := content.(type) {
	case TextContent:
		return map[string]any{"type": "input_text", "text": part.Text}, nil
	case FileContent:
		return m.convertResponsesUserFileContent(part)
	default:
		return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported user content type %T", content)}
	}
}

func (m *openaiResponsesModel) convertResponsesUserFileContent(part FileContent) (map[string]any, error) {
	mediaType := part.MediaType
	prefixes := m.provider.fileIDPrefixes
	if prefixes == nil {
		prefixes = []string{"file-"}
	}
	if v, ok := part.ProviderOptions["openai"].(map[string]any); ok {
		if p, ok := v["fileIdPrefixes"].([]string); ok {
			prefixes = p
		}
	}
	isImage := strings.HasPrefix(mediaType, "image/")
	// Reference data.
	if s, ok := part.Data.(string); ok && s == "reference" {
		resolved, err := m.resolveFileReference(part)
		if err != nil {
			return nil, err
		}
		if isImage {
			img := map[string]any{"type": "input_image", "file_id": resolved}
			if detail := imageDetailFromOptions(part.ProviderOptions); detail != "" {
				img["detail"] = detail
			}
			return img, nil
		}
		return map[string]any{"type": "input_file", "file_id": resolved}, nil
	}
	// URL data.
	if u, ok := part.Data.(*url.URL); ok {
		if isImage {
			img := map[string]any{"type": "input_image", "image_url": u.String()}
			if detail := imageDetailFromOptions(part.ProviderOptions); detail != "" {
				img["detail"] = detail
			}
			return img, nil
		}
		return map[string]any{"type": "input_file", "file_url": u.String()}, nil
	}
	// Base64 data.
	data, err := base64Data(part.Data)
	if err != nil {
		return nil, err
	}
	if isImage {
		// Check for fileIdPrefixes.
		for _, p := range prefixes {
			if strings.HasPrefix(data, p) {
				img := map[string]any{"type": "input_image", "file_id": data}
				if detail := imageDetailFromOptions(part.ProviderOptions); detail != "" {
					img["detail"] = detail
				}
				return img, nil
			}
		}
		img := map[string]any{"type": "input_image", "image_url": "data:" + mediaType + ";base64," + data}
		if detail := imageDetailFromOptions(part.ProviderOptions); detail != "" {
			img["detail"] = detail
		}
		return img, nil
	}
	// Non-image data.
	for _, p := range prefixes {
		if strings.HasPrefix(data, p) {
			return map[string]any{"type": "input_file", "file_id": data}, nil
		}
	}
	if mediaType == "application/pdf" {
		filename := part.Filename
		if filename == "" {
			filename = "part.pdf"
		}
		return map[string]any{
			"type":      "input_file",
			"filename":  filename,
			"file_data": "data:application/pdf;base64," + data,
		}, nil
	}
	passThrough := m.provider.passThroughUnsupportedFiles
	if v, ok := part.ProviderOptions["openai"].(map[string]any); ok {
		if b, ok := v["passThroughUnsupportedFiles"].(bool); ok {
			passThrough = b
		}
	}
	if !passThrough {
		return nil, UnsupportedFunctionalityError{Functionality: "file part media type " + mediaType}
	}
	filename := part.Filename
	if filename == "" {
		filename = "part"
	}
	return map[string]any{
		"type":      "input_file",
		"filename":  filename,
		"file_data": "data:" + mediaType + ";base64," + data,
	}, nil
}

func (m *openaiResponsesModel) resolveFileReference(part FileContent) (string, error) {
	if part.ProviderOptions == nil {
		return "", InvalidPromptError{Message: "file part has Data: \"reference\" but no providerOptions[\"openai\"].reference"}
	}
	openaiOpts, ok := part.ProviderOptions["openai"].(map[string]any)
	if !ok {
		return "", InvalidPromptError{Message: "file part has Data: \"reference\" but no providerOptions[\"openai\"].reference"}
	}
	ref, ok := openaiOpts["reference"].(string)
	if !ok || ref == "" {
		return "", InvalidPromptError{Message: "file part has Data: \"reference\" but no providerOptions[\"openai\"].reference"}
	}
	return ref, nil
}

func (m *openaiResponsesModel) convertResponsesAssistantMessage(msg AssistantMessage) ([]any, []Warning, error) {
	var warnings []Warning
	var out []any
	for _, content := range msg.Content {
		switch part := content.(type) {
		case TextContent:
			out = append(out, map[string]any{
				"role": "assistant",
				"content": []map[string]any{{
					"type": "output_text",
					"text": part.Text,
				}},
			})
		case ReasoningContent:
			// Non-OpenAI reasoning parts (no ItemID, no EncryptedContent) are
			// dropped with a warning.
			if part.ItemID == "" && part.EncryptedContent == "" {
				warnings = append(warnings, Warning{
					Type:    "other",
					Message: "Non-OpenAI reasoning parts are not supported. Skipping.",
				})
				continue
			}
			// Encrypted reasoning content is preserved.
			item := map[string]any{"type": "reasoning"}
			if part.ItemID != "" {
				item["id"] = part.ItemID
			}
			if part.EncryptedContent != "" {
				item["encrypted_content"] = part.EncryptedContent
			}
			if len(part.Summary) > 0 {
				summary := make([]map[string]any, 0, len(part.Summary))
				for _, s := range part.Summary {
					summary = append(summary, map[string]any{"type": "summary_text", "text": s})
				}
				item["summary"] = summary
			}
			out = append(out, item)
		case openaicompatible.ToolCallContent:
			item := map[string]any{
				"type":      "function_call",
				"call_id":   part.ToolCallID,
				"name":      part.ToolName,
				"arguments": string(part.Input),
			}
			if id, ok := part.ProviderMetadata["itemId"].(string); ok && id != "" {
				item["id"] = id
			}
			out = append(out, item)
		case ToolCallContent:
			item := map[string]any{
				"type":      "function_call",
				"call_id":   part.ToolCallID,
				"name":      part.ToolName,
				"arguments": string(part.Input),
			}
			if id, ok := part.ProviderMetadata["itemId"].(string); ok && id != "" {
				item["id"] = id
			}
			out = append(out, item)
		case ToolApprovalResponse:
			item := map[string]any{
				"type":        "mcp_approval_response",
				"approval_request_id": part.ApprovalID,
				"approve":      part.Approve,
			}
			if part.Reason != "" {
				item["reason"] = part.Reason
			}
			out = append(out, item)
		default:
			return nil, warnings, InvalidPromptError{Message: fmt.Sprintf("unsupported assistant content type %T", content)}
		}
	}
	return out, warnings, nil
}

func (m *openaiResponsesModel) convertResponsesToolMessage(msg ToolMessage) ([]any, error) {
	var out []any
	for _, content := range msg.Content {
		part, ok := content.(ToolResultContent)
		if !ok {
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported tool content type %T", content)}
		}
		if part.Output.Type == "tool-approval-response" {
			// Mapped separately for the Responses API; spec says the
			// mcp_approval_response item is added in the assistant
			// message conversion, not tool result.
			continue
		}
		switch part.ToolName {
		case "local_shell":
			out = append(out, map[string]any{
				"type":     "local_shell_call_output",
				"call_id":  part.ToolCallID,
				"output":   stringifyToolResult(part.Output),
			})
		case "shell":
			out = append(out, map[string]any{
				"type":     "shell_call_output",
				"call_id":  part.ToolCallID,
				"output":   buildShellCallOutputArray(part.Output),
			})
		case "apply_patch":
			out = append(out, map[string]any{
				"type":     "apply_patch_call_output",
				"call_id":  part.ToolCallID,
				"status":   "completed",
				"output":   stringifyToolResult(part.Output),
			})
		case "tool_search":
			out = append(out, map[string]any{
				"type":      "tool_search_output",
				"execution": "client",
				"call_id":   part.ToolCallID,
				"status":    "completed",
				"tools":     part.Output.Value,
			})
		default:
			out = append(out, map[string]any{
				"type":     "function_call_output",
				"call_id":  part.ToolCallID,
				"output":   stringifyToolResult(part.Output),
			})
		}
	}
	return out, nil
}

func stringifyToolResult(output ToolResultOutput) any {
	switch output.Type {
	case "text", "error-text":
		if s, ok := output.Value.(string); ok {
			return s
		}
		return fmt.Sprint(output.Value)
	case "execution-denied":
		if output.Reason != "" {
			return output.Reason
		}
		return "Tool call execution denied."
	case "json", "error-json", "content":
		bytes, _ := json.Marshal(output.Value)
		return string(bytes)
	}
	return fmt.Sprint(output.Value)
}

func buildShellCallOutputArray(output ToolResultOutput) []map[string]any {
	if output.Type != "content" {
		return nil
	}
	if arr, ok := output.Value.([]any); ok {
		out := make([]map[string]any, 0, len(arr))
		for _, item := range arr {
			if m, ok := item.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	}
	return nil
}

func (m *openaiResponsesModel) convertResponsesTools(tools []Tool) ([]map[string]any, []Warning, error) {
	var warnings []Warning
	var out []map[string]any
	for _, t := range tools {
		if t.Type == "function" {
			payload, warns, err := buildResponsesFunctionTool(t)
			warnings = append(warnings, warns...)
			if err != nil {
				return nil, warnings, err
			}
			out = append(out, payload)
			continue
		}
		if t.Type == "provider" {
			payload, warns, err := buildResponsesProviderTool(t)
			warnings = append(warnings, warns...)
			if err != nil {
				return nil, warnings, err
			}
			if payload == nil {
				continue
			}
			out = append(out, payload)
			continue
		}
		return nil, warnings, InvalidPromptError{Message: "unsupported tool type " + t.Type}
	}
	return out, warnings, nil
}

func buildResponsesFunctionTool(t Tool) (map[string]any, []Warning, error) {
	tool := map[string]any{
		"type": "function",
		"name": t.Name,
	}
	if t.Description != "" {
		tool["description"] = t.Description
	}
	if t.InputSchema != nil {
		tool["parameters"] = t.InputSchema
	}
	if t.Strict != nil {
		tool["strict"] = *t.Strict
	}
	if t.ProviderOptions != nil {
		if openaiOpts, ok := t.ProviderOptions["openai"].(map[string]any); ok {
			if v, ok := openaiOpts["deferLoading"].(bool); ok && v {
				tool["defer_loading"] = true
			}
		}
	}
	return tool, nil, nil
}

func buildResponsesProviderTool(t Tool) (map[string]any, []Warning, error) {
	if len(t.ID) < len("openai.")+1 || t.ID[:len("openai.")] != "openai." {
		return nil, nil, InvalidPromptError{Message: "provider tool ID must be openai.<kind>"}
	}
	kind := t.ID[len("openai."):]
	args, err := coerceProviderArgs(t.Args)
	if err != nil {
		return nil, nil, err
	}
	switch kind {
	case "fileSearch":
		tool := map[string]any{"type": "file_search"}
		if v, ok := args["vectorStoreIDs"].([]string); ok {
			tool["vector_store_ids"] = v
		} else if v, ok := args["vectorStoreIDs"].([]any); ok {
			strs := make([]string, 0, len(v))
			for _, x := range v {
				if s, ok := x.(string); ok {
					strs = append(strs, s)
				}
			}
			tool["vector_store_ids"] = strs
		}
		if v, ok := args["maxNumResults"].(int); ok {
			tool["max_num_results"] = v
		} else if v, ok := args["maxNumResults"].(float64); ok {
			tool["max_num_results"] = int(v)
		}
		if v, ok := args["ranking"].(map[string]any); ok {
			tool["ranking_options"] = v
		}
		if v, ok := args["filters"]; ok {
			tool["filters"] = v
		}
		return tool, nil, nil
	case "webSearch":
		tool := map[string]any{"type": "web_search"}
		if v, ok := args["externalWebAccess"].(bool); ok {
			tool["external_web_access"] = v
		}
		if v, ok := args["filters"].(map[string]any); ok {
			tool["filters"] = v
		}
		if v, ok := args["searchContextSize"].(string); ok && v != "" {
			tool["search_context_size"] = v
		}
		if v, ok := args["userLocation"].(map[string]any); ok {
			tool["user_location"] = v
		}
		return tool, nil, nil
	case "webSearchPreview":
		tool := map[string]any{"type": "web_search_preview"}
		if v, ok := args["searchContextSize"].(string); ok && v != "" {
			tool["search_context_size"] = v
		}
		if v, ok := args["userLocation"].(map[string]any); ok {
			tool["user_location"] = v
		}
		return tool, nil, nil
	case "codeInterpreter":
		tool := map[string]any{"type": "code_interpreter"}
		if c, ok := args["container"]; ok {
			tool["container"] = c
		}
		return tool, nil, nil
	case "imageGeneration":
		tool := map[string]any{"type": "image_generation"}
		for _, mapping := range []struct {
			wire, key string
		}{
			{"background", "background"},
			{"input_fidelity", "inputFidelity"},
			{"model", "model"},
			{"moderation", "moderation"},
			{"output_compression", "outputCompression"},
			{"output_format", "outputFormat"},
			{"quality", "quality"},
			{"size", "size"},
		} {
			if v, ok := args[mapping.key].(string); ok && v != "" {
				tool[mapping.wire] = v
			}
		}
		if v, ok := args["inputImageMask"]; ok {
			tool["input_image_mask"] = v
		}
		if v, ok := args["n"]; ok {
			tool["n"] = v
		}
		if v, ok := args["partialImages"]; ok {
			tool["partial_images"] = v
		}
		return tool, nil, nil
	case "localShell":
		return map[string]any{"type": "local_shell"}, nil, nil
	case "shell":
		tool := map[string]any{"type": "shell"}
		if v, ok := args["environment"]; ok {
			tool["environment"] = v
		}
		return tool, nil, nil
	case "applyPatch":
		return map[string]any{"type": "apply_patch"}, nil, nil
	case "mcp":
		tool := map[string]any{"type": "mcp", "server_label": args["serverLabel"], "server_url": args["serverUrl"]}
		for _, key := range []string{"allowedTools", "authorization", "connectorId", "headers", "requireApproval", "serverDescription"} {
			if v, ok := args[key]; ok {
				tool[camelToSnake(key)] = v
			}
		}
		return tool, nil, nil
	case "custom":
		tool := map[string]any{"type": "custom", "name": args["name"]}
		if v, ok := args["description"].(string); ok && v != "" {
			tool["description"] = v
		}
		if v, ok := args["format"]; ok {
			tool["format"] = v
		}
		return tool, nil, nil
	case "toolSearch":
		tool := map[string]any{"type": "tool_search"}
		if v, ok := args["execution"].(string); ok && v != "" {
			tool["execution"] = v
		}
		if v, ok := args["description"]; ok {
			tool["description"] = v
		}
		if v, ok := args["parameters"]; ok {
			tool["parameters"] = v
		}
		return tool, nil, nil
	default:
		return nil, nil, InvalidPromptError{Message: "unsupported provider tool kind " + kind}
	}
}

func camelToSnake(s string) string {
	var b strings.Builder
	for i, r := range s {
		if i > 0 && r >= 'A' && r <= 'Z' {
			b.WriteByte('_')
		}
		if r >= 'A' && r <= 'Z' {
			b.WriteRune(r + 32)
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// ensure encoding/json is referenced
var _ = json.RawMessage(nil)

// coerceProviderArgs normalizes a tool's Args into a map[string]any. It
// accepts either an already-decoded map, a JSON-encoded string, or a
// typed struct (e.g. FileSearchArgs) produced by the openai/tools
// subpackage factories. Typed structs are converted via a JSON
// round-trip, so the resulting keys are the JSON field names (which
// match the openaicompatible convention).
func coerceProviderArgs(args any) (map[string]any, error) {
	if args == nil {
		return map[string]any{}, nil
	}
	switch v := args.(type) {
	case map[string]any:
		return v, nil
	case map[string]string:
		out := make(map[string]any, len(v))
		for k, val := range v {
			out[k] = val
		}
		return out, nil
	case string:
		var m map[string]any
		if err := json.Unmarshal([]byte(v), &m); err != nil {
			return nil, InvalidPromptError{Message: "tool Args string is not valid JSON: " + err.Error()}
		}
		return m, nil
	default:
		raw, err := json.Marshal(args)
		if err != nil {
			return nil, InvalidPromptError{Message: "tool Args not marshalable: " + err.Error()}
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, InvalidPromptError{Message: "tool Args not unmarshalable as object: " + err.Error()}
		}
		return m, nil
	}
}

// buildAutoInclude returns the SDK-auto-added include values based on
// the request context: logprobs (when logprobs requested), web-search
// sources (when web_search tool is present), code-interpreter outputs
// (when code_interpreter tool is present), and reasoning.encrypted_content
// (when store=false for a reasoning model).
func buildAutoInclude(chatOptions map[string]any, tools []Tool, store *bool) []string {
	var out []string
	if v, ok := chatOptions["logprobs"].(bool); ok && v {
		out = append(out, "message.output_text.logprobs")
	}
	hasWebSearch := false
	hasCodeInterpreter := false
	for _, t := range tools {
		if t.ID == "openai.webSearch" || t.ID == "openai.webSearchPreview" {
			hasWebSearch = true
		}
		if t.ID == "openai.codeInterpreter" {
			hasCodeInterpreter = true
		}
	}
	if hasWebSearch {
		out = append(out, "web_search_call.action.sources")
	}
	if hasCodeInterpreter {
		out = append(out, "code_interpreter_call.outputs")
	}
	if store != nil && !*store {
		// Encrypted reasoning content is needed when store=false to round-trip.
		// Reasoning capability is gated on the model; for the spec's
		// primary use case (gpt-5/o-series) we add the include.
		out = append(out, "reasoning.encrypted_content")
	}
	return out
}

// mergeIncludes returns the union of include values, preserving order,
// without duplicates.
func mergeIncludes(a, b []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(a)+len(b))
	for _, s := range a {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	for _, s := range b {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// dropUnencryptedReasoning removes reasoning items that have no
// encrypted_content and reports whether any were dropped.
func dropUnencryptedReasoning(input []any) ([]any, bool) {
	dropped := false
	out := make([]any, 0, len(input))
	for _, item := range input {
		m, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		if t, _ := m["type"].(string); t == "reasoning" {
			if _, has := m["encrypted_content"]; !has {
				dropped = true
				continue
			}
		}
		out = append(out, item)
	}
	return out, dropped
}
