package openai

import (
	"encoding/json"
	"fmt"
	"time"
)

// parseChatResponse converts a buffered OpenAI chat-completion response
// into a GenerateResult. It maps each choice into Content parts, derives
// the finish reason, and parses usage.
func (m *openaiChatLanguageModel) parseChatResponse(body []byte, requestBody []byte, providerOptions ProviderOptions) (GenerateResult, error) {
	var raw map[string]any
	if err := jsonUnmarshalStrict(body, &raw); err != nil {
		return GenerateResult{}, InvalidResponseDataError{Message: "failed to parse response: " + err.Error()}
	}
	result := GenerateResult{
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
	if v, ok := raw["created"].(float64); ok {
		ts := time.Unix(int64(v), 0)
		result.Response.Timestamp = &ts
	} else if v, ok := raw["created"].(int64); ok {
		ts := time.Unix(v, 0)
		result.Response.Timestamp = &ts
	}
	choices, _ := raw["choices"].([]any)
	// First-pass: collect logprobs from the first choice (per spec).
	var firstLogprobs any
	for _, c := range choices {
		choice, ok := c.(map[string]any)
		if !ok {
			continue
		}
		if firstLogprobs == nil {
			if lp, ok := choice["logprobs"]; ok {
				firstLogprobs = lp
			}
		}
		if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
			result.FinishReason = chatFinishReasonFromString(fr)
		}
		if msg, ok := choice["message"].(map[string]any); ok {
			if content, ok := msg["content"].(string); ok && content != "" {
				result.Content = append(result.Content, TextContent{Text: content})
			}
			if reasoning, ok := msg["reasoning_content"].(string); ok && reasoning != "" {
				result.Content = append(result.Content, ReasoningContent{Text: reasoning})
			}
			if annotations, ok := msg["annotations"].([]any); ok {
				for _, a := range annotations {
					am, ok := a.(map[string]any)
					if !ok {
						continue
					}
					typ, _ := am["type"].(string)
					switch typ {
					case "url_citation":
						parts, _ := am["url_citation"].(map[string]any)
						if parts == nil {
							parts = am
						}
						result.Content = append(result.Content, SourceContent{
							SourceType: typ,
							ID:         stringValue(parts["uuid"]),
							URL:        stringValue(parts["url"]),
							Title:      stringValue(parts["title"]),
						})
					}
				}
			}
			if toolCalls, ok := msg["tool_calls"].([]any); ok {
				for _, tc := range toolCalls {
					tcm, ok := tc.(map[string]any)
					if !ok {
						continue
					}
					call, err := convertChatToolCallFromResponse(tcm)
					if err != nil {
						return result, err
					}
					result.Content = append(result.Content, call)
				}
			}
		}
	}
	if usage, ok := raw["usage"].(map[string]any); ok {
		var shape chatUsageShape
		encoded, _ := json.Marshal(usage)
		_ = json.Unmarshal(encoded, &shape)
		// Cache / reasoning details.
		if pd, ok := usage["prompt_tokens_details"].(map[string]any); ok {
			if v, ok := pd["cached_tokens"].(float64); ok {
				shape.PromptTokens -= int(v) // cached_tokens subset of prompt.
			}
		}
		result.Usage = buildChatUsage(encoded, shape)
		// Per-spec providerMetadata["openai"] keys (always populated when we
		// have a usage block, since responseId / modelId / logprobs may also
		// be useful to callers).
		om := map[string]any{}
		if id, ok := raw["id"].(string); ok && id != "" {
			om["responseId"] = id
		}
		if model, ok := raw["model"].(string); ok && model != "" {
			om["modelId"] = model
		}
		if firstLogprobs != nil {
			om["logprobs"] = firstLogprobs
		}
		if cd, ok := usage["completion_tokens_details"].(map[string]any); ok {
			if v, ok := cd["accepted_prediction_tokens"].(float64); ok {
				om["acceptedPredictionTokens"] = int(v)
			}
			if v, ok := cd["rejected_prediction_tokens"].(float64); ok {
				om["rejectedPredictionTokens"] = int(v)
			}
		}
		if len(om) > 0 {
			result.ProviderMetadata = ProviderMetadata{"openai": om}
		}
	}
	return result, nil
}

// convertChatToolCallFromResponse maps a single tool_call from the API
// response into a ToolCallContent.
func convertChatToolCallFromResponse(tc map[string]any) (ToolCallContent, error) {
	id, _ := tc["id"].(string)
	fn, _ := tc["function"].(map[string]any)
	name, _ := fn["name"].(string)
	args, _ := fn["arguments"].(string)
	raw := json.RawMessage(args)
	if len(raw) == 0 {
		raw = json.RawMessage("{}")
	}
	provExec := false
	if v, ok := tc["provider_executed"].(bool); ok {
		provExec = v
	}
	dynamic := false
	if v, ok := tc["dynamic"].(bool); ok {
		dynamic = v
	}
	if name == "" {
		return ToolCallContent{}, InvalidResponseDataError{Message: "tool_call missing function.name"}
	}
	return ToolCallContent{
		ToolCallContentEmbed: ToolCallContentEmbed{
			ToolCallID: id,
			ToolName:   name,
			Input:      raw,
		},
		ProviderExecuted: provExec,
		Dynamic:          dynamic,
	}, nil
}

func convertProviderMetadata(raw map[string]any) (ProviderMetadata, error) {
	pm := ProviderMetadata{}
	if v, ok := raw["openai"].(map[string]any); ok {
		pm["openai"] = v
	}
	if v, ok := raw["anthropic"].(map[string]any); ok {
		pm["anthropic"] = v
	}
	if v, ok := raw["google"].(map[string]any); ok {
		pm["google"] = v
	}
	if len(pm) == 0 {
		return nil, nil
	}
	return pm, nil
}

// stringValue returns the string value of v if it is a string, else "".
func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

// keep fmt import alive
var _ = fmt.Sprintf
