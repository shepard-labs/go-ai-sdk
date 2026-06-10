package cohere

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
)

func convertToCohereChatPrompt(messages []Message) (coherePrompt, error) {
	out := coherePrompt{Messages: make([]map[string]any, 0, len(messages))}
	for _, message := range messages {
		switch msg := message.(type) {
		case SystemMessage:
			out.Messages = append(out.Messages, map[string]any{"role": "system", "content": msg.Content})
		case UserMessage:
			m, docs, err := convertCohereUserMessage(msg)
			if err != nil {
				return out, err
			}
			out.Messages = append(out.Messages, m)
			out.Documents = append(out.Documents, docs...)
		case AssistantMessage:
			m, err := convertCohereAssistantMessage(msg)
			if err != nil {
				return out, err
			}
			out.Messages = append(out.Messages, m)
		case ToolMessage:
			ms, err := convertCohereToolMessage(msg)
			if err != nil {
				return out, err
			}
			out.Messages = append(out.Messages, ms...)
		default:
			return out, InvalidPromptError{Message: fmt.Sprintf("unsupported message type %T", message)}
		}
	}
	return out, nil
}

func convertCohereUserMessage(msg UserMessage) (map[string]any, []cohereDocument, error) {
	var docs []cohereDocument
	var text strings.Builder
	parts := []map[string]any{}
	hasImage := false
	for _, c := range msg.Content {
		switch part := c.(type) {
		case TextContent:
			if part.Text == "" {
				continue
			}
			text.WriteString(part.Text)
			parts = append(parts, map[string]any{"type": "text", "text": part.Text})
		case FileContent:
			if isImageMediaType(part.MediaType) {
				hasImage = true
				image, err := convertCohereImagePart(part)
				if err != nil {
					return nil, nil, err
				}
				parts = append(parts, image)
				continue
			}
			doc, err := convertCohereDocument(part)
			if err != nil {
				return nil, nil, err
			}
			docs = append(docs, doc)
		default:
			return nil, nil, InvalidPromptError{Message: fmt.Sprintf("unsupported user content type %T", c)}
		}
	}
	if hasImage {
		return map[string]any{"role": "user", "content": parts}, docs, nil
	}
	return map[string]any{"role": "user", "content": text.String()}, docs, nil
}

func isImageMediaType(mt string) bool { return mt == "image" || strings.HasPrefix(mt, "image/") }
func convertCohereImagePart(part FileContent) (map[string]any, error) {
	mt := part.MediaType
	if mt == "image" || mt == "image/*" {
		mt = "image/jpeg"
	}
	urlValue, err := fileURLOrDataURL(part.Data, mt)
	if err != nil {
		return nil, err
	}
	imageURL := map[string]any{"url": urlValue}
	opts, err := parseImageOptions(part.ProviderOptions)
	if err != nil {
		return nil, err
	}
	if opts.Detail != "" {
		imageURL["detail"] = opts.Detail
	}
	return map[string]any{"type": "image_url", "image_url": imageURL}, nil
}
func fileURLOrDataURL(data any, mediaType string) (string, error) {
	if u, ok := data.(*url.URL); ok {
		return u.String(), nil
	}
	encoded, err := base64Data(data)
	if err != nil {
		return "", err
	}
	return "data:" + mediaType + ";base64," + encoded, nil
}
func base64Data(data any) (string, error) {
	switch v := data.(type) {
	case []byte:
		return base64.StdEncoding.EncodeToString(v), nil
	case string:
		return v, nil
	default:
		return "", InvalidPromptError{Message: fmt.Sprintf("unsupported file data type %T", data)}
	}
}
func convertCohereDocument(part FileContent) (cohereDocument, error) {
	if _, ok := part.Data.(*url.URL); ok {
		return cohereDocument{}, UnsupportedFunctionalityError{Functionality: "File URL data", Message: "URLs should be downloaded by the AI SDK and not reach this point. This indicates a configuration issue."}
	}
	text := ""
	switch v := part.Data.(type) {
	case string:
		text = v
	case []byte:
		if !(strings.HasPrefix(part.MediaType, "text/") || part.MediaType == "application/json") {
			return cohereDocument{}, UnsupportedFunctionalityError{Functionality: "document media type: " + part.MediaType, Message: "Media type '" + part.MediaType + "' is not supported. Supported media types are: text/* and application/json."}
		}
		text = string(v)
	default:
		return cohereDocument{}, InvalidPromptError{Message: fmt.Sprintf("unsupported file data type %T", part.Data)}
	}
	data := map[string]any{"text": text}
	if part.Filename != "" {
		data["title"] = part.Filename
	}
	return cohereDocument{Data: data}, nil
}

func convertCohereAssistantMessage(msg AssistantMessage) (map[string]any, error) {
	var text strings.Builder
	var calls []map[string]any
	for _, c := range msg.Content {
		switch part := c.(type) {
		case TextContent:
			text.WriteString(part.Text)
		case ReasoningContent:
			continue
		case ToolCallContent:
			calls = append(calls, map[string]any{"id": part.ToolCallID, "type": "function", "function": map[string]any{"name": part.ToolName, "arguments": string(part.Input)}})
		default:
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported assistant content type %T", c)}
		}
	}
	out := map[string]any{"role": "assistant"}
	if len(calls) > 0 {
		out["tool_calls"] = calls
	} else {
		out["content"] = text.String()
	}
	return out, nil
}
func convertCohereToolMessage(msg ToolMessage) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(msg.Content))
	for _, c := range msg.Content {
		part, ok := c.(ToolResultContent)
		if !ok {
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported tool content type %T", c)}
		}
		if part.Output.Type == "tool-approval-response" {
			continue
		}
		value, err := convertCohereToolOutput(part.Output)
		if err != nil {
			return nil, err
		}
		out = append(out, map[string]any{"role": "tool", "content": value, "tool_call_id": part.ToolCallID})
	}
	return out, nil
}
func convertCohereToolOutput(output ToolResultOutput) (string, error) {
	switch output.Type {
	case "text", "error-text":
		if s, ok := output.Value.(string); ok {
			return s, nil
		}
		return fmt.Sprint(output.Value), nil
	case "execution-denied":
		if output.Reason != "" {
			return output.Reason, nil
		}
		return "Tool execution denied.", nil
	case "content", "json", "error-json":
		b, err := json.Marshal(output.Value)
		return string(b), err
	default:
		return "", InvalidPromptError{Message: "unsupported tool result output type " + output.Type}
	}
}
