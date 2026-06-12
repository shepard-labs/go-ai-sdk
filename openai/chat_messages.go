package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// convertChatMessages converts an AI-SDK message list to OpenAI chat
// completion messages. It applies OpenAI-specific behavior:
//   - system messages honor the SystemMessageMode (system / developer / remove).
//   - user file parts support image URL, image data, image reference, audio
//     wav/mp3 data, PDF data, and explicit error cases for unsupported parts.
//   - assistant messages emit text and tool calls; nil content when there are
//     tool calls and empty text.
//   - tool messages emit a single {role: "tool", tool_call_id, content} per
//     result (skipping tool-approval-response parts).
func (m *openaiChatLanguageModel) convertChatMessages(messages []Message, chatOptions map[string]any, body map[string]any) ([]map[string]any, error) {
	systemMode := "system"
	if v, ok := body["__systemMessageMode"]; ok {
		if s, ok := v.(string); ok && s != "" {
			systemMode = s
		}
	}
	converted := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		switch msg := message.(type) {
		case SystemMessage:
			if systemMode == "remove" {
				// Skip silently; no warning here (caller can check the
				// count of converted messages).
				continue
			}
			role := "system"
			if systemMode == "developer" {
				role = "developer"
			}
			out := map[string]any{
				"role":    role,
				"content": msg.Content,
			}
			converted = append(converted, out)
		case UserMessage:
			out, err := m.convertUserMessage(msg)
			if err != nil {
				return nil, err
			}
			converted = append(converted, out)
		case AssistantMessage:
			out, err := m.convertAssistantMessage(msg)
			if err != nil {
				return nil, err
			}
			converted = append(converted, out)
		case ToolMessage:
			outs, err := convertToolMessage(msg)
			if err != nil {
				return nil, err
			}
			converted = append(converted, outs...)
		default:
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported message type %T", message)}
		}
	}
	return converted, nil
}

func (m *openaiChatLanguageModel) convertUserMessage(msg UserMessage) (map[string]any, error) {
	if len(msg.Content) == 1 {
		if text, ok := msg.Content[0].(TextContent); ok {
			return map[string]any{
				"role":    "user",
				"content": text.Text,
			}, nil
		}
	}
	out := map[string]any{"role": "user"}
	parts := make([]map[string]any, 0, len(msg.Content))
	for _, content := range msg.Content {
		part, err := m.convertUserContent(content)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	out["content"] = parts
	return out, nil
}

func (m *openaiChatLanguageModel) convertUserContent(content UserContent) (map[string]any, error) {
	switch part := content.(type) {
	case TextContent:
		return map[string]any{
			"type": "text",
			"text": part.Text,
		}, nil
	case FileContent:
		return m.convertUserFileContent(part)
	default:
		return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported user content type %T", content)}
	}
}

func (m *openaiChatLanguageModel) convertUserFileContent(part FileContent) (map[string]any, error) {
	mediaType := part.MediaType
	// Reference data.
	if s, ok := part.Data.(string); ok && s == "reference" {
		resolved, err := m.resolveFileReference(part)
		if err != nil {
			return nil, err
		}
		if strings.HasPrefix(mediaType, "image/") {
			return map[string]any{
				"type": "file",
				"file": map[string]any{"file_id": resolved},
			}, nil
		}
		return map[string]any{
			"type": "file",
			"file": map[string]any{"file_id": resolved},
		}, nil
	}
	switch {
	case strings.HasPrefix(mediaType, "image/"):
		urlValue, err := fileDataURL(part.Data, mediaType)
		if err != nil {
			return nil, err
		}
		imageURL := map[string]any{"url": urlValue}
		if detail := imageDetailFromOptions(part.ProviderOptions); detail != "" {
			imageURL["detail"] = detail
		}
		return map[string]any{
			"type":      "image_url",
			"image_url": imageURL,
		}, nil
	case strings.HasPrefix(mediaType, "audio/"):
		if isURLData(part.Data) {
			return nil, UnsupportedFunctionalityError{Functionality: "audio file parts with URLs"}
		}
		format := ""
		switch mediaType {
		case "audio/wav", "audio/wave", "audio/x-wav":
			format = "wav"
		case "audio/mp3", "audio/mpeg":
			format = "mp3"
		default:
			return nil, UnsupportedFunctionalityError{Functionality: "audio content parts with media type " + mediaType}
		}
		data, err := base64Data(part.Data)
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"type":        "input_audio",
			"input_audio": map[string]any{"data": data, "format": format},
		}, nil
	case mediaType == "application/pdf":
		if isURLData(part.Data) {
			return nil, UnsupportedFunctionalityError{Functionality: "PDF file parts with URLs"}
		}
		data, err := base64Data(part.Data)
		if err != nil {
			return nil, err
		}
		filename := part.Filename
		if filename == "" {
			filename = "part.pdf"
		}
		return map[string]any{
			"type": "file",
			"file": map[string]any{"filename": filename, "file_data": "data:application/pdf;base64," + data},
		}, nil
	case strings.HasPrefix(mediaType, "text/"):
		// OpenAI chat does not support text file parts.
		return nil, UnsupportedFunctionalityError{Functionality: "text file parts"}
	default:
		return nil, UnsupportedFunctionalityError{Functionality: "file part media type " + mediaType}
	}
}

// resolveFileReference resolves a `Data: "reference"` file part to a file
// id. It honors the openai.reference provider option.
func (m *openaiChatLanguageModel) resolveFileReference(part FileContent) (string, error) {
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

// imageDetailFromOptions extracts the openai.imageDetail value.
func imageDetailFromOptions(opts ProviderMetadata) string {
	if opts == nil {
		return ""
	}
	openaiOpts, ok := opts["openai"].(map[string]any)
	if !ok {
		return ""
	}
	if detail, ok := openaiOpts["imageDetail"].(string); ok {
		return detail
	}
	return ""
}

func (m *openaiChatLanguageModel) convertAssistantMessage(msg AssistantMessage) (map[string]any, error) {
	out := map[string]any{"role": "assistant"}
	var text strings.Builder
	var toolCalls []map[string]any
	for _, content := range msg.Content {
		// Use IsAssistantContent marker via type assertion; assistant parts
		// are typed via the openaicompatible union.
		switch part := content.(type) {
		case TextContent:
			text.WriteString(part.Text)
		case ReasoningContent:
			// Chat completions do not carry reasoning on input; the
			// reasoning_content sibling key is set only on output.
		case openaicompatible.ToolCallContent:
			tc := map[string]any{
				"id":   part.ToolCallID,
				"type": "function",
				"function": map[string]any{
					"name":      part.ToolName,
					"arguments": string(part.Input),
				},
			}
			toolCalls = append(toolCalls, tc)
		default:
			// Some callers may use the openai.ToolCallContent wrapper.
			if tc, ok := content.(ToolCallContent); ok {
				tcMap := map[string]any{
					"id":   tc.ToolCallID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.ToolName,
						"arguments": string(tc.Input),
					},
				}
				toolCalls = append(toolCalls, tcMap)
				continue
			}
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported assistant content type %T", content)}
		}
	}
	if len(toolCalls) > 0 {
		if text.Len() == 0 {
			out["content"] = nil
		} else {
			out["content"] = text.String()
		}
		out["tool_calls"] = toolCalls
	} else {
		out["content"] = text.String()
	}
	return out, nil
}

func convertToolMessage(msg ToolMessage) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(msg.Content))
	for _, content := range msg.Content {
		part, ok := content.(ToolResultContent)
		if !ok {
			return nil, InvalidPromptError{Message: fmt.Sprintf("unsupported tool content type %T", content)}
		}
		if part.Output.Type == "tool-approval-response" {
			// Skip tool-approval-response parts (no chat-completion
			// equivalent).
			continue
		}
		value, err := convertToolResultOutput(part.Output)
		if err != nil {
			return nil, err
		}
		out = append(out, map[string]any{
			"role":         "tool",
			"tool_call_id": part.ToolCallID,
			"content":      value,
		})
	}
	return out, nil
}

func convertToolResultOutput(output ToolResultOutput) (string, error) {
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
		return "Tool call execution denied.", nil
	case "content", "json", "error-json":
		bytes, err := json.Marshal(output.Value)
		return string(bytes), err
	default:
		return "", InvalidPromptError{Message: "unsupported tool result output type " + output.Type}
	}
}

func isURLData(data any) bool {
	_, ok := data.(*url.URL)
	return ok
}

func fileDataURL(data any, mediaType string) (string, error) {
	if u, ok := data.(*url.URL); ok {
		return u.String(), nil
	}
	if mediaType == "image/*" {
		mediaType = "image/jpeg"
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
