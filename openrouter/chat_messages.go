package openrouter

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

type apiMessage struct {
	Role             string            `json:"role"`
	Content          any               `json:"content,omitempty"`
	Reasoning        string            `json:"reasoning,omitempty"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details,omitempty"`
	ToolCalls        []apiToolCall     `json:"tool_calls,omitempty"`
	ToolCallID       string            `json:"tool_call_id,omitempty"`
	Name             string            `json:"name,omitempty"`
	CacheControl     *CacheControl     `json:"cache_control,omitempty"`
	Annotations      any               `json:"annotations,omitempty"`
}

type apiPart struct {
	Type         string         `json:"type"`
	Text         string         `json:"text,omitempty"`
	ImageURL     *apiURLPayload `json:"image_url,omitempty"`
	VideoURL     *apiURLPayload `json:"video_url,omitempty"`
	InputAudio   *apiAudio      `json:"input_audio,omitempty"`
	File         *apiFile       `json:"file,omitempty"`
	CacheControl *CacheControl  `json:"cache_control,omitempty"`
}

type apiURLPayload struct {
	URL string `json:"url"`
}
type apiAudio struct{ Data, Format string }
type apiFile struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data"`
}
type apiToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function apiFunctionCall `json:"function"`
}
type apiFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func convertChatMessages(messages []Message) ([]apiMessage, error) {
	seenReasoning := map[string]struct{}{}
	out := make([]apiMessage, 0, len(messages))
	for _, message := range messages {
		switch msg := message.(type) {
		case SystemMessage:
			part := apiPart{Type: "text", Text: msg.Content}
			if cc := cacheControlFromOptions(msg.ProviderOptions); cc != nil {
				part.CacheControl = cc
			}
			out = append(out, apiMessage{Role: "system", Content: []apiPart{part}})
		case UserMessage:
			content, err := convertUserContent(msg.Content, cacheControlFromOptions(msg.ProviderOptions))
			if err != nil {
				return nil, err
			}
			out = append(out, apiMessage{Role: "user", Content: content})
		case AssistantMessage:
			converted, err := convertAssistantMessage(msg, seenReasoning)
			if err != nil {
				return nil, err
			}
			out = append(out, converted)
		case ToolMessage:
			msgs, err := convertToolMessage(msg)
			if err != nil {
				return nil, err
			}
			out = append(out, msgs...)
		default:
			return nil, InvalidPromptError{Message: "unknown message type"}
		}
	}
	return out, nil
}

func convertUserContent(contents []UserContent, msgCC *CacheControl) (any, error) {
	if len(contents) == 1 {
		if t, ok := contents[0].(TextContent); ok && msgCC == nil && cacheControlFromOptions(t.ProviderOptions) == nil {
			return t.Text, nil
		}
	}
	parts := make([]apiPart, 0, len(contents))
	lastText := -1
	for _, content := range contents {
		switch c := content.(type) {
		case TextContent:
			parts = append(parts, apiPart{Type: "text", Text: c.Text, CacheControl: cacheControlFromOptions(c.ProviderOptions)})
			lastText = len(parts) - 1
		case FileContent:
			part, err := fileContentPart(c)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		default:
			return nil, InvalidPromptError{Message: "unsupported user content"}
		}
	}
	if msgCC != nil && lastText >= 0 && parts[lastText].CacheControl == nil {
		parts[lastText].CacheControl = msgCC
	}
	return parts, nil
}

func fileContentPart(c FileContent) (apiPart, error) {
	url, media := fileURL(c)
	if strings.HasPrefix(media, "image/") || strings.HasPrefix(url, "data:image/") || looksImageURL(url) {
		return apiPart{Type: "image_url", ImageURL: &apiURLPayload{URL: url}}, nil
	}
	if strings.HasPrefix(media, "video/") || strings.HasPrefix(url, "data:video/") {
		return apiPart{Type: "video_url", VideoURL: &apiURLPayload{URL: url}}, nil
	}
	if strings.HasPrefix(media, "audio/") || strings.HasPrefix(url, "data:audio/") {
		format := strings.TrimPrefix(media, "audio/")
		if format == "mpeg" {
			format = "mp3"
		}
		return apiPart{Type: "input_audio", InputAudio: &apiAudio{Data: strings.TrimPrefix(url, "data:"+media+";base64,"), Format: format}}, nil
	}
	filename := c.Filename
	if opts := c.ProviderOptions.OpenRouter(); opts != nil {
		if v, ok := opts["filename"].(string); ok {
			filename = v
		}
	}
	return apiPart{Type: "file", File: &apiFile{Filename: filename, FileData: url}}, nil
}

func fileURL(c FileContent) (string, string) {
	media := c.MediaType
	if media == "" {
		media = "application/pdf"
	}
	switch v := c.Data.(type) {
	case string:
		if strings.HasPrefix(v, "http://") || strings.HasPrefix(v, "https://") || strings.HasPrefix(v, "data:") {
			return v, media
		}
		return "data:" + media + ";base64," + v, media
	case []byte:
		return "data:" + media + ";base64," + base64.StdEncoding.EncodeToString(v), media
	default:
		return fmt.Sprint(v), media
	}
}

func convertAssistantMessage(msg AssistantMessage, seen map[string]struct{}) (apiMessage, error) {
	out := apiMessage{Role: "assistant", CacheControl: cacheControlFromOptions(msg.ProviderOptions)}
	var text, reasoning strings.Builder
	for _, c := range msg.Content {
		switch p := c.(type) {
		case TextContent:
			text.WriteString(p.Text)
		case ReasoningContent:
			reasoning.WriteString(p.Text)
			if rd, ok := reasoningDetailsFromMetadata(p.ProviderMetadata); ok {
				out.ReasoningDetails = append(out.ReasoningDetails, filterReasoningDetails(rd, seen)...)
			}
		case ToolCallContent:
			args, _ := json.Marshal(p.Input)
			if len(args) == 0 || string(args) == "null" {
				args = []byte("{}")
			}
			out.ToolCalls = append(out.ToolCalls, apiToolCall{ID: p.ToolCallID, Type: "function", Function: apiFunctionCall{Name: p.ToolName, Arguments: string(args)}})
			if len(out.ReasoningDetails) == 0 {
				if rd, ok := reasoningDetailsFromMetadata(p.ProviderMetadata); ok {
					out.ReasoningDetails = append(out.ReasoningDetails, filterReasoningDetails(rd, seen)...)
				}
			}
		}
	}
	if rd, ok := reasoningDetailsFromOptions(msg.ProviderOptions); ok {
		out.ReasoningDetails = filterReasoningDetails(rd, seen)
	}
	if annotations, ok := msg.ProviderOptions.OpenRouter()["annotations"]; ok {
		out.Annotations = annotations
	}
	if text.Len() > 0 {
		out.Content = text.String()
	}
	if reasoning.Len() > 0 && len(out.ReasoningDetails) > 0 {
		out.Reasoning = reasoning.String()
	}
	if out.Content == nil && len(out.ToolCalls) == 0 {
		out.Content = ""
	}
	return out, nil
}

func convertToolMessage(msg ToolMessage) ([]apiMessage, error) {
	out := make([]apiMessage, 0, len(msg.Content))
	for _, c := range msg.Content {
		tr, ok := c.(ToolResultContent)
		if !ok {
			return nil, InvalidPromptError{Message: "unsupported tool content"}
		}
		var content any
		switch v := tr.Output.(type) {
		case string:
			content = v
		case []UserContent:
			converted, err := convertUserContent(v, nil)
			if err != nil {
				return nil, err
			}
			content = converted
		default:
			b, _ := json.Marshal(v)
			content = string(b)
		}
		out = append(out, apiMessage{Role: "tool", Content: content, ToolCallID: tr.ToolCallID, Name: tr.ToolName, CacheControl: cacheControlFromOptions(tr.ProviderOptions)})
	}
	return out, nil
}

func cacheControlFromOptions(opts ProviderOptions) *CacheControl {
	for _, key := range []string{"openrouter", "anthropic"} {
		m := opts[key]
		if m == nil {
			continue
		}
		for _, ccKey := range []string{"cache_control", "cacheControl"} {
			if v, ok := m[ccKey]; ok {
				switch c := v.(type) {
				case CacheControl:
					return &c
				case *CacheControl:
					return c
				}
				b, _ := json.Marshal(v)
				var cc CacheControl
				if json.Unmarshal(b, &cc) == nil && cc.Type != "" {
					return &cc
				}
			}
		}
	}
	return nil
}

func reasoningDetailsFromOptions(opts ProviderOptions) ([]ReasoningDetail, bool) {
	if opts.OpenRouter() == nil {
		return nil, false
	}
	v, ok := opts.OpenRouter()["reasoning_details"]
	if !ok {
		return nil, false
	}
	return decodeReasoningDetails(v), true
}

func reasoningDetailsFromMetadata(md ProviderMetadata) ([]ReasoningDetail, bool) {
	v, ok := md["openrouter"].(map[string]any)
	if !ok {
		return nil, false
	}
	raw, ok := v["reasoning_details"]
	if !ok {
		return nil, false
	}
	return decodeReasoningDetails(raw), true
}

func decodeReasoningDetails(v any) []ReasoningDetail {
	b, _ := json.Marshal(v)
	var out []ReasoningDetail
	_ = json.Unmarshal(b, &out)
	return out
}

func filterReasoningDetails(in []ReasoningDetail, seen map[string]struct{}) []ReasoningDetail {
	out := make([]ReasoningDetail, 0, len(in))
	for _, d := range in {
		if d.Type == "" {
			continue
		}
		if d.Type == "reasoning.text" && d.Signature == "" && (strings.Contains(d.Format, "anthropic") || strings.Contains(d.Format, "gemini")) {
			continue
		}
		key := d.ID
		if key == "" {
			key = d.Type + ":" + d.Format + ":" + d.Data + ":" + d.Signature + ":" + d.Text + ":" + d.Summary
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, d)
	}
	return out
}

func looksImageURL(s string) bool {
	l := strings.ToLower(s)
	return strings.HasSuffix(l, ".jpg") || strings.HasSuffix(l, ".jpeg") || strings.HasSuffix(l, ".png") || strings.HasSuffix(l, ".gif") || strings.HasSuffix(l, ".webp")
}
