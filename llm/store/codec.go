package store

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// wireState is the JSON representation of a RunState. llm.Content is an
// interface, so each content part is tagged with its type to round-trip through
// JSON without modifying the core llm types.
type wireState struct {
	ID       string            `json:"id"`
	Messages []wireMessage     `json:"messages"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type wireMessage struct {
	Role    string        `json:"role"`
	Content []wireContent `json:"content"`
}

type wireContent struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	// Image fields — present when Type == contentTypeImage. spec §1.2
	MIME      string `json:"mime,omitempty"`
	ImageURL  string `json:"image_url,omitempty"`
	ImageData string `json:"image_data,omitempty"` // base64-encoded bytes
}

const (
	contentTypeText       = "text"
	contentTypeToolUse    = "tool_use"
	contentTypeToolResult = "tool_result"
	contentTypeReasoning  = "reasoning"
	contentTypeImage      = "image"
)

// MarshalState encodes a RunState to JSON, tagging each content part with its
// concrete type so it can be reconstructed by UnmarshalState.
func MarshalState(state *RunState) ([]byte, error) {
	wire := wireState{ID: state.ID, Metadata: state.Metadata}
	wire.Messages = make([]wireMessage, len(state.Messages))
	for i, message := range state.Messages {
		contents := make([]wireContent, 0, len(message.Content))
		for _, content := range message.Content {
			wc, err := toWireContent(content)
			if err != nil {
				return nil, err
			}
			contents = append(contents, wc)
		}
		wire.Messages[i] = wireMessage{Role: message.Role, Content: contents}
	}
	return json.Marshal(wire)
}

// UnmarshalState decodes JSON produced by MarshalState back into a RunState.
func UnmarshalState(data []byte) (*RunState, error) {
	var wire wireState
	if err := json.Unmarshal(data, &wire); err != nil {
		return nil, err
	}
	state := &RunState{ID: wire.ID, Metadata: wire.Metadata}
	state.Messages = make([]llm.Message, len(wire.Messages))
	for i, message := range wire.Messages {
		contents := make([]llm.Content, 0, len(message.Content))
		for _, wc := range message.Content {
			content, err := fromWireContent(wc)
			if err != nil {
				return nil, err
			}
			contents = append(contents, content)
		}
		state.Messages[i] = llm.Message{Role: message.Role, Content: contents}
	}
	return state, nil
}

func toWireContent(content llm.Content) (wireContent, error) {
	switch c := content.(type) {
	case llm.TextContent:
		return wireContent{Type: contentTypeText, Text: c.Text}, nil
	case llm.ToolUseContent:
		return wireContent{Type: contentTypeToolUse, ID: c.ID, Name: c.Name, Input: c.Input}, nil
	case llm.ToolResultContent:
		return wireContent{Type: contentTypeToolResult, ToolUseID: c.ToolUseID, Text: c.Text, IsError: c.IsError}, nil
	case llm.ReasoningContent:
		return wireContent{Type: contentTypeReasoning, Text: c.Text}, nil
	case llm.ImageContent:
		wc := wireContent{Type: contentTypeImage, MIME: c.MIME}
		switch src := c.Source.(type) {
		case llm.ImageURLSource:
			wc.ImageURL = src.URL
		case llm.ImageInlineSource:
			wc.ImageData = base64.StdEncoding.EncodeToString(src.Data)
		default:
			return wireContent{}, fmt.Errorf("store: unknown image source %T", c.Source)
		}
		return wc, nil
	default:
		return wireContent{}, fmt.Errorf("store: unsupported content type %T", content)
	}
}

func fromWireContent(wc wireContent) (llm.Content, error) {
	switch wc.Type {
	case contentTypeText:
		return llm.TextContent{Text: wc.Text}, nil
	case contentTypeToolUse:
		return llm.ToolUseContent{ID: wc.ID, Name: wc.Name, Input: wc.Input}, nil
	case contentTypeToolResult:
		return llm.ToolResultContent{ToolUseID: wc.ToolUseID, Text: wc.Text, IsError: wc.IsError}, nil
	case contentTypeReasoning:
		return llm.ReasoningContent{Text: wc.Text}, nil
	case contentTypeImage:
		img := llm.ImageContent{MIME: wc.MIME}
		if wc.ImageURL != "" {
			img.Source = llm.ImageURLSource{URL: wc.ImageURL}
		} else if wc.ImageData != "" {
			data, err := base64.StdEncoding.DecodeString(wc.ImageData)
			if err != nil {
				return nil, fmt.Errorf("store: decode image data: %w", err)
			}
			img.Source = llm.ImageInlineSource{Data: data}
		} else {
			return nil, fmt.Errorf("store: image content has no source")
		}
		return img, nil
	default:
		return nil, fmt.Errorf("store: unknown content type %q", wc.Type)
	}
}
