package google

// prompt_convert.go mirrors the upstream convert-to-google-messages.ts module.
// It walks the [GenerateOptions.Messages] slice, hoists system messages into a
// top-level systemInstruction, maps roles to "user" / "model", and converts
// each part of a content message into a wire [internal.APIPart].
//
// Special cases:
//
//   - System messages must be at the beginning of the conversation; a system
//     message that follows a non-system message produces an InvalidPromptError.
//   - Gemma models receive the joined system text prepended to the first user
//     message's parts (with a trailing "\n\n"); the top-level
//     systemInstruction is omitted.
//   - Assistant ToolResultContent (client-side) is dropped — tool results are
//     not echoed in the model role.
//   - Server tool responses (those with serverToolCallId / serverToolType in
//     provider options) are appended to the last "model" content rather than
//     pushed as a new "user" content.
//   - Thought signatures on assistant parts are round-tripped into
//     part.thoughtSignature.
//   - For Gemma and pre-Gemini-3 models, tool-result content is collapsed to a
//     single functionResponse with response.content = <text>; image parts are
//     emitted as sibling top-level inlineData parts.
//   - For Gemini 3+, tool-result content of type "content" may carry
//     []ContentPart that is split into response.content (text) and
//     functionResponse.parts[] (inlineData).

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// ConvertPrompt walks opts.Messages and returns the wire contents, optional
// top-level system instruction, and any non-fatal warnings (e.g. dropped
// unknown parts).
func ConvertPrompt(modelID string, opts GenerateOptions) ([]internal.APIContent, *internal.APIContent, []Warning, error) {
	contents, system, warnings, err := convertToGoogleMessages(modelID, opts.Messages)
	if err != nil {
		return nil, nil, warnings, err
	}
	return contents, system, warnings, nil
}

// convertToGoogleMessages is the main prompt-conversion entry point.
func convertToGoogleMessages(modelID string, messages []Message) ([]internal.APIContent, *internal.APIContent, []Warning, error) {
	cap := ModelCapabilitiesForID(modelID)
	supportsFunctionResponseParts := isGemini3(modelID)
	gemma := isGemma(modelID)

	var systemTexts []string
	var systemSeen bool
	// currentUserOrModel is the role of the most recent content we have not
	// finalized; "user" or "model". Empty at the start.
	var currentRole string
	var currentParts []internal.APIPart
	var contents []internal.APIContent
	var warnings []Warning

	flushCurrent := func() {
		if currentRole == "" {
			return
		}
		if len(currentParts) == 0 {
			// Reset state but do not emit an empty content turn.
			currentRole = ""
			return
		}
		contents = append(contents, internal.APIContent{Role: currentRole, Parts: append([]internal.APIPart(nil), currentParts...)})
		currentParts = nil
		currentRole = ""
	}

	appendPart := func(role string, part internal.APIPart) {
		if currentRole != role {
			flushCurrent()
			currentRole = role
		}
		currentParts = append(currentParts, part)
	}

	for _, message := range messages {
		switch msg := message.(type) {
		case SystemMessage:
			if systemSeen || len(contents) > 0 || currentRole != "" {
				return nil, nil, warnings, InvalidPromptError{Message: "system messages are only supported at the beginning of the conversation"}
			}
			systemSeen = true
			if msg.Content != "" {
				systemTexts = append(systemTexts, msg.Content)
			}
		case UserMessage:
			parts, ws, err := convertUserParts(modelID, msg.Content, cap)
			if err != nil {
				return nil, nil, append(warnings, ws...), err
			}
			warnings = append(warnings, ws...)
			if gemma && len(systemTexts) > 0 {
				// Prepend joined system text to the first user content.
				prelude := internal.APIPart{Text: strings.Join(systemTexts, "\n\n") + "\n\n"}
				parts = append([]internal.APIPart{prelude}, parts...)
				systemTexts = nil
			}
			for _, p := range parts {
				appendPart("user", p)
			}
		case AssistantMessage:
			parts, ws, err := convertAssistantParts(modelID, msg.Content)
			if err != nil {
				return nil, nil, append(warnings, ws...), err
			}
			warnings = append(warnings, ws...)
			for _, p := range parts {
				appendPart("model", p)
			}
		case ToolMessage:
			parts, ws, err := convertToolParts(modelID, msg.Content, supportsFunctionResponseParts)
			if err != nil {
				return nil, nil, append(warnings, ws...), err
			}
			warnings = append(warnings, ws...)
			// Decide whether the parts belong in a new "user" content (default)
			// or are appended to the last "model" content (server tool response).
			server := isServerToolResponse(msg)
			if server {
				// Flush any pending content so the model content we want to
				// append to is in `contents` (not in `currentParts`).
				flushCurrent()
				if len(contents) > 0 && contents[len(contents)-1].Role == "model" {
					// Append to the last "model" content.
					last := &contents[len(contents)-1]
					last.Parts = append(last.Parts, parts...)
					continue
				}
				// No prior model content; fall through to a new user content.
			}
			for _, p := range parts {
				appendPart("user", p)
			}
		default:
			return nil, nil, warnings, InvalidPromptError{Message: fmt.Sprintf("unsupported message type %T", message)}
		}
	}
	flushCurrent()

	// Build the systemInstruction content.
	var systemContent *internal.APIContent
	if gemma {
		// Gemma: no systemInstruction; system text was prepended to the first
		// user message (or dropped entirely).
		systemContent = nil
	} else if len(systemTexts) > 0 {
		systemContent = &internal.APIContent{
			Role:  "user",
			Parts: []internal.APIPart{{Text: strings.Join(systemTexts, "\n\n")}},
		}
	}

	return contents, systemContent, warnings, nil
}

// convertUserParts converts a slice of [UserContent] into wire [internal.APIPart]s.
func convertUserParts(modelID string, content []UserContent, cap ModelCapabilities) ([]internal.APIPart, []Warning, error) {
	var out []internal.APIPart
	var warnings []Warning
	for _, c := range content {
		parts, ws, err := convertUserPart(c)
		if err != nil {
			return nil, warnings, err
		}
		warnings = append(warnings, ws...)
		out = append(out, parts...)
	}
	return out, warnings, nil
}

func convertUserPart(content UserContent) ([]internal.APIPart, []Warning, error) {
	switch part := content.(type) {
	case TextContent:
		return []internal.APIPart{textPartWithSignature(part.Text, part.ProviderOptions)}, nil, nil
	case ImageContent:
		return imageParts(part)
	case AudioContent:
		return audioDocumentVideoParts("audio", part.Source, part.ProviderOptions)
	case DocumentContent:
		return audioDocumentVideoParts("document", part.Source, part.ProviderOptions)
	case VideoContent:
		return audioDocumentVideoParts("video", part.Source, part.ProviderOptions)
	case FileContent:
		return fileContentParts(part)
	default:
		return nil, nil, nil
	}
}

// imageParts converts an ImageContent into one or two wire parts (inline data
// for "data" sources, file data for "url" / "reference" sources).
func imageParts(part ImageContent) ([]internal.APIPart, []Warning, error) {
	src := part.Source
	if src.Type == "url" || src.Type == "reference" {
		uri := src.URL
		return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
			FileData: &internal.APIFileData{MimeType: src.MediaType, FileURI: uri},
		}, part.ProviderOptions)}, nil, nil
	}
	data := src.Data
	if data == "" && src.URL != "" {
		// Allow callers to supply base64 in URL.
		data = src.URL
	}
	return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
		InlineData: &internal.APIInlineData{MimeType: src.MediaType, Data: data},
	}, part.ProviderOptions)}, nil, nil
}

// audioDocumentVideoParts handles audio/document/video content where the wire
// shape is the same as ImageContent.
func audioDocumentVideoParts(kind string, src ImageSource, po ProviderOptions) ([]internal.APIPart, []Warning, error) {
	_ = kind
	if src.Type == "url" || src.Type == "reference" {
		return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
			FileData: &internal.APIFileData{MimeType: src.MediaType, FileURI: src.URL},
		}, po)}, nil, nil
	}
	return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
		InlineData: &internal.APIInlineData{MimeType: src.MediaType, Data: src.Data},
	}, po)}, nil, nil
}

// fileContentParts converts a FileContent (data/bytes/url) to wire parts.
func fileContentParts(part FileContent) ([]internal.APIPart, []Warning, error) {
	po := part.ProviderOptions
	if u, ok := part.Data.(*url.URL); ok {
		return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
			FileData: &internal.APIFileData{MimeType: part.MediaType, FileURI: u.String()},
		}, po)}, nil, nil
	}
	if s, ok := part.Data.(string); ok {
		if isLikelyURL(s) && part.MediaType == "" {
			return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
				FileData: &internal.APIFileData{FileURI: s},
			}, po)}, nil, nil
		}
		// Heuristic: base64 if non-empty and not raw URL.
		return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
			InlineData: &internal.APIInlineData{MimeType: part.MediaType, Data: s},
		}, po)}, nil, nil
	}
	if b, ok := part.Data.([]byte); ok {
		return []internal.APIPart{thoughtSignatureWrap(internal.APIPart{
			InlineData: &internal.APIInlineData{MimeType: part.MediaType, Data: base64.StdEncoding.EncodeToString(b)},
		}, po)}, nil, nil
	}
	return nil, nil, InvalidPromptError{Message: fmt.Sprintf("unsupported FileContent.Data type %T", part.Data)}
}

func isLikelyURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") || strings.HasPrefix(s, "gs://")
}

// convertAssistantParts converts assistant content into wire parts. Client-side
// tool results inside assistant content are dropped (return nil).
func convertAssistantParts(modelID string, content []AssistantContent) ([]internal.APIPart, []Warning, error) {
	var out []internal.APIPart
	var warnings []Warning
	for _, c := range content {
		switch part := c.(type) {
		case TextContent:
			out = append(out, textPartWithSignature(part.Text, part.ProviderOptions))
		case ReasoningContent:
			thought := true
			p := internal.APIPart{
				Text:    part.Text,
				Thought: &thought,
			}
			if sig := thoughtSignatureFrom(part.ProviderOptions, "thoughtSignature", part.Signature); sig != "" {
				p.ThoughtSignature = sig
			}
			out = append(out, p)
		case ToolCallContent:
			p, ws, err := assistantToolCallPart(modelID, part)
			if err != nil {
				return nil, warnings, err
			}
			warnings = append(warnings, ws...)
			out = append(out, p)
		case ExecutableCodeContent:
			out = append(out, internal.APIPart{
				ExecutableCode: &internal.APIExecutableCode{Language: part.Language, Code: part.Code},
			})
		case CodeExecutionResultContent:
			out = append(out, internal.APIPart{
				CodeExecutionResult: &internal.APICodeExecutionResult{Outcome: part.Outcome, Output: part.Output},
			})
		default:
			// ToolResultContent only implements ToolContent; it cannot appear
			// here. Upstream also drops assistant-side tool results.
			warnings = append(warnings, Warning{Type: "other", Feature: fmt.Sprintf("%T", c), Message: "unrecognized assistant content; dropped"})
		}
	}
	return out, warnings, nil
}

// assistantToolCallPart converts a single ToolCallContent into a wire part
// (functionCall for client-side, toolCall for server-side).
func assistantToolCallPart(modelID string, part ToolCallContent) (internal.APIPart, []Warning, error) {
	var warnings []Warning
	// Server tool call: Tool has Type=="provider" with serverToolType metadata
	// OR the input ToolCallContent has a serverToolCallId.
	serverType, isServer := serverToolTypeFromMetadata(part.ProviderMetadata)
	if isServer {
		toolCall := &internal.APIServerToolCall{
			ToolType: serverType,
			ID:       part.ToolCallID,
		}
		if len(part.Input) > 0 {
			toolCall.Args = part.Input
		}
		wire := internal.APIPart{ToolCall: toolCall}
		if sig := thoughtSignatureFromMetadata(part.ProviderMetadata); sig != "" {
			wire.ThoughtSignature = sig
		}
		return wire, warnings, nil
	}
	fc := &internal.APIFunctionCall{Name: part.ToolName}
	if part.ToolCallID != "" {
		fc.ID = part.ToolCallID
	}
	if len(part.Input) > 0 {
		fc.Args = part.Input
	}
	wire := internal.APIPart{FunctionCall: fc}
	if sig := thoughtSignatureFromMetadata(part.ProviderMetadata); sig != "" {
		wire.ThoughtSignature = sig
	}
	// Skip-thought-signature sentinel injection for Gemini 3+ when the
	// tool-call part has no signature at all.
	if isGemini3(modelID) && wire.ThoughtSignature == "" {
		wire.ThoughtSignature = SkipThoughtSignatureValidator
		warnings = append(warnings, Warning{
			Type:    "other",
			Feature: "thoughtSignature",
			Message: fmt.Sprintf("injected SkipThoughtSignatureValidator sentinel for tool call %q on Gemini 3+; see https://ai.google.dev/gemini-api/docs/thought-signatures", part.ToolName),
		})
	}
	return wire, warnings, nil
}

// isServerToolResponse reports whether the tool message's provider options
// indicate a server-side (provider-executed) tool result.
func isServerToolResponse(msg ToolMessage) bool {
	if hasServerToolMarker(msg.ProviderOptions) {
		return true
	}
	for _, c := range msg.Content {
		if tr, ok := c.(ToolResultContent); ok {
			if hasServerToolMarker(tr.ProviderOptions) {
				return true
			}
		}
	}
	return false
}

func hasServerToolMarker(po ProviderOptions) bool {
	if po == nil {
		return false
	}
	for _, ns := range []string{"google", "googleVertex", "vertex"} {
		raw, ok := po[ns]
		if !ok {
			continue
		}
		if _, ok := raw["serverToolCallId"]; ok {
			return true
		}
		if _, ok := raw["serverToolType"]; ok {
			return true
		}
	}
	return false
}

// convertToolParts converts tool-result contents into wire parts under a
// "user" content. Server-side results become a toolResponse part; client-side
// results become functionResponse parts (with optional parts[] when the model
// supports Gemini-3 multimodal responses and the output is "content" type).
func convertToolParts(modelID string, content []ToolContent, supportsFunctionResponseParts bool) ([]internal.APIPart, []Warning, error) {
	var out []internal.APIPart
	var warnings []Warning
	for _, c := range content {
		tr, ok := c.(ToolResultContent)
		if !ok {
			warnings = append(warnings, Warning{Type: "other", Feature: fmt.Sprintf("%T", c), Message: "unrecognized tool content; dropped"})
			continue
		}
		parts, ws, err := toolResultParts(modelID, tr, supportsFunctionResponseParts)
		if err != nil {
			return nil, append(warnings, ws...), err
		}
		warnings = append(warnings, ws...)
		out = append(out, parts...)
	}
	return out, warnings, nil
}

func toolResultParts(modelID string, tr ToolResultContent, supportsFunctionResponseParts bool) ([]internal.APIPart, []Warning, error) {
	// Server tool response? Check both ProviderOptions (caller-set) and
	// ProviderMetadata (read-back from a prior response).
	if serverType, isServer := serverToolTypeFromMetadata(tr.ProviderOptions); isServer {
		wire := internal.APIPart{
			ToolResponse: &internal.APIServerToolResponse{
				ToolType: serverType,
				ID:       tr.ToolCallID,
			},
		}
		if raw, ok := marshalToolResultValue(tr.Output); ok {
			wire.ToolResponse.Response = raw
		}
		if sig := thoughtSignatureFrom(tr.ProviderOptions, "thoughtSignature", ""); sig != "" {
			wire.ThoughtSignature = sig
		}
		return []internal.APIPart{wire}, nil, nil
	}

	// Client-side function response.
	if supportsFunctionResponseParts && tr.Output.Type == "content" {
		return gemini3FunctionResponseWithContent(tr)
	}
	return legacyFunctionResponse(tr)
}

// gemini3FunctionResponseWithContent splits a "content" output into
// response.content (joined text) and functionResponse.parts[] (inlineData for
// data-URLs, JSON-stringified text for url/file parts).
func gemini3FunctionResponseWithContent(tr ToolResultContent) ([]internal.APIPart, []Warning, error) {
	partsAny, ok := tr.Output.Value.([]ContentPart)
	if !ok {
		// Fall back to legacy path.
		return legacyFunctionResponse(tr)
	}
	var textChunks []string
	var inlineParts []internal.APIPart
	for _, p := range partsAny {
		if p.Text != "" {
			textChunks = append(textChunks, p.Text)
		}
		if p.InlineData != nil {
			inlineParts = append(inlineParts, internal.APIPart{
				InlineData: &internal.APIInlineData{MimeType: p.InlineData.MimeType, Data: p.InlineData.Data},
			})
		}
		if p.FileData != nil {
			// Non-data URLs are JSON-stringified as text per upstream.
			b, _ := json.Marshal(map[string]any{"fileData": map[string]any{"fileUri": p.FileData.FileURI, "mimeType": p.FileData.MimeType}})
			inlineParts = append(inlineParts, internal.APIPart{Text: string(b)})
		}
	}
	resp := map[string]any{"content": strings.Join(textChunks, "\n")}
	raw, _ := json.Marshal(resp)
	fr := &internal.APIFunctionResponse{
		Name:     tr.ToolCallID,
		Response: raw,
	}
	if tr.ToolCallID != "" {
		// Tool call id is used as the response name when no separate name is
		// supplied; the API allows either but the response name is the more
		// correct field.
	}
	if len(inlineParts) > 0 {
		fr.Parts = inlineParts
	}
	return []internal.APIPart{{FunctionResponse: fr}}, nil, nil
}

// legacyFunctionResponse collapses tool-result content to a single
// functionResponse with response.content = <text>; image data parts become
// sibling top-level inlineData parts. This is the pre-Gemini-3 wire shape.
func legacyFunctionResponse(tr ToolResultContent) ([]internal.APIPart, []Warning, error) {
	// First: assemble the textual "content" summary and any image data siblings.
	var textSummary string
	switch tr.Output.Type {
	case "text", "execution-denied":
		if s, ok := tr.Output.Value.(string); ok {
			textSummary = s
		} else if tr.Output.Reason != "" {
			textSummary = tr.Output.Reason
		} else {
			textSummary = fmt.Sprint(tr.Output.Value)
		}
	case "json":
		if tr.Output.Value != nil {
			b, err := json.Marshal(tr.Output.Value)
			if err != nil {
				return nil, nil, err
			}
			textSummary = string(b)
		}
	default:
		textSummary = ""
	}

	var out []internal.APIPart
	resp := map[string]any{"content": textSummary}
	raw, _ := json.Marshal(resp)
	out = append(out, internal.APIPart{
		FunctionResponse: &internal.APIFunctionResponse{
			Name:     tr.ToolCallID,
			Response: raw,
		},
	})

	return out, nil, nil
}

func marshalToolResultValue(out ToolResultOutput) (json.RawMessage, bool) {
	switch out.Type {
	case "text", "execution-denied":
		if s, ok := out.Value.(string); ok {
			b, _ := json.Marshal(map[string]any{"content": s})
			return b, true
		}
		if out.Reason != "" {
			b, _ := json.Marshal(map[string]any{"content": out.Reason})
			return b, true
		}
		b, _ := json.Marshal(map[string]any{"content": fmt.Sprint(out.Value)})
		return b, true
	case "json":
		b, err := json.Marshal(out.Value)
		if err != nil {
			return nil, false
		}
		return b, true
	}
	return nil, false
}

// textPartWithSignature returns a TextPart possibly carrying a thoughtSignature.
func textPartWithSignature(text string, po ProviderOptions) internal.APIPart {
	p := internal.APIPart{Text: text}
	if sig := thoughtSignatureFrom(po, "thoughtSignature", ""); sig != "" {
		p.ThoughtSignature = sig
	}
	return p
}

// thoughtSignatureWrap attaches a thoughtSignature to a part built from
// per-part provider options.
func thoughtSignatureWrap(part internal.APIPart, po ProviderOptions) internal.APIPart {
	if sig := thoughtSignatureFrom(po, "thoughtSignature", ""); sig != "" {
		part.ThoughtSignature = sig
	}
	return part
}

// thoughtSignatureFrom reads a thoughtSignature from a provider options map.
// It checks the "google" key first, then falls back to "googleVertex" and
// "vertex" for cross-namespace callers. When the per-field value is the
// literal string from a struct (e.g. ReasoningContent.Signature), that
// override is preferred.
func thoughtSignatureFrom(po ProviderOptions, key, fallback string) string {
	if fallback != "" {
		return fallback
	}
	if po == nil {
		return ""
	}
	for _, ns := range []string{"google", "googleVertex", "vertex"} {
		raw, ok := po[ns]
		if !ok {
			continue
		}
		if v, ok := raw[key]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}
	return ""
}

// thoughtSignatureFromMetadata reads a thoughtSignature from a ProviderMetadata
// map. ProviderMetadata is the SDK's read-back metadata type (map[string]any).
func thoughtSignatureFromMetadata(pm ProviderMetadata) string {
	if pm == nil {
		return ""
	}
	for _, ns := range []string{"google", "googleVertex", "vertex"} {
		raw, ok := pm[ns].(map[string]any)
		if !ok {
			continue
		}
		if s, ok := raw["thoughtSignature"].(string); ok {
			return s
		}
	}
	return ""
}

// serverToolTypeFromMetadata returns the serverToolType and a boolean
// indicating presence. Accepts ProviderOptions (caller input) or ProviderMetadata
// (read-back values).
func serverToolTypeFromMetadata(m any) (string, bool) {
	po, _ := m.(ProviderOptions)
	if po != nil {
		for _, ns := range []string{"google", "googleVertex", "vertex"} {
			raw, ok := po[ns]
			if !ok {
				continue
			}
			if v, ok := raw["serverToolType"].(string); ok && v != "" {
				return v, true
			}
		}
	}
	pm, _ := m.(ProviderMetadata)
	if pm != nil {
		for _, ns := range []string{"google", "googleVertex", "vertex"} {
			raw, ok := pm[ns].(map[string]any)
			if !ok {
				continue
			}
			if v, ok := raw["serverToolType"].(string); ok && v != "" {
				return v, true
			}
		}
	}
	return "", false
}
