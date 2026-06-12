package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
)

// openaiRealtimeFactory implements ExperimentalRealtimeFactory.
type openaiRealtimeFactory struct {
	provider *openaiProvider
}

func (f *openaiRealtimeFactory) RealtimeModel(modelID string) RealtimeModel {
	return &openaiRealtimeModel{provider: f.provider, modelID: modelID}
}

// GetToken fetches an ephemeral client secret.
func (f *openaiRealtimeFactory) GetToken(opts ClientSecretOptions) (ClientSecretResult, error) {
	return ClientSecretResult{}, nil
}

// openaiRealtimeModel implements RealtimeModel.
type openaiRealtimeModel struct {
	provider *openaiProvider
	modelID  string
}

func (m *openaiRealtimeModel) ModelID() string  { return m.modelID }
func (m *openaiRealtimeModel) Provider() string { return "openai.realtime" }

func (m *openaiRealtimeModel) DoCreateClientSecret(ctx context.Context, opts ClientSecretOptions) (*ClientSecretResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}
	body := map[string]any{}
	if opts.SessionConfig != nil {
		body["session"] = m.BuildSessionConfig(*opts.SessionConfig)
	} else {
		body["session"] = map[string]any{
			"type":  "realtime",
			"model": m.modelID,
		}
	}
	if opts.ExpiresAfterSeconds != nil {
		body["expires_after"] = map[string]any{
			"anchor":  "created_at",
			"seconds": *opts.ExpiresAfterSeconds,
		}
	}
	encoded, err := jsonMarshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.executeJSON(ctx, endpointRealtimeClientSec, encoded, nil)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Value     string `json:"value"`
		ExpiresAt *int64 `json:"expires_at"`
	}
	if err := json.Unmarshal(resp.Body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse: " + err.Error()}
	}
	u, _ := url.Parse(m.provider.baseURL)
	host := ""
	if u != nil {
		host = u.Host
	}
	wsURL := fmt.Sprintf("wss://%s/v1/realtime?model=%s", host, url.QueryEscape(m.modelID))
	return &ClientSecretResult{
		Token:     raw.Value,
		URL:       wsURL,
		ExpiresAt: raw.ExpiresAt,
	}, nil
}

func (m *openaiRealtimeModel) GetWebSocketConfig(opts WebSocketConfigInput) WebSocketConfig {
	host := ""
	if u, err := url.Parse(m.provider.baseURL); err == nil {
		host = u.Host
	}
	if host == "" {
		host = "api.openai.com"
	}
	urlStr := opts.URL
	if urlStr == "" {
		urlStr = fmt.Sprintf("wss://%s/v1/realtime?model=%s", host, url.QueryEscape(m.modelID))
	}
	return WebSocketConfig{
		URL: urlStr,
		Protocols: []string{
			"realtime",
			"openai-insecure-api-key." + opts.Token,
		},
	}
}

func (m *openaiRealtimeModel) ParseServerEvent(raw []byte) RealtimeServerEvent {
	var envelope struct {
		Type        string          `json:"type"`
		SessionID   string          `json:"session_id"`
		ItemID      string          `json:"item_id"`
		PreviousItemID string       `json:"previous_item_id"`
		Item        json.RawMessage `json:"item"`
		Transcript  string          `json:"transcript"`
		ResponseID  string          `json:"response_id"`
		Status      string          `json:"status"`
		CallID      string          `json:"call_id"`
		Delta       string          `json:"delta"`
		Name        string          `json:"name"`
		Arguments   string          `json:"arguments"`
		Text        string          `json:"text"`
		Error       *struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return RealtimeServerEvent{
			Type:    RealtimeEventCustom,
			RawType: "",
			Raw:     raw,
		}
	}
	ev := RealtimeServerEvent{
		SessionID:      envelope.SessionID,
		ItemID:         envelope.ItemID,
		PreviousItemID: envelope.PreviousItemID,
		Item:           envelope.Item,
		Transcript:     envelope.Transcript,
		ResponseID:     envelope.ResponseID,
		Status:         envelope.Status,
		CallID:         envelope.CallID,
		Delta:          envelope.Delta,
		Name:           envelope.Name,
		Arguments:      envelope.Arguments,
		Text:           envelope.Text,
	}
	switch envelope.Type {
	case "session.created":
		ev.Type = RealtimeEventSessionCreated
	case "session.updated":
		ev.Type = RealtimeEventSessionUpdated
	case "input_audio_buffer.speech_started":
		ev.Type = RealtimeEventSpeechStarted
	case "input_audio_buffer.speech_stopped":
		ev.Type = RealtimeEventSpeechStopped
	case "input_audio_buffer.committed":
		ev.Type = RealtimeEventAudioCommitted
	case "conversation.item.added":
		ev.Type = RealtimeEventConversationItemAdded
	case "conversation.item.input_audio_transcription.completed":
		ev.Type = RealtimeEventInputTranscriptionCompleted
	case "response.created":
		ev.Type = RealtimeEventResponseCreated
	case "response.done":
		ev.Type = RealtimeEventResponseDone
	case "response.output_item.added":
		ev.Type = RealtimeEventOutputItemAdded
	case "response.output_item.done":
		ev.Type = RealtimeEventOutputItemDone
	case "response.content_part.added":
		ev.Type = RealtimeEventContentPartAdded
	case "response.content_part.done":
		ev.Type = RealtimeEventContentPartDone
	case "response.output_audio.delta":
		ev.Type = RealtimeEventAudioDelta
	case "response.output_audio.done":
		ev.Type = RealtimeEventAudioDone
	case "response.output_audio_transcript.delta":
		ev.Type = RealtimeEventAudioTranscriptDelta
	case "response.output_audio_transcript.done":
		ev.Type = RealtimeEventAudioTranscriptDone
	case "response.output_text.delta":
		ev.Type = RealtimeEventTextDelta
	case "response.output_text.done":
		ev.Type = RealtimeEventTextDone
	case "response.function_call_arguments.delta":
		ev.Type = RealtimeEventFunctionCallArgumentsDelta
	case "response.function_call_arguments.done":
		ev.Type = RealtimeEventFunctionCallArgumentsDone
	case "error":
		ev.Type = RealtimeEventError
		if envelope.Error != nil {
			ev.Message = envelope.Error.Message
			ev.Code = envelope.Error.Code
		}
	default:
		ev.Type = RealtimeEventCustom
		ev.RawType = envelope.Type
		ev.Raw = raw
	}
	return ev
}

func (m *openaiRealtimeModel) SerializeClientEvent(event RealtimeClientEvent) ([]byte, error) {
	switch event.Type {
	case RealtimeClientSessionUpdate:
		session := map[string]any{}
		if event.Session != nil {
			session = m.BuildSessionConfig(*event.Session)
		} else {
			session["type"] = "realtime"
			session["model"] = m.modelID
		}
		return jsonMarshal(map[string]any{
			"type":    "session.update",
			"session": session,
		})
	case RealtimeClientInputAudioAppend:
		return jsonMarshal(map[string]any{
			"type":  "input_audio_buffer.append",
			"audio": event.Audio,
		})
	case RealtimeClientInputAudioCommit:
		return jsonMarshal(map[string]any{"type": "input_audio_buffer.commit"})
	case RealtimeClientInputAudioClear:
		return jsonMarshal(map[string]any{"type": "input_audio_buffer.clear"})
	case RealtimeClientConversationItemCreate:
		if event.Item == nil {
			return nil, InvalidPromptError{Message: "conversation-item-create requires Item"}
		}
		payload := map[string]any{
			"type": event.Item.Type,
		}
		if event.Item.Role != "" {
			payload["role"] = event.Item.Role
		}
		if len(event.Item.Content) > 0 {
			content := make([]map[string]any, 0, len(event.Item.Content))
			for _, c := range event.Item.Content {
				part := map[string]any{"type": c.Type}
				if c.Text != "" {
					part["text"] = c.Text
				}
				if c.Audio != "" {
					part["audio"] = c.Audio
				}
				content = append(content, part)
			}
			payload["content"] = content
		}
		if event.Item.CallID != "" {
			payload["call_id"] = event.Item.CallID
		}
		if event.Item.Output != "" {
			payload["output"] = event.Item.Output
		}
		return jsonMarshal(map[string]any{
			"type": "conversation.item.create",
			"item": payload,
		})
	case RealtimeClientConversationItemTruncate:
		body := map[string]any{
			"type":    "conversation.item.truncate",
			"item_id": event.ItemID,
		}
		if event.ContentIndex != nil {
			body["content_index"] = *event.ContentIndex
		}
		if event.AudioEndMs != nil {
			body["audio_end_ms"] = *event.AudioEndMs
		}
		return jsonMarshal(body)
	case RealtimeClientResponseCreate:
		body := map[string]any{"type": "response.create"}
		if len(event.OutputModalities) > 0 {
			body["output_modalities"] = event.OutputModalities
		}
		if event.Instructions != "" {
			body["instructions"] = event.Instructions
		}
		if len(event.Metadata) > 0 {
			body["metadata"] = event.Metadata
		}
		return jsonMarshal(body)
	case RealtimeClientResponseCancel:
		return jsonMarshal(map[string]any{"type": "response.cancel"})
	default:
		return nil, InvalidPromptError{Message: "unknown client event type: " + string(event.Type)}
	}
}

func (m *openaiRealtimeModel) BuildSessionConfig(cfg SessionConfig) map[string]any {
	session := map[string]any{
		"type":  "realtime",
		"model": m.modelID,
	}
	if cfg.Instructions != "" {
		session["instructions"] = cfg.Instructions
	}
	if len(cfg.OutputModalities) > 0 {
		session["output_modalities"] = cfg.OutputModalities
	}
	audio := map[string]any{}
	if cfg.InputAudioFormat != nil {
		inputFmt := map[string]any{"type": cfg.InputAudioFormat.Type}
		if cfg.InputAudioFormat.Rate != nil {
			inputFmt["rate"] = *cfg.InputAudioFormat.Rate
		}
		input := map[string]any{"format": inputFmt}
		if cfg.InputAudioTranscription != nil {
			transcription := map[string]any{}
			model := "gpt-realtime-whisper"
			if cfg.InputAudioTranscription.Model != nil && *cfg.InputAudioTranscription.Model != "" {
				model = *cfg.InputAudioTranscription.Model
			}
			transcription["model"] = model
			if cfg.InputAudioTranscription.Language != nil {
				transcription["language"] = *cfg.InputAudioTranscription.Language
			}
			if cfg.InputAudioTranscription.Prompt != nil {
				transcription["prompt"] = *cfg.InputAudioTranscription.Prompt
			}
			input["transcription"] = transcription
		}
		if cfg.TurnDetection != nil {
			switch cfg.TurnDetection.Type {
			case "disabled":
				input["turn_detection"] = nil
			case "server-vad":
				input["turn_detection"] = buildVADConfig(cfg.TurnDetection)
			case "semantic-vad":
				input["turn_detection"] = buildVADConfig(cfg.TurnDetection)
			default:
				// unknown - omit
			}
		}
		audio["input"] = input
	}
	if cfg.OutputAudioFormat != nil {
		outFmt := map[string]any{"type": cfg.OutputAudioFormat.Type}
		if cfg.OutputAudioFormat.Rate != nil {
			outFmt["rate"] = *cfg.OutputAudioFormat.Rate
		}
		output := map[string]any{"format": outFmt}
		if cfg.Voice != "" {
			output["voice"] = cfg.Voice
		}
		if cfg.OutputAudioTranscription != nil {
			transcription := map[string]any{}
			if cfg.OutputAudioTranscription.Model != nil {
				transcription["model"] = *cfg.OutputAudioTranscription.Model
			}
			output["transcription"] = transcription
		}
		audio["output"] = output
	}
	if len(audio) > 0 {
		session["audio"] = audio
	}
	if len(cfg.Tools) > 0 {
		tools := make([]map[string]any, 0, len(cfg.Tools))
		for _, t := range cfg.Tools {
			tools = append(tools, map[string]any{
				"type":        t.Type,
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			})
		}
		session["tools"] = tools
		session["tool_choice"] = "auto"
	}
	// Provider options spread last so they take precedence.
	for k, v := range cfg.ProviderOptions {
		session[k] = v
	}
	return session
}

func buildVADConfig(td *TurnDetection) map[string]any {
	typ := td.Type
	switch typ {
	case "server-vad":
		typ = "server_vad"
	case "semantic-vad":
		typ = "semantic_vad"
	}
	out := map[string]any{"type": typ}
	if td.Threshold != nil {
		out["threshold"] = *td.Threshold
	}
	if td.SilenceDurationMs != nil {
		out["silence_duration_ms"] = *td.SilenceDurationMs
	}
	if td.PrefixPaddingMs != nil {
		out["prefix_padding_ms"] = *td.PrefixPaddingMs
	}
	return out
}
