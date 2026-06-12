package openai

import (
	"strings"
	"testing"
)

// TestRealtimeParseAllServerEventTypes verifies that every wire event
// type from the spec's event mapping table is normalized correctly.
func TestRealtimeParseAllServerEventTypes(t *testing.T) {
	cases := []struct {
		raw     string
		want    RealtimeServerEventType
		checks  func(t *testing.T, ev RealtimeServerEvent)
	}{
		{
			raw:  `{"type":"session.updated","session_id":"s1"}`,
			want: RealtimeEventSessionUpdated,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.SessionID != "s1" {
					t.Errorf("SessionID: %q", ev.SessionID)
				}
			},
		},
		{
			raw:  `{"type":"input_audio_buffer.speech_started","item_id":"i1"}`,
			want: RealtimeEventSpeechStarted,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.ItemID != "i1" {
					t.Errorf("ItemID: %q", ev.ItemID)
				}
			},
		},
		{
			raw:  `{"type":"input_audio_buffer.speech_stopped","item_id":"i2"}`,
			want: RealtimeEventSpeechStopped,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.ItemID != "i2" {
					t.Errorf("ItemID: %q", ev.ItemID)
				}
			},
		},
		{
			raw:  `{"type":"input_audio_buffer.committed","item_id":"i3","previous_item_id":"i2"}`,
			want: RealtimeEventAudioCommitted,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.ItemID != "i3" {
					t.Errorf("ItemID: %q", ev.ItemID)
				}
				if ev.PreviousItemID != "i2" {
					t.Errorf("PreviousItemID: %q", ev.PreviousItemID)
				}
			},
		},
		{
			raw:  `{"type":"conversation.item.added","item_id":"i4","item":{"type":"message"}}`,
			want: RealtimeEventConversationItemAdded,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.ItemID != "i4" {
					t.Errorf("ItemID: %q", ev.ItemID)
				}
				if ev.Item == nil {
					t.Errorf("Item should be preserved")
				}
			},
		},
		{
			raw:  `{"type":"response.created","response_id":"r1"}`,
			want: RealtimeEventResponseCreated,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.ResponseID != "r1" {
					t.Errorf("ResponseID: %q", ev.ResponseID)
				}
			},
		},
		{
			raw:  `{"type":"response.done","response_id":"r2","status":"completed"}`,
			want: RealtimeEventResponseDone,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.ResponseID != "r2" {
					t.Errorf("ResponseID: %q", ev.ResponseID)
				}
				if ev.Status != "completed" {
					t.Errorf("Status: %q", ev.Status)
				}
			},
		},
		{
			raw:  `{"type":"response.output_item.added","response_id":"r3","item_id":"o1"}`,
			want: RealtimeEventOutputItemAdded,
		},
		{
			raw:  `{"type":"response.output_item.done","response_id":"r4","item_id":"o2"}`,
			want: RealtimeEventOutputItemDone,
		},
		{
			raw:  `{"type":"response.content_part.added","response_id":"r5","item_id":"o3"}`,
			want: RealtimeEventContentPartAdded,
		},
		{
			raw:  `{"type":"response.content_part.done","response_id":"r6","item_id":"o4"}`,
			want: RealtimeEventContentPartDone,
		},
		{
			raw:  `{"type":"response.output_audio.done","response_id":"r7","item_id":"o5"}`,
			want: RealtimeEventAudioDone,
		},
		{
			raw:  `{"type":"response.output_audio_transcript.delta","response_id":"r8","item_id":"o6","delta":"hi"}`,
			want: RealtimeEventAudioTranscriptDelta,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.Delta != "hi" {
					t.Errorf("Delta: %q", ev.Delta)
				}
			},
		},
		{
			raw:  `{"type":"response.output_audio_transcript.done","response_id":"r9","item_id":"o7","transcript":"done"}`,
			want: RealtimeEventAudioTranscriptDone,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.Transcript != "done" {
					t.Errorf("Transcript: %q", ev.Transcript)
				}
			},
		},
		{
			raw:  `{"type":"response.output_text.delta","response_id":"r10","item_id":"o8","delta":"hello"}`,
			want: RealtimeEventTextDelta,
		},
		{
			raw:  `{"type":"response.output_text.done","response_id":"r11","item_id":"o9","text":"hello"}`,
			want: RealtimeEventTextDone,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.Text != "hello" {
					t.Errorf("Text: %q", ev.Text)
				}
			},
		},
		{
			raw:  `{"type":"response.function_call_arguments.delta","response_id":"r12","item_id":"o10","call_id":"c1","delta":"{\"a\":"}`,
			want: RealtimeEventFunctionCallArgumentsDelta,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.CallID != "c1" {
					t.Errorf("CallID: %q", ev.CallID)
				}
			},
		},
		{
			raw:  `{"type":"response.function_call_arguments.done","response_id":"r13","item_id":"o11","call_id":"c2","name":"f","arguments":"{}"}`,
			want: RealtimeEventFunctionCallArgumentsDone,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.CallID != "c2" {
					t.Errorf("CallID: %q", ev.CallID)
				}
				if ev.Name != "f" {
					t.Errorf("Name: %q", ev.Name)
				}
				if ev.Arguments != "{}" {
					t.Errorf("Arguments: %q", ev.Arguments)
				}
			},
		},
		{
			raw:  `{"type":"session.created","session_id":"sess-1"}`,
			want: RealtimeEventSessionCreated,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.SessionID != "sess-1" {
					t.Errorf("SessionID: %q", ev.SessionID)
				}
			},
		},
		{
			raw:  `{"type":"conversation.item.input_audio_transcription.completed","item_id":"i9","transcript":"user said hi"}`,
			want: RealtimeEventInputTranscriptionCompleted,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.ItemID != "i9" {
					t.Errorf("ItemID: %q", ev.ItemID)
				}
				if ev.Transcript != "user said hi" {
					t.Errorf("Transcript: %q", ev.Transcript)
				}
			},
		},
		{
			raw:  `{"type":"response.output_audio.delta","response_id":"r14","item_id":"o12","delta":"BASE64PCM"}`,
			want: RealtimeEventAudioDelta,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.Delta != "BASE64PCM" {
					t.Errorf("Delta: %q", ev.Delta)
				}
			},
		},
		{
			raw:  `{"type":"error","error":{"message":"bad input","code":"invalid_request"}}`,
			want: RealtimeEventError,
			checks: func(t *testing.T, ev RealtimeServerEvent) {
				if ev.Message != "bad input" {
					t.Errorf("Message: %q", ev.Message)
				}
				if ev.Code != "invalid_request" {
					t.Errorf("Code: %q", ev.Code)
				}
			},
		},
	}
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	rt := p.ExperimentalRealtime().RealtimeModel("gpt-realtime")
	for _, c := range cases {
		ev := rt.ParseServerEvent([]byte(c.raw))
		if ev.Type != c.want {
			t.Errorf("event %q: got %q, want %q", c.raw, ev.Type, c.want)
		}
		if c.checks != nil {
			c.checks(t, ev)
		}
	}
}

// TestRealtimeSerializeAllClientEventTypes verifies that every normalized
// client event is serialized to the correct wire format.
func TestRealtimeSerializeAllClientEventTypes(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	rt := p.ExperimentalRealtime().RealtimeModel("gpt-realtime")

	cases := []struct {
		name     string
		event    RealtimeClientEvent
		contains []string
	}{
		{
			name:     "audio-commit",
			event:    RealtimeClientEvent{Type: RealtimeClientInputAudioCommit},
			contains: []string{`"type":"input_audio_buffer.commit"`},
		},
		{
			name:     "audio-clear",
			event:    RealtimeClientEvent{Type: RealtimeClientInputAudioClear},
			contains: []string{`"type":"input_audio_buffer.clear"`},
		},
		{
			name: "conversation-item-create-text",
			event: RealtimeClientEvent{
				Type: RealtimeClientConversationItemCreate,
				Item: &RealtimeClientItem{
					Type: "message",
					Role: "user",
					Content: []RealtimeClientItemContent{
						{Type: "input_text", Text: "hi"},
					},
				},
			},
			contains: []string{
				`"type":"conversation.item.create"`,
				`"role":"user"`,
				`"text":"hi"`,
			},
		},
		{
			name: "conversation-item-create-audio",
			event: RealtimeClientEvent{
				Type: RealtimeClientConversationItemCreate,
				Item: &RealtimeClientItem{
					Type: "message",
					Role: "user",
					Content: []RealtimeClientItemContent{
						{Type: "input_audio", Audio: "AAAA"},
					},
				},
			},
			contains: []string{
				`"audio":"AAAA"`,
			},
		},
		{
			name: "conversation-item-create-function-output",
			event: RealtimeClientEvent{
				Type: RealtimeClientConversationItemCreate,
				Item: &RealtimeClientItem{
					Type:   "function_call_output",
					CallID: "c1",
					Output: "ok",
				},
			},
			contains: []string{
				`"type":"function_call_output"`,
				`"call_id":"c1"`,
				`"output":"ok"`,
			},
		},
		{
			name: "conversation-item-truncate",
			event: RealtimeClientEvent{
				Type:        RealtimeClientConversationItemTruncate,
				ItemID:      "i1",
				ContentIndex: intP(0),
				AudioEndMs:  intP(500),
			},
			contains: []string{
				`"type":"conversation.item.truncate"`,
				`"item_id":"i1"`,
				`"content_index":0`,
				`"audio_end_ms":500`,
			},
		},
		{
			name: "response-create",
			event: RealtimeClientEvent{
				Type:             RealtimeClientResponseCreate,
				OutputModalities: []string{"text", "audio"},
				Instructions:     "be brief",
				Metadata:         map[string]string{"trace": "1"},
			},
			contains: []string{
				`"type":"response.create"`,
				`"output_modalities":["text","audio"]`,
				`"instructions":"be brief"`,
				`"metadata":{"trace":"1"}`,
			},
		},
		{
			name:  "response-cancel",
			event: RealtimeClientEvent{Type: RealtimeClientResponseCancel},
			contains: []string{
				`"type":"response.cancel"`,
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b, err := rt.SerializeClientEvent(c.event)
			if err != nil {
				t.Fatalf("Serialize: %v", err)
			}
			s := string(b)
			for _, want := range c.contains {
				if !strings.Contains(s, want) {
					t.Errorf("missing %q in %q", want, s)
				}
			}
		})
	}
}

// TestRealtimeParseServerEventSessionCreated is a regression sanity test.
func TestRealtimeParseServerEventSessionCreatedRegression(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	ev := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").ParseServerEvent([]byte(`{"type":"session.created","session_id":"sess-1"}`))
	if ev.Type != RealtimeEventSessionCreated {
		t.Errorf("Type: %q", ev.Type)
	}
	if ev.SessionID != "sess-1" {
		t.Errorf("SessionID: %q", ev.SessionID)
	}
}
