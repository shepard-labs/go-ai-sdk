package openai

// RealtimeServerEventType is the normalized server-event Type.
type RealtimeServerEventType string

const (
	RealtimeEventSessionCreated              RealtimeServerEventType = "session-created"
	RealtimeEventSessionUpdated              RealtimeServerEventType = "session-updated"
	RealtimeEventSpeechStarted               RealtimeServerEventType = "speech-started"
	RealtimeEventSpeechStopped               RealtimeServerEventType = "speech-stopped"
	RealtimeEventAudioCommitted              RealtimeServerEventType = "audio-committed"
	RealtimeEventConversationItemAdded       RealtimeServerEventType = "conversation-item-added"
	RealtimeEventInputTranscriptionCompleted RealtimeServerEventType = "input-transcription-completed"
	RealtimeEventResponseCreated             RealtimeServerEventType = "response-created"
	RealtimeEventResponseDone                RealtimeServerEventType = "response-done"
	RealtimeEventOutputItemAdded             RealtimeServerEventType = "output-item-added"
	RealtimeEventOutputItemDone              RealtimeServerEventType = "output-item-done"
	RealtimeEventContentPartAdded            RealtimeServerEventType = "content-part-added"
	RealtimeEventContentPartDone             RealtimeServerEventType = "content-part-done"
	RealtimeEventAudioDelta                  RealtimeServerEventType = "audio-delta"
	RealtimeEventAudioDone                   RealtimeServerEventType = "audio-done"
	RealtimeEventAudioTranscriptDelta        RealtimeServerEventType = "audio-transcript-delta"
	RealtimeEventAudioTranscriptDone         RealtimeServerEventType = "audio-transcript-done"
	RealtimeEventTextDelta                   RealtimeServerEventType = "text-delta"
	RealtimeEventTextDone                    RealtimeServerEventType = "text-done"
	RealtimeEventFunctionCallArgumentsDelta  RealtimeServerEventType = "function-call-arguments-delta"
	RealtimeEventFunctionCallArgumentsDone   RealtimeServerEventType = "function-call-arguments-done"
	RealtimeEventError                       RealtimeServerEventType = "error"
	RealtimeEventCustom                      RealtimeServerEventType = "custom"
)

// RealtimeServerEvent is a normalized server event from the Realtime API.
type RealtimeServerEvent struct {
	Type           RealtimeServerEventType
	SessionID      string
	ItemID         string
	PreviousItemID string
	Item           any
	Transcript     string
	ResponseID     string
	Status         string
	CallID         string
	Delta          string
	Name           string
	Arguments      string
	Text           string
	Message        string
	Code           string
	RawType        string
	Raw            []byte
}

// RealtimeClientEventType is the normalized client-event Type.
type RealtimeClientEventType string

const (
	RealtimeClientSessionUpdate            RealtimeClientEventType = "session-update"
	RealtimeClientInputAudioAppend         RealtimeClientEventType = "input-audio-append"
	RealtimeClientInputAudioCommit         RealtimeClientEventType = "input-audio-commit"
	RealtimeClientInputAudioClear          RealtimeClientEventType = "input-audio-clear"
	RealtimeClientConversationItemCreate   RealtimeClientEventType = "conversation-item-create"
	RealtimeClientConversationItemTruncate RealtimeClientEventType = "conversation-item-truncate"
	RealtimeClientResponseCreate           RealtimeClientEventType = "response-create"
	RealtimeClientResponseCancel           RealtimeClientEventType = "response-cancel"
)

// RealtimeClientEvent is a normalized client event.
type RealtimeClientEvent struct {
	Type             RealtimeClientEventType
	Session          *SessionConfig
	Audio            string
	Item             *RealtimeClientItem
	ItemID           string
	ContentIndex     *int
	AudioEndMs       *int
	OutputModalities []string
	Instructions     string
	Metadata         map[string]string
}

// RealtimeClientItem is the payload of a conversation.item.create event.
type RealtimeClientItem struct {
	Type    string
	Role    string
	Content []RealtimeClientItemContent
	CallID  string
	Output  string
}

// RealtimeClientItemContent is a content part in a client item.
type RealtimeClientItemContent struct {
	Type  string
	Text  string
	Audio string
}
