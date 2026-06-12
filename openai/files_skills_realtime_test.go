package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestFilesUploadFile(t *testing.T) {
	respBody := `{"id":"file-abc","object":"file","bytes":42,"created_at":1700000000,"filename":"foo.txt","purpose":"assistants","status":"processed"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Files().UploadFile(context.Background(), FilesUploadOptions{
		Data:      []byte("hello world"),
		Filename:  "foo.txt",
		MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	if result.ProviderReference["openai"] != "file-abc" {
		t.Errorf("ProviderReference: %v", result.ProviderReference)
	}
	if result.Filename != "foo.txt" {
		t.Errorf("Filename: %q", result.Filename)
	}
	if result.MediaType != "text/plain" {
		t.Errorf("MediaType: %q", result.MediaType)
	}
	if len(result.Warnings) != 0 {
		t.Errorf("Warnings: %v", result.Warnings)
	}
	if f.calls != 1 {
		t.Errorf("calls = %d, want 1", f.calls)
	}
}

func TestSkillsUploadSkill(t *testing.T) {
	respBody := `{"id":"skill-1","name":"my-skill","description":"a skill","default_version":"v1","latest_version":"v1","created_at":1700000000}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		Files: []SkillsFile{{
			Path:      "skill.zip",
			Data:      []byte("zip-data"),
			MediaType: "application/zip",
		}},
	})
	if err != nil {
		t.Fatalf("UploadSkill: %v", err)
	}
	if result.ProviderReference["openai"] != "skill-1" {
		t.Errorf("ProviderReference: %v", result.ProviderReference)
	}
	if result.Name != "my-skill" {
		t.Errorf("Name: %q", result.Name)
	}
	if result.Description != "a skill" {
		t.Errorf("Description: %q", result.Description)
	}
	if result.LatestVersion != "v1" {
		t.Errorf("LatestVersion: %q", result.LatestVersion)
	}
}

func TestSkillsDisplayTitleWarning(t *testing.T) {
	respBody := `{"id":"skill-1","name":"x","created_at":1}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	result, err := p.Skills().UploadSkill(context.Background(), SkillsUploadOptions{
		Files:        []SkillsFile{{Path: "s.zip", Data: []byte("x")}},
		DisplayTitle: "ignored",
	})
	if err != nil {
		t.Fatalf("UploadSkill: %v", err)
	}
	if len(result.Warnings) == 0 {
		t.Errorf("expected warning for DisplayTitle")
	}
}

func TestRealtimeGetWebSocketConfig(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	cfg := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").GetWebSocketConfig(WebSocketConfigInput{Token: "ek_xyz"})
	if !strings.HasPrefix(cfg.URL, "wss://") {
		t.Errorf("URL: %q", cfg.URL)
	}
	if !strings.Contains(cfg.URL, "model=gpt-realtime") {
		t.Errorf("URL missing model: %q", cfg.URL)
	}
	if len(cfg.Protocols) != 2 {
		t.Fatalf("Protocols len = %d", len(cfg.Protocols))
	}
	if cfg.Protocols[0] != "realtime" {
		t.Errorf("Protocols[0]: %q", cfg.Protocols[0])
	}
	if cfg.Protocols[1] != "openai-insecure-api-key.ek_xyz" {
		t.Errorf("Protocols[1]: %q", cfg.Protocols[1])
	}
}

func TestRealtimeParseServerEventSessionCreated(t *testing.T) {
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

func TestRealtimeParseServerEventAudioDelta(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	raw := `{"type":"response.output_audio.delta","response_id":"r1","item_id":"i1","delta":"AAAA"}`
	ev := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").ParseServerEvent([]byte(raw))
	if ev.Type != RealtimeEventAudioDelta {
		t.Errorf("Type: %q", ev.Type)
	}
	if ev.ResponseID != "r1" || ev.ItemID != "i1" || ev.Delta != "AAAA" {
		t.Errorf("fields: %+v", ev)
	}
}

func TestRealtimeParseServerEventError(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	raw := `{"type":"error","error":{"message":"oops","code":"bad_request"}}`
	ev := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").ParseServerEvent([]byte(raw))
	if ev.Type != RealtimeEventError {
		t.Errorf("Type: %q", ev.Type)
	}
	if ev.Message != "oops" || ev.Code != "bad_request" {
		t.Errorf("fields: %+v", ev)
	}
}

func TestRealtimeParseServerEventUnknownFallsBackToCustom(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	raw := `{"type":"weird.thing","foo":"bar"}`
	ev := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").ParseServerEvent([]byte(raw))
	if ev.Type != RealtimeEventCustom {
		t.Errorf("Type: %q", ev.Type)
	}
	if ev.RawType != "weird.thing" {
		t.Errorf("RawType: %q", ev.RawType)
	}
}

func TestRealtimeSerializeClientEventSessionUpdate(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	instructions := "be helpful"
	ev := RealtimeClientEvent{Type: RealtimeClientSessionUpdate, Session: &SessionConfig{
		Instructions:      instructions,
		Voice:             "alloy",
		OutputAudioFormat: &AudioFormat{Type: "audio/pcm"},
	}}
	b, err := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").SerializeClientEvent(ev)
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, `"type":"session.update"`) {
		t.Errorf("missing type: %q", s)
	}
	if !strings.Contains(s, `"instructions":"be helpful"`) {
		t.Errorf("missing instructions: %q", s)
	}
	if !strings.Contains(s, `"voice":"alloy"`) {
		t.Errorf("missing voice: %q", s)
	}
	if !strings.Contains(s, `"model":"gpt-realtime"`) {
		t.Errorf("missing model: %q", s)
	}
}

func TestRealtimeSerializeClientEventAudioAppend(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	b, err := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").SerializeClientEvent(RealtimeClientEvent{Type: RealtimeClientInputAudioAppend, Audio: "AAAA"})
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, `"type":"input_audio_buffer.append"`) {
		t.Errorf("missing type: %q", s)
	}
	if !strings.Contains(s, `"audio":"AAAA"`) {
		t.Errorf("missing audio: %q", s)
	}
}

func TestRealtimeBuildSessionConfigIncludesToolsAndAudio(t *testing.T) {
	f := &recordingFetcher{}
	p := newOpenAIForTest(f, "https://example.test/v1")
	cfg := SessionConfig{
		Instructions:      "test",
		OutputModalities:  []string{"text", "audio"},
		InputAudioFormat:  &AudioFormat{Type: "audio/pcm"},
		OutputAudioFormat: &AudioFormat{Type: "audio/pcm"},
		Voice:             "alloy",
		Tools: []RealtimeToolDefinition{
			{Type: "function", Name: "f", Description: "d", Parameters: map[string]any{}},
		},
	}
	sess := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").BuildSessionConfig(cfg)
	if sess["type"] != "realtime" {
		t.Errorf("type: %v", sess["type"])
	}
	if sess["instructions"] != "test" {
		t.Errorf("instructions: %v", sess["instructions"])
	}
	if _, ok := sess["audio"]; !ok {
		t.Errorf("missing audio")
	}
	if _, ok := sess["tools"]; !ok {
		t.Errorf("missing tools")
	}
	if sess["tool_choice"] != "auto" {
		t.Errorf("tool_choice: %v", sess["tool_choice"])
	}
}

func TestRealtimeCreateClientSecret(t *testing.T) {
	respBody := `{"value":"ek_abc123","expires_at":1735689600}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").DoCreateClientSecret(context.Background(), ClientSecretOptions{})
	if err != nil {
		t.Fatalf("DoCreateClientSecret: %v", err)
	}
	if res.Token != "ek_abc123" {
		t.Errorf("Token: %q", res.Token)
	}
	if !strings.HasPrefix(res.URL, "wss://") {
		t.Errorf("URL: %q", res.URL)
	}
	if res.ExpiresAt == nil || *res.ExpiresAt != 1735689600 {
		t.Errorf("ExpiresAt: %v", res.ExpiresAt)
	}
}

// Verifies that ExpiresAfterSeconds emits the expires_after { anchor, seconds }
// object on the wire (per spec), and that SessionConfig populates the
// session object in the request body.
func TestRealtimeCreateClientSecretWithExpiresAndSession(t *testing.T) {
	respBody := `{"value":"ek_xyz","expires_at":1735689600}`
	var capturedBody []byte
	f := &captureFetcher{
		response: response(200, respBody),
		capture: func(r *http.Request) {
			if r.Body != nil {
				b, _ := io.ReadAll(r.Body)
				capturedBody = b
			}
		},
	}
	p := newOpenAIForTest(f, "https://example.test/v1")
	expires := 600
	_, err := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").DoCreateClientSecret(context.Background(), ClientSecretOptions{
		ExpiresAfterSeconds: &expires,
		SessionConfig: &SessionConfig{
			Instructions: "be helpful",
			Voice:        "alloy",
		},
	})
	if err != nil {
		t.Fatalf("DoCreateClientSecret: %v", err)
	}
	if len(capturedBody) == 0 {
		t.Fatal("no body captured")
	}
	var body map[string]any
	if err := json.Unmarshal(capturedBody, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ea, ok := body["expires_after"].(map[string]any)
	if !ok {
		t.Fatalf("expires_after missing: %v", body)
	}
	if ea["anchor"] != "created_at" {
		t.Errorf("anchor: %v", ea["anchor"])
	}
	if v, ok := ea["seconds"].(float64); !ok || int(v) != 600 {
		t.Errorf("seconds: %v", ea["seconds"])
	}
	sess, ok := body["session"].(map[string]any)
	if !ok {
		t.Fatalf("session missing: %v", body)
	}
	if sess["instructions"] != "be helpful" {
		t.Errorf("instructions: %v", sess["instructions"])
	}
	if sess["model"] != "gpt-realtime" {
		t.Errorf("model: %v", sess["model"])
	}
	if sess["type"] != "realtime" {
		t.Errorf("type: %v", sess["type"])
	}
}

// Verifies that when no ExpiresAfterSeconds is provided, the body
// omits the expires_after field.
func TestRealtimeCreateClientSecretOmitsExpiresAfterWhenUnset(t *testing.T) {
	respBody := `{"value":"ek_xyz"}`
	var capturedBody []byte
	f := &captureFetcher{
		response: response(200, respBody),
		capture: func(r *http.Request) {
			if r.Body != nil {
				b, _ := io.ReadAll(r.Body)
				capturedBody = b
			}
		},
	}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.ExperimentalRealtime().RealtimeModel("gpt-realtime").DoCreateClientSecret(context.Background(), ClientSecretOptions{})
	if err != nil {
		t.Fatalf("DoCreateClientSecret: %v", err)
	}
	if bytes.Contains(capturedBody, []byte("expires_after")) {
		t.Errorf("expires_after should be absent: %s", string(capturedBody))
	}
}

func TestImageEditMultipartBuildsFileParts(t *testing.T) {
	respBody := `{"data":[{"b64_json":""}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("gpt-image-1").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "edit me",
		Files:  []ImageFile{{Type: "bytes", Data: []byte("fakepng"), MediaType: "image/png"}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if f.calls != 1 {
		t.Errorf("calls = %d, want 1", f.calls)
	}
}

func TestTranscriptionMultipart(t *testing.T) {
	respBody := `{"text":"hello world","language":"english"}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	res, err := p.Transcription("gpt-4o-transcribe").DoGenerate(context.Background(), TranscriptionOptions{
		Audio:    []byte("fakemp3"),
		Filename: "audio.mp3",
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if res.Text != "hello world" {
		t.Errorf("Text: %q", res.Text)
	}
	if res.Language != "en" {
		t.Errorf("Language: %q", res.Language)
	}
}

func TestImageGenerateRequestWithProviderOptions(t *testing.T) {
	respBody := `{"data":[{"url":"https://example.test/img.png"}]}`
	f := &recordingFetcher{responses: []*http.Response{response(200, respBody)}}
	p := newOpenAIForTest(f, "https://example.test/v1")
	_, err := p.Image("dall-e-3").DoGenerate(context.Background(), ImageGenerateOptions{
		Prompt: "a cat",
		ProviderOptions: ProviderOptions{
			"openai": map[string]any{
				"n":       2,
				"quality": "hd",
				"style":   "vivid",
				"user":    "u1",
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}
