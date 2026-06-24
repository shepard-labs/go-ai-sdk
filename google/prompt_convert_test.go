package google

import (
	"strings"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

func TestConvertPrompt_SystemHoisted(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			SystemMessage{Content: "be concise"},
			UserMessage{Content: []UserContent{TextContent{Text: "hello"}}},
		},
	}
	contents, system, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if system == nil {
		t.Fatal("expected system instruction, got nil")
	}
	if len(system.Parts) != 1 || system.Parts[0].Text != "be concise" {
		t.Errorf("system parts = %+v, want [be concise]", system.Parts)
	}
	if len(contents) != 1 {
		t.Fatalf("contents = %d, want 1", len(contents))
	}
	if contents[0].Role != "user" {
		t.Errorf("role = %q, want user", contents[0].Role)
	}
}

func TestConvertPrompt_SystemAfterUser_Errors(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
			SystemMessage{Content: "nope"},
		},
	}
	_, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "system messages are only supported at the beginning") {
		t.Errorf("error = %v, want system-after-user message", err)
	}
}

func TestConvertPrompt_GemmaPrepend(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			SystemMessage{Content: "be terse"},
			UserMessage{Content: []UserContent{TextContent{Text: "hello"}}},
		},
	}
	contents, system, _, err := ConvertPrompt("gemma-3-4b-it", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if system != nil {
		t.Errorf("Gemma: system = %+v, want nil", system)
	}
	if len(contents) != 1 {
		t.Fatalf("contents = %d, want 1", len(contents))
	}
	parts := contents[0].Parts
	if len(parts) < 2 {
		t.Fatalf("parts len = %d, want >= 2", len(parts))
	}
	if !strings.HasPrefix(parts[0].Text, "be terse") {
		t.Errorf("part[0] = %q, want prefix 'be terse'", parts[0].Text)
	}
	if parts[1].Text != "hello" {
		t.Errorf("part[1] = %q, want 'hello'", parts[1].Text)
	}
}

func TestConvertPrompt_ImageData(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{
				TextContent{Text: "what's in this image?"},
				ImageContent{Source: ImageSource{Type: "data", MediaType: "image/png", Data: "AAA="}},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(contents) != 1 {
		t.Fatalf("contents = %d, want 1", len(contents))
	}
	parts := contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2", len(parts))
	}
	if parts[1].InlineData == nil || parts[1].InlineData.MimeType != "image/png" || parts[1].InlineData.Data != "AAA=" {
		t.Errorf("image inline data = %+v, want image/png AAA=", parts[1].InlineData)
	}
}

func TestConvertPrompt_ImageURL(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{
				ImageContent{Source: ImageSource{Type: "url", MediaType: "image/jpeg", URL: "https://x/y.jpg"}},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := contents[0].Parts[0]
	if p.FileData == nil || p.FileData.FileURI != "https://x/y.jpg" {
		t.Errorf("fileData = %+v, want https://x/y.jpg", p.FileData)
	}
}

func TestConvertPrompt_Audio(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{
				AudioContent{Source: ImageSource{Type: "data", MediaType: "audio/mp3", Data: "BBB="}},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := contents[0].Parts[0]
	if p.InlineData == nil || p.InlineData.MimeType != "audio/mp3" {
		t.Errorf("audio inlineData = %+v", p.InlineData)
	}
}

func TestConvertPrompt_VideoURL(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{
				VideoContent{Source: ImageSource{Type: "url", URL: "https://youtu.be/abc123"}},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := contents[0].Parts[0]
	if p.FileData == nil || p.FileData.FileURI != "https://youtu.be/abc123" {
		t.Errorf("video fileData = %+v", p.FileData)
	}
}

func TestConvertPrompt_FileBytes(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{
				FileContent{Data: []byte("hi"), MediaType: "text/plain"},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := contents[0].Parts[0]
	if p.InlineData == nil || p.InlineData.MimeType != "text/plain" {
		t.Errorf("file inlineData = %+v", p.InlineData)
	}
}

func TestConvertPrompt_FileURLString(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{
				FileContent{Data: "https://x/y.pdf", MediaType: "application/pdf"},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := contents[0].Parts[0]
	if p.InlineData == nil {
		t.Errorf("expected inlineData for file URL with mediaType, got %+v", p)
	}
}

func TestConvertPrompt_Reasoning(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
			AssistantMessage{Content: []AssistantContent{
				ReasoningContent{Text: "thinking...", Signature: "sig-1"},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(contents) != 2 {
		t.Fatalf("contents = %d, want 2", len(contents))
	}
	p := contents[1].Parts[0]
	if p.Text != "thinking..." {
		t.Errorf("text = %q, want thinking...", p.Text)
	}
	if p.Thought == nil || !*p.Thought {
		t.Errorf("thought = %v, want true", p.Thought)
	}
	if p.ThoughtSignature != "sig-1" {
		t.Errorf("thoughtSignature = %q, want sig-1", p.ThoughtSignature)
	}
}

func TestConvertPrompt_ToolCallSignatureRoundtrip(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "go"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{
					ToolName:         "get_weather",
					ToolCallID:       "call-1",
					Input:            []byte(`{"city":"sf"}`),
					ProviderMetadata: ProviderMetadata{"google": map[string]any{"thoughtSignature": "sig-tool"}},
				},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := contents[1].Parts[0]
	if p.FunctionCall == nil || p.FunctionCall.Name != "get_weather" {
		t.Fatalf("functionCall = %+v", p.FunctionCall)
	}
	if p.ThoughtSignature != "sig-tool" {
		t.Errorf("thoughtSignature = %q, want sig-tool", p.ThoughtSignature)
	}
}

func TestConvertPrompt_Gemini3_SkipSentinelInjected(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "go"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{
					ToolName:   "search",
					ToolCallID: "call-1",
					Input:      []byte(`{}`),
				},
			}},
		},
	}
	contents, _, warnings, err := ConvertPrompt(ModelGemini3ProPreview, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := contents[1].Parts[0]
	if p.ThoughtSignature != SkipThoughtSignatureValidator {
		t.Errorf("thoughtSignature = %q, want sentinel", p.ThoughtSignature)
	}
	found := false
	for _, w := range warnings {
		if w.Feature == "thoughtSignature" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected thoughtSignature warning, got %+v", warnings)
	}
}

func TestConvertPrompt_ServerToolPairing_AppendsToLastModel(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "search"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{
					ToolName:         "google_search",
					ToolCallID:       "srv-1",
					Input:            []byte(`{}`),
					ProviderMetadata: ProviderMetadata{"google": map[string]any{"serverToolType": "google_search"}},
				},
			}},
			ToolMessage{
				Content: []ToolContent{
					ToolResultContent{
						ToolCallID:      "srv-1",
						Output:          ToolResultOutput{Type: "text", Value: "result"},
						ProviderOptions: ProviderOptions{"google": map[string]any{"serverToolType": "google_search"}},
					},
				},
			},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini3ProPreview, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(contents) != 2 {
		t.Fatalf("contents = %d, want 2 (user, model-with-appended-tool-response)", len(contents))
	}
	if contents[1].Role != "model" {
		t.Errorf("contents[1].role = %q, want model (appended)", contents[1].Role)
	}
	if len(contents[1].Parts) != 2 {
		t.Fatalf("model content parts = %d, want 2 (call + response)", len(contents[1].Parts))
	}
	if contents[1].Parts[1].ToolResponse == nil {
		t.Errorf("expected toolResponse part appended to model content, got %+v", contents[1].Parts[1])
	}
}

func TestConvertPrompt_FunctionResponse_Legacy(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "weather?"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{ToolName: "weather", ToolCallID: "t1", Input: []byte(`{}`)},
			}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{ToolCallID: "t1", Output: ToolResultOutput{Type: "text", Value: "sunny, 70F"}},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Find the user content with the functionResponse.
	var frPart *internal.APIPart
	for _, c := range contents {
		if c.Role == "user" {
			for i, p := range c.Parts {
				if p.FunctionResponse != nil {
					frPart = &contents[len(contents)-1].Parts[i]
				}
			}
		}
	}
	if frPart == nil || frPart.FunctionResponse == nil {
		t.Fatalf("expected a functionResponse part, got %+v", contents)
	}
	if frPart.FunctionResponse.Name != "t1" {
		t.Errorf("functionResponse.name = %q, want t1", frPart.FunctionResponse.Name)
	}
}

func TestConvertPrompt_FunctionResponse_Gemini3_ContentWithImage(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "x"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{ToolName: "render", ToolCallID: "t2", Input: []byte(`{}`)},
			}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{
					ToolCallID: "t2",
					Output: ToolResultOutput{
						Type: "content",
						Value: []ContentPart{
							{Text: "rendered:"},
							{InlineData: &InlineDataPart{MimeType: "image/png", Data: "PNG"}},
						},
					},
				},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini3ProPreview, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var frPart *internal.APIPart
	for i, c := range contents {
		if c.Role == "user" {
			for j, p := range c.Parts {
				if p.FunctionResponse != nil {
					frPart = &contents[i].Parts[j]
				}
			}
		}
	}
	if frPart == nil {
		t.Fatalf("expected a functionResponse part, got %+v", contents)
	}
	if frPart.FunctionResponse.Name != "t2" {
		t.Errorf("functionResponse.name = %q, want t2", frPart.FunctionResponse.Name)
	}
	if len(frPart.FunctionResponse.Parts) != 1 {
		t.Errorf("functionResponse.parts = %d, want 1", len(frPart.FunctionResponse.Parts))
	}
	if frPart.FunctionResponse.Parts[0].InlineData == nil || frPart.FunctionResponse.Parts[0].InlineData.MimeType != "image/png" {
		t.Errorf("functionResponse.parts[0].inlineData = %+v", frPart.FunctionResponse.Parts[0].InlineData)
	}
}

func TestConvertPrompt_AssistantToolResult_Dropped(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "x"}}},
			AssistantMessage{Content: []AssistantContent{
				ToolCallContent{ToolName: "f", ToolCallID: "c", Input: []byte(`{}`)},
			}},
			ToolMessage{Content: []ToolContent{
				ToolResultContent{ToolCallID: "c", Output: ToolResultOutput{Type: "text", Value: "ok"}},
			}},
			AssistantMessage{Content: []AssistantContent{
				TextContent{Text: "got it"},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// After the user-tool-result, the next "model" content should only have
	// the text part, not a tool-result.
	var lastModel *internal.APIContent
	for i := range contents {
		if contents[i].Role == "model" {
			lastModel = &contents[i]
		}
	}
	if lastModel == nil {
		t.Fatal("no model content")
	}
	for _, p := range lastModel.Parts {
		if p.FunctionResponse != nil || p.ToolResponse != nil {
			t.Errorf("assistant side: tool result must be dropped, got %+v", p)
		}
	}
}

func TestConvertPrompt_ExecutableCode(t *testing.T) {
	opts := GenerateOptions{
		Messages: []Message{
			AssistantMessage{Content: []AssistantContent{
				ExecutableCodeContent{Language: "PYTHON", Code: "print(1)"},
				CodeExecutionResultContent{Outcome: "OUTCOME_OK", Output: "1"},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2", len(parts))
	}
	if parts[0].ExecutableCode == nil || parts[0].ExecutableCode.Code != "print(1)" {
		t.Errorf("executableCode = %+v", parts[0].ExecutableCode)
	}
	if parts[1].CodeExecutionResult == nil || parts[1].CodeExecutionResult.Output != "1" {
		t.Errorf("codeExecutionResult = %+v", parts[1].CodeExecutionResult)
	}
}

func TestConvertPrompt_VertexNamespace_Fallback(t *testing.T) {
	opts := GenerateOptions{
		ProviderOptions: ProviderOptions{
			"googleVertex": map[string]any{
				"responseModalities": []any{"TEXT"},
			},
		},
		Messages: []Message{
			UserMessage{Content: []UserContent{TextContent{Text: "hi"}}},
		},
	}
	// When not vertex-like, vertex options are ignored.
	_, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// And we don't check GoogleOptions here; the test is just that prompt
	// conversion does not depend on provider options, so no error.
}

func TestConvertPrompt_EmptyMessages(t *testing.T) {
	contents, system, _, err := ConvertPrompt(ModelGemini35Flash, GenerateOptions{Messages: nil})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(contents) != 0 {
		t.Errorf("contents = %d, want 0", len(contents))
	}
	if system != nil {
		t.Errorf("system = %+v, want nil", system)
	}
}

func TestConvertPrompt_AllUserContent(t *testing.T) {
	// Mix of Text, Image, File into a single user message.
	opts := GenerateOptions{
		Messages: []Message{
			UserMessage{Content: []UserContent{
				TextContent{Text: "look:"},
				ImageContent{Source: ImageSource{Type: "data", MediaType: "image/jpeg", Data: "A"}},
				FileContent{Data: "https://x/y.txt"},
			}},
		},
	}
	contents, _, _, err := ConvertPrompt(ModelGemini35Flash, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := contents[0].Parts
	if len(parts) != 3 {
		t.Errorf("parts = %d, want 3", len(parts))
	}
}
