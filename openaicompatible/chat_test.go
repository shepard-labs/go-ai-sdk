package openaicompatible

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"testing"
	"time"
)

func TestChatGenerateRequestStandardOptionsAndProviderOptions(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"id":"resp","created":10,"model":"chat-model","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test/v1", Name: "acme-provider", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	maxTokens := 10
	temperature := 0.5
	topK := 2
	topP := 0.7
	frequencyPenalty := 0.1
	presencePenalty := 0.2
	seed := 123
	result, err := p.Chat("chat-model").DoGenerate(context.Background(), GenerateOptions{
		Messages:         []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}},
		MaxOutputTokens:  &maxTokens,
		Temperature:      &temperature,
		TopK:             &topK,
		TopP:             &topP,
		StopSequences:    []string{"stop"},
		FrequencyPenalty: &frequencyPenalty,
		PresencePenalty:  &presencePenalty,
		Seed:             &seed,
		ProviderOptions: ProviderOptions{
			"openai-compatible": map[string]any{"user": "deprecated", "reasoningEffort": "low"},
			"openaiCompatible":  map[string]any{"reasoningEffort": "medium"},
			"acme-provider":     map[string]any{"reasoningEffort": "high", "textVerbosity": "low", "passthrough": "raw", "strictJsonSchema": false},
			"acmeProvider":      map[string]any{"user": "user-1", "extra": 7},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate error = %v", err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if body["model"] != "chat-model" || body["user"] != "user-1" || body["max_tokens"].(float64) != 10 || body["temperature"].(float64) != 0.5 || body["top_p"].(float64) != 0.7 || body["frequency_penalty"].(float64) != 0.1 || body["presence_penalty"].(float64) != 0.2 || body["seed"].(float64) != 123 || body["reasoning_effort"] != "high" || body["verbosity"] != "low" || body["passthrough"] != "raw" || body["extra"].(float64) != 7 {
		t.Fatalf("body = %#v", body)
	}
	if _, ok := body["top_k"]; ok {
		t.Fatalf("top_k was sent: %#v", body)
	}
	if _, ok := body["strictJsonSchema"]; ok {
		t.Fatalf("recognized option leaked: %#v", body)
	}
	if len(result.Warnings) != 2 || result.Warnings[0].Message != deprecatedProviderOptionsWarningMessage || result.Warnings[1].Type != "unsupported" || result.Warnings[1].Feature != "topK" {
		t.Fatalf("warnings = %#v", result.Warnings)
	}
	if f.requests[0].URL.Path != "/v1/chat/completions" {
		t.Fatalf("path = %q", f.requests[0].URL.Path)
	}
}

func TestChatGenerateOmitsAbsentOptionalFields(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	for _, key := range []string{"user", "max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "response_format", "stop", "seed", "reasoning_effort", "verbosity", "tools", "tool_choice"} {
		if _, ok := body[key]; ok {
			t.Fatalf("optional field %q present in %#v", key, body)
		}
	}
}

func TestChatGenerateResponseFormatAndTransform(t *testing.T) {
	strictFalse := false
	for name, tc := range map[string]struct {
		supportsStructured bool
		options            GenerateOptions
		providerOptions    ProviderOptions
		wantType           string
		wantStrict         any
		wantWarning        string
	}{
		"schema without structured outputs": {
			options:     GenerateOptions{ResponseFormat: &ResponseFormat{Type: "json", Schema: map[string]any{"type": "object"}}},
			wantType:    "json_object",
			wantWarning: "responseFormat",
		},
		"schema with structured outputs": {
			supportsStructured: true,
			options:            GenerateOptions{ResponseFormat: &ResponseFormat{Type: "json", Schema: map[string]any{"type": "object"}}},
			wantType:           "json_schema",
			wantStrict:         true,
		},
		"strict false": {
			supportsStructured: true,
			options:            GenerateOptions{ResponseFormat: &ResponseFormat{Type: "json", Schema: map[string]any{"type": "object"}}},
			providerOptions:    ProviderOptions{"acme": map[string]any{"strictJsonSchema": &strictFalse}},
			wantType:           "json_schema",
			wantStrict:         false,
		},
		"structured output precedence": {
			supportsStructured: true,
			options: GenerateOptions{
				ResponseFormat:   &ResponseFormat{Type: "text"},
				StructuredOutput: &StructuredOutput{Name: "custom", Description: "desc", Schema: map[string]any{"type": "object"}},
			},
			wantType:    "json_schema",
			wantStrict:  true,
			wantWarning: "StructuredOutput takes precedence over ResponseFormat.",
		},
	} {
		t.Run(name, func(t *testing.T) {
			f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
			p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, SupportsStructuredOutputs: tc.supportsStructured, TransformRequestBody: func(body map[string]any) map[string]any {
				body["transformed"] = true
				return body
			}})
			opts := tc.options
			opts.Messages = []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}
			opts.ProviderOptions = tc.providerOptions
			result, err := p.Chat("m").DoGenerate(context.Background(), opts)
			if err != nil {
				t.Fatal(err)
			}
			body := decodeRequestBody(t, result.Request.Body)
			if body["transformed"] != true {
				t.Fatalf("transform missing: %#v", body)
			}
			format := body["response_format"].(map[string]any)
			if format["type"] != tc.wantType {
				t.Fatalf("response_format = %#v", format)
			}
			if tc.wantStrict != nil {
				jsonSchema := format["json_schema"].(map[string]any)
				if jsonSchema["name"] == "" || jsonSchema["strict"] != tc.wantStrict {
					t.Fatalf("json_schema = %#v", jsonSchema)
				}
			}
			if tc.wantWarning != "" && len(result.Warnings) == 0 {
				t.Fatalf("missing warning")
			}
		})
	}
}

func TestChatMessageConversion(t *testing.T) {
	imageURL, _ := url.Parse("https://example.test/image.png")
	textURL, _ := url.Parse("https://example.test/file.txt")
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	prompt := []Message{}
	prompt = append(prompt, SystemMessage{Content: "sys", ProviderOptions: ProviderMetadata{"openaiCompatible": map[string]any{"cache_control": "system"}}})
	prompt = append(prompt, UserMessage{Content: []UserContent{TextContent{Text: "single", ProviderOptions: ProviderMetadata{"openaiCompatible": map[string]any{"single": true}}}}})
	prompt = append(prompt, UserMessage{ProviderOptions: ProviderMetadata{"openaiCompatible": map[string]any{"msg": true}}, Content: []UserContent{
		TextContent{Text: "multi", ProviderOptions: ProviderMetadata{"openaiCompatible": map[string]any{"part": "text"}}},
		FileContent{MediaType: "image/*", Data: []byte("img")},
		FileContent{MediaType: "image/png", Data: imageURL},
		FileContent{MediaType: "audio/wav", Data: []byte("wav")},
		FileContent{MediaType: "audio/mpeg", Data: base64.StdEncoding.EncodeToString([]byte("mp3"))},
		FileContent{MediaType: "application/pdf", Data: []byte("pdf")},
		FileContent{MediaType: "text/plain", Data: base64.StdEncoding.EncodeToString([]byte("hello text"))},
		FileContent{MediaType: "text/plain", Data: textURL},
	}})
	prompt = append(prompt, AssistantMessage{Content: []AssistantContent{
		TextContent{Text: "answer"},
		ReasoningContent{Text: "think"},
		ToolCallContent{ToolCallID: "call-1", ToolName: "weather", Input: json.RawMessage(`{"city":"SF"}`), ProviderOptions: ProviderMetadata{"openaiCompatible": map[string]any{"tc": true}, "google": map[string]any{"thoughtSignature": 123}}},
	}})
	prompt = append(prompt, AssistantMessage{Content: []AssistantContent{ToolCallContent{ToolCallID: "call-2", ToolName: "empty", Input: json.RawMessage(`{}`)}}})
	prompt = append(prompt, AssistantMessage{})
	prompt = append(prompt, ToolMessage{Content: []ToolContent{
		ToolResultContent{ToolCallID: "call-1", Output: ToolResultOutput{Type: "text", Value: "plain"}, ProviderOptions: ProviderMetadata{"openaiCompatible": map[string]any{"tool": true}}},
		ToolResultContent{ToolCallID: "call-2", Output: ToolResultOutput{Type: "error-text", Value: "bad"}},
		ToolResultContent{ToolCallID: "call-3", Output: ToolResultOutput{Type: "execution-denied"}},
		ToolResultContent{ToolCallID: "call-4", Output: ToolResultOutput{Type: "json", Value: map[string]any{"ok": true}}},
		ToolResultContent{ToolCallID: "call-5", Output: ToolResultOutput{Type: "tool-approval-response", Value: "skip"}},
	}})
	_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: prompt})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBodyFromFetcher(t, f)
	messages := body["messages"].([]any)
	if messages[0].(map[string]any)["cache_control"] != "system" || messages[1].(map[string]any)["content"] != "single" || messages[1].(map[string]any)["single"] != true {
		t.Fatalf("metadata/string messages = %#v", messages[:2])
	}
	parts := messages[2].(map[string]any)["content"].([]any)
	if messages[2].(map[string]any)["msg"] != true || parts[0].(map[string]any)["part"] != "text" {
		t.Fatalf("multipart metadata = %#v", messages[2])
	}
	if got := parts[1].(map[string]any)["image_url"].(map[string]any)["url"]; got != "data:image/jpeg;base64,"+base64.StdEncoding.EncodeToString([]byte("img")) {
		t.Fatalf("image bytes = %#v", got)
	}
	if got := parts[2].(map[string]any)["image_url"].(map[string]any)["url"]; got != imageURL.String() {
		t.Fatalf("image URL = %#v", got)
	}
	if parts[3].(map[string]any)["input_audio"].(map[string]any)["format"] != "wav" || parts[4].(map[string]any)["input_audio"].(map[string]any)["format"] != "mp3" {
		t.Fatalf("audio = %#v %#v", parts[3], parts[4])
	}
	if file := parts[5].(map[string]any)["file"].(map[string]any); file["filename"] != "document.pdf" || file["file_data"] != "data:application/pdf;base64,"+base64.StdEncoding.EncodeToString([]byte("pdf")) {
		t.Fatalf("pdf = %#v", file)
	}
	if parts[6].(map[string]any)["text"] != "hello text" || parts[7].(map[string]any)["text"] != textURL.String() {
		t.Fatalf("text files = %#v %#v", parts[6], parts[7])
	}
	assistant := messages[3].(map[string]any)
	if assistant["content"] != "answer" || assistant["reasoning_content"] != "think" {
		t.Fatalf("assistant = %#v", assistant)
	}
	toolCall := assistant["tool_calls"].([]any)[0].(map[string]any)
	if toolCall["tc"] != true || toolCall["extra_content"].(map[string]any)["google"].(map[string]any)["thought_signature"] != "123" {
		t.Fatalf("tool call = %#v", toolCall)
	}
	if messages[4].(map[string]any)["content"] != nil || messages[5].(map[string]any)["content"] != "" {
		t.Fatalf("assistant empty content = %#v %#v", messages[4], messages[5])
	}
	toolMessages := messages[6:10]
	if toolMessages[0].(map[string]any)["content"] != "plain" || toolMessages[0].(map[string]any)["tool"] != true || toolMessages[1].(map[string]any)["content"] != "bad" || toolMessages[2].(map[string]any)["content"] != "Tool execution denied." || toolMessages[3].(map[string]any)["content"] != `{"ok":true}` {
		t.Fatalf("tool messages = %#v", toolMessages)
	}
}

func TestChatMessageConversionErrors(t *testing.T) {
	audioURL, _ := url.Parse("https://example.test/audio.wav")
	pdfURL, _ := url.Parse("https://example.test/doc.pdf")
	for name, content := range map[string]UserContent{
		"audio URL":         FileContent{MediaType: "audio/wav", Data: audioURL},
		"PDF URL":           FileContent{MediaType: "application/pdf", Data: pdfURL},
		"unsupported audio": FileContent{MediaType: "audio/ogg", Data: []byte("x")},
		"unsupported file":  FileContent{MediaType: "application/octet-stream", Data: []byte("x")},
	} {
		t.Run(name, func(t *testing.T) {
			p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: &recordingFetcher{}, Retry: &RetryOptions{MaxRetries: 0}})
			_, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{content}}}})
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestChatToolConversionAndToolChoice(t *testing.T) {
	strict := true
	for name, choice := range map[string]*ToolChoice{
		"auto":     {Type: "auto"},
		"none":     {Type: "none"},
		"required": {Type: "required"},
		"tool":     {Type: "tool", ToolName: "weather"},
	} {
		t.Run(name, func(t *testing.T) {
			f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
			p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
			result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{
				Messages:   []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}},
				ToolChoice: choice,
				Tools: []Tool{
					{Type: "function", Name: "weather", Description: "desc", InputSchema: map[string]any{"type": "object"}, Strict: &strict},
					{Type: "provider", ID: "web-search"},
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			body := decodeRequestBody(t, result.Request.Body)
			tools := body["tools"].([]any)
			if len(tools) != 1 || tools[0].(map[string]any)["function"].(map[string]any)["strict"] != true {
				t.Fatalf("tools = %#v", tools)
			}
			if _, ok := body["tool_choice"]; !ok {
				t.Fatalf("tool_choice missing: %#v", body)
			}
			if len(result.Warnings) != 1 || result.Warnings[0].Feature != "provider-defined tool web-search" {
				t.Fatalf("warnings = %#v", result.Warnings)
			}
		})
	}
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}]}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	body := decodeRequestBody(t, result.Request.Body)
	if _, ok := body["tools"]; ok {
		t.Fatalf("tools present: %#v", body)
	}
	_, err = p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, Tools: []Tool{{Type: "function", Name: "weather"}}, ToolChoice: &ToolChoice{Type: "bad"}})
	if err == nil || err.Error() != "openaicompatible: unsupported functionality: tool choice type: bad" {
		t.Fatalf("unsupported choice error = %v", err)
	}
}

func TestChatGenerateResponseParsingMetadataUsageAndSupportedURLs(t *testing.T) {
	extractor := &recordingMetadataExtractor{metadata: ProviderMetadata{"acmeProvider": map[string]any{"custom": "value"}}}
	created := int64(1710000000)
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"id":"resp-id","created":1710000000,"model":"served-model","choices":[{"message":{"content":"text","reasoning_content":"preferred","reasoning":"fallback","tool_calls":[{"function":{"name":"weather","arguments":"{ \"b\" : 2 }"},"extra_content":{"google":{"thought_signature":"sig"}}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":6,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2,"accepted_prediction_tokens":4,"rejected_prediction_tokens":5}}}`)}}
	p := CreateOpenAICompatible(ProviderSettings{
		BaseURL: "https://example.test", Name: "acme-provider", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, GenerateID: func() string { return "generated-id" }, MetadataExtractor: extractor,
		SupportedURLs: func() map[string][]*regexp.Regexp {
			return map[string][]*regexp.Regexp{"docs": {regexp.MustCompile(`https://docs\.example`)}}
		},
	})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}, ProviderOptions: ProviderOptions{"acmeProvider": map[string]any{}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 3 {
		t.Fatalf("content = %#v", result.Content)
	}
	if result.Content[0].(TextContent).Text != "text" || result.Content[1].(ReasoningContent).Text != "preferred" {
		t.Fatalf("content order = %#v", result.Content)
	}
	toolCall := result.Content[2].(ToolCallContent)
	if toolCall.ToolCallID != "generated-id" || toolCall.ToolName != "weather" || string(toolCall.Input) != `{ "b" : 2 }` {
		t.Fatalf("tool call = %#v", toolCall)
	}
	if toolCall.ProviderMetadata["acmeProvider"].(map[string]any)["thoughtSignature"] != "sig" {
		t.Fatalf("tool metadata = %#v", toolCall.ProviderMetadata)
	}
	if result.FinishReason.Unified != "tool-calls" || result.FinishReason.Raw != "tool_calls" {
		t.Fatalf("finish = %#v", result.FinishReason)
	}
	if *result.Usage.InputTokens.Total != 10 || *result.Usage.InputTokens.NoCache != 7 || *result.Usage.InputTokens.CacheRead != 3 || *result.Usage.OutputTokens.Total != 6 || *result.Usage.OutputTokens.Text != 4 || *result.Usage.OutputTokens.Reasoning != 2 || len(result.Usage.Raw) == 0 {
		t.Fatalf("usage = %#v", result.Usage)
	}
	metadata := result.ProviderMetadata["acmeProvider"].(map[string]any)
	if metadata["acceptedPredictionTokens"] != 4 || metadata["rejectedPredictionTokens"] != 5 || metadata["custom"] != "value" {
		t.Fatalf("metadata = %#v", result.ProviderMetadata)
	}
	if string(extractor.raw) != string(result.Response.Body) || extractor.decoded["id"] != "resp-id" {
		t.Fatalf("extractor raw=%s decoded=%#v", extractor.raw, extractor.decoded)
	}
	if result.Response.ID != "resp-id" || result.Response.ModelID != "served-model" || !result.Response.Timestamp.Equal(time.Unix(created, 0)) || result.Response.Headers.Get("x-request-id") != "resp-id" || len(result.Response.Body) == 0 {
		t.Fatalf("response = %#v", result.Response)
	}
	urls := p.Chat("m").SupportURLs()
	urls["docs"] = nil
	if len(p.Chat("m").SupportURLs()["docs"]) != 1 {
		t.Fatalf("SupportURLs did not clone")
	}
}

func TestChatGenerateCustomUsageConverter(t *testing.T) {
	f := &recordingFetcher{responses: []*http.Response{response(200, `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1}}`)}}
	p := CreateOpenAICompatible(ProviderSettings{BaseURL: "https://example.test", Name: "acme", Fetch: f, Retry: &RetryOptions{MaxRetries: 0}, ConvertUsage: func(OpenAICompatibleTokenUsage) Usage {
		return Usage{InputTokens: TokenCounts{Total: intPtr(99)}}
	}})
	result, err := p.Chat("m").DoGenerate(context.Background(), GenerateOptions{Messages: []Message{UserMessage{Content: []UserContent{TextContent{Text: "hello"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	if *result.Usage.InputTokens.Total != 99 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}

type recordingMetadataExtractor struct {
	raw             []byte
	decoded         map[string]any
	metadata        ProviderMetadata
	streamExtractor StreamMetadataExtractor
}

func (e *recordingMetadataExtractor) ExtractMetadata(raw []byte, decoded map[string]any) (ProviderMetadata, error) {
	e.raw = append([]byte(nil), raw...)
	e.decoded = decoded
	return e.metadata, nil
}

func (e *recordingMetadataExtractor) CreateStreamExtractor() StreamMetadataExtractor {
	return e.streamExtractor
}

func decodeRequestBodyFromFetcher(t *testing.T, f *recordingFetcher) map[string]any {
	t.Helper()
	if len(f.requests) == 0 {
		t.Fatal("no request recorded")
	}
	var body map[string]any
	if err := json.NewDecoder(f.requests[0].Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	return body
}

func decodeRequestBody(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	return body
}
