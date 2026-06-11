package google

// chat.go implements the Google Generative AI language model (non-streaming).
//
// Mirrors the upstream google-language-model.ts doGenerate path:
//
//   1. getArgs(opts) — merge provider options into a typed GoogleOptions
//      view + a passthrough map; resolve thinking config; resolve
//      safetySettings with the top-level Threshold default.
//   2. ConvertPrompt(opts) → []APIContent + optional systemInstruction.
//   3. buildChatRequest — assemble the body map.
//   4. POST :generateContent via executeJSON.
//   5. parseGenerateResponse — decode the response into a GenerateResult.

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
	"github.com/shepard-labs/go-ai-sdk/google/tools"
)

// googleLanguageModel implements the LanguageModel interface.
type googleLanguageModel struct {
	provider *googleProvider
	modelID  string
}

func (m *googleLanguageModel) ModelID() string  { return m.modelID }
func (m *googleLanguageModel) Provider() string { return m.provider.name + ".chat" }

func (m *googleLanguageModel) SupportURLs() map[string][]*regexp.Regexp {
	return cloneRegexpMap(m.provider.SupportURLs())
}

// getArgsResult is the typed view returned by getArgs.
type getArgsResult struct {
	Options      GoogleOptions
	Passthrough  map[string]any
	Warnings     []Warning
	IsVertexLike bool
	IsVertex     bool
}

// getArgs merges the ProviderOptions map into a typed GoogleOptions view and
// collects the unrecognized keys into a passthrough map for body spread.
func (m *googleLanguageModel) getArgs(opts GenerateOptions) (getArgsResult, error) {
	raw := cloneProviderOptions(opts.ProviderOptions)
	vertexLike := isVertexLike(m.provider.baseURL, m.provider.useVertexAIHeaders, raw)
	typed := googleOptionsFromProviderOptions(raw, vertexLike)

	// Resolve per-request provider options that aren't covered by the
	// googleOptionsFromProviderOptions helper.
	merged := mergeGoogleNamespaces(raw, vertexLike)
	if v, ok := merged["thinkingConfig"]; ok {
		if tc, ok := thinkingConfigFromAny(v); ok {
			typed.ThinkingConfig = tc
		}
	}
	if v, ok := merged["safetySettings"]; ok {
		if ss, ok := safetySettingsFromAny(v); ok {
			typed.SafetySettings = ss
		}
	}
	if v, ok := merged["imageConfig"]; ok {
		if ic, ok := imageConfigFromAny(v); ok {
			typed.ImageConfig = ic
		}
	}
	if v, ok := merged["retrievalConfig"]; ok {
		if rc, ok := retrievalConfigFromAny(v); ok {
			typed.RetrievalConfig = rc
		}
	}

	// Apply top-level Threshold default to empty-Threshold safetySettings.
	if typed.Threshold != "" {
		for i := range typed.SafetySettings {
			if typed.SafetySettings[i].Threshold == "" {
				typed.SafetySettings[i].Threshold = typed.Threshold
			}
		}
	}

	// Pull out known/recognized keys; everything else is passthrough.
	passthrough := map[string]any{}
	for k, v := range merged {
		if isRecognizedGoogleOptionKey(k) {
			continue
		}
		passthrough[k] = v
	}

	// Vertex-only header warnings.
	var warnings []Warning
	if (typed.SharedRequestType != "" || typed.RequestType != "") && !m.provider.isVertex {
		warnings = append(warnings, Warning{
			Type:    "other",
			Feature: "sharedRequestType/requestType",
			Message: "sharedRequestType / requestType are Vertex-only and will be ignored on the public Gemini API",
		})
	}

	// Per-GenerateOptions.Reasoning resolution: when Reasoning is set, it
	// takes precedence over provider options. We resolve into ThinkingConfig
	// here.
	if opts.Reasoning != "" {
		resolved := mapReasoningForModel(m.modelID, opts.Reasoning, typed.ThinkingConfig, opts.MaxOutputTokens)
		typed.ThinkingConfig = resolved
	}

	return getArgsResult{
		Options:      typed,
		Passthrough:  passthrough,
		Warnings:     warnings,
		IsVertexLike: vertexLike,
		IsVertex:     m.provider.isVertex,
	}, nil
}

// recognizedGoogleOptionKeys are the camelCase keys pulled out of the
// providerOptions map; everything else is forwarded as body passthrough.
var recognizedGoogleOptionKeys = map[string]struct{}{
	"responseModalities":          {},
	"thinkingConfig":              {},
	"cachedContent":               {},
	"structuredOutputs":           {},
	"safetySettings":              {},
	"audioTimestamp":              {},
	"labels":                      {},
	"mediaResolution":             {},
	"imageConfig":                 {},
	"retrievalConfig":             {},
	"streamFunctionCallArguments": {},
	"serviceTier":                 {},
	"sharedRequestType":           {},
	"requestType":                 {},
	"threshold":                   {},
	"generationConfig":            {},
}

func isRecognizedGoogleOptionKey(k string) bool {
	_, ok := recognizedGoogleOptionKeys[k]
	return ok
}

// thinkingConfigFromAny coerces an arbitrary JSON value to *ThinkingConfig.
func thinkingConfigFromAny(v any) (*ThinkingConfig, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	out := &ThinkingConfig{}
	if b, ok := m["includeThoughts"].(bool); ok {
		out.IncludeThoughts = &b
	}
	if n, ok := toInt(m["thinkingBudget"]); ok {
		out.ThinkingBudget = &n
	}
	if s, ok := m["thinkingLevel"].(string); ok {
		out.ThinkingLevel = s
	}
	return out, true
}

func safetySettingsFromAny(v any) ([]SafetySetting, bool) {
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]SafetySetting, 0, len(arr))
	for _, item := range arr {
		m, ok := item.(map[string]any)
		if !ok {
			return nil, false
		}
		s := SafetySetting{}
		if s2, ok := m["category"].(string); ok {
			s.Category = s2
		}
		if s2, ok := m["threshold"].(string); ok {
			s.Threshold = s2
		}
		out = append(out, s)
	}
	return out, true
}

func imageConfigFromAny(v any) (*ImageConfig, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	out := &ImageConfig{}
	if s, ok := m["aspectRatio"].(string); ok {
		out.AspectRatio = s
	}
	if s, ok := m["imageSize"].(string); ok {
		out.ImageSize = s
	}
	return out, true
}

func retrievalConfigFromAny(v any) (*RetrievalConfig, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	out := &RetrievalConfig{}
	if raw, ok := m["latLng"]; ok {
		if mm, ok := raw.(map[string]any); ok {
			ll := &LatLng{}
			if n, ok := toFloat(mm["latitude"]); ok {
				ll.Latitude = n
			}
			if n, ok := toFloat(mm["longitude"]); ok {
				ll.Longitude = n
			}
			out.LatLng = ll
		}
	}
	return out, true
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}

// mapReasoningForModel resolves opts.Reasoning into a ThinkingConfig,
// honoring model-specific budget/level semantics. Existing provider
// options are merged on top (shallow).
func mapReasoningForModel(modelID, reasoning string, existing *ThinkingConfig, maxTokens *int) *ThinkingConfig {
	out := &ThinkingConfig{}
	if existing != nil {
		*out = *existing
	}
	if isGemini3(modelID) {
		// Gemini 3: thinkingLevel
		switch reasoning {
		case "none":
			out.ThinkingLevel = "minimal"
		case "minimal", "low", "medium", "high":
			out.ThinkingLevel = reasoning
		case "xhigh":
			out.ThinkingLevel = "high"
		default:
			out.ThinkingLevel = reasoning
		}
		return out
	}
	// Gemini 2.5 (and pro-image): thinkingBudget
	if reasoning == "none" {
		zero := 0
		out.ThinkingBudget = &zero
		return out
	}
	// Convert reasoning effort → budget fraction.
	cap := 32768
	if isGemini2Point5(modelID) {
		// 2.5 pro / 3-pro-image use 32768; other 2.5 use 24576.
		if isProImage(modelID) {
			cap = 32768
		} else if strings.Contains(modelID, "2.5-pro") {
			cap = 32768
		} else {
			cap = 24576
		}
	}
	max := 65536
	if maxTokens != nil {
		max = *maxTokens
	}
	if max < cap {
		cap = max
	}
	var budget int
	switch reasoning {
	case "minimal", "low":
		budget = cap / 4
	case "medium":
		budget = cap / 2
	case "high":
		budget = int(float64(cap) * 0.8)
	case "xhigh":
		budget = cap
	default:
		budget = cap / 2
	}
	if budget < 0 {
		budget = 0
	}
	if budget > max {
		budget = max
	}
	out.ThinkingBudget = &budget
	return out
}

func isProImage(modelID string) bool {
	id := strings.ToLower(modelID)
	return strings.Contains(id, "image")
}

// buildChatRequest assembles the request body as map[string]any and returns
// any extra headers to attach (e.g. Vertex LLM headers) plus any non-fatal
// warnings produced by the tool dispatcher. The body shape mirrors the
// §"Chat Request Body" spec.
func (m *googleLanguageModel) buildChatRequest(args getArgsResult, opts GenerateOptions, contents []internal.APIContent, system *internal.APIContent) (map[string]any, http.Header, []Warning, error) {
	body := map[string]any{
		"contents": contentsToMaps(contents),
	}
	if system != nil {
		body["systemInstruction"] = contentToMap(*system)
	}

	gc := map[string]any{}

	if opts.MaxOutputTokens != nil {
		gc["maxOutputTokens"] = *opts.MaxOutputTokens
	}
	if opts.Temperature != nil {
		gc["temperature"] = *opts.Temperature
	}
	if opts.TopK != nil {
		gc["topK"] = *opts.TopK
	}
	if opts.TopP != nil {
		gc["topP"] = *opts.TopP
	}
	if opts.FrequencyPenalty != nil {
		gc["frequencyPenalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		gc["presencePenalty"] = *opts.PresencePenalty
	}
	if len(opts.StopSequences) > 0 {
		gc["stopSequences"] = append([]string(nil), opts.StopSequences...)
	}
	if opts.Seed != nil {
		gc["seed"] = *opts.Seed
	}

	if opts.ResponseFormat != nil || opts.StructuredOutput != nil {
		rf := opts.ResponseFormat
		so := opts.StructuredOutput
		if so != nil || (rf != nil && rf.Type == "json") {
			gc["responseMimeType"] = "application/json"
			var schema any
			var name string
			if so != nil {
				schema = so.Schema
				name = so.Name
			} else if rf != nil {
				schema = rf.Schema
				name = rf.Name
			}
			if schema != nil {
				gc["responseSchema"] = internal.ConvertJSONSchemaToOpenAPISchema(schema)
			}
			if name != "" {
				if _, ok := gc["responseSchema"].(map[string]any); ok {
					gc["responseSchema"].(map[string]any)["title"] = name
				}
			}
		}
	}

	if args.Options.ResponseModalities != nil {
		gc["responseModalities"] = append([]string(nil), args.Options.ResponseModalities...)
	}
	if args.Options.AudioTimestamp != nil {
		gc["audioTimestamp"] = *args.Options.AudioTimestamp
	}
	if args.Options.MediaResolution != "" {
		gc["mediaResolution"] = args.Options.MediaResolution
	}
	if args.Options.ImageConfig != nil {
		ic := map[string]any{}
		if args.Options.ImageConfig.AspectRatio != "" {
			ic["aspectRatio"] = args.Options.ImageConfig.AspectRatio
		}
		if args.Options.ImageConfig.ImageSize != "" {
			ic["imageSize"] = args.Options.ImageConfig.ImageSize
		}
		gc["imageConfig"] = ic
	}
	if args.Options.ThinkingConfig != nil {
		tc := map[string]any{}
		if args.Options.ThinkingConfig.IncludeThoughts != nil {
			tc["includeThoughts"] = *args.Options.ThinkingConfig.IncludeThoughts
		}
		if args.Options.ThinkingConfig.ThinkingBudget != nil {
			tc["thinkingBudget"] = *args.Options.ThinkingConfig.ThinkingBudget
		}
		if args.Options.ThinkingConfig.ThinkingLevel != "" {
			tc["thinkingLevel"] = args.Options.ThinkingConfig.ThinkingLevel
		}
		gc["thinkingConfig"] = tc
	}
	if len(gc) > 0 {
		body["generationConfig"] = gc
	}

	if len(args.Options.SafetySettings) > 0 {
		ss := make([]map[string]any, 0, len(args.Options.SafetySettings))
		for _, s := range args.Options.SafetySettings {
			entry := map[string]any{"category": s.Category}
			if s.Threshold != "" {
				entry["threshold"] = s.Threshold
			}
			ss = append(ss, entry)
		}
		body["safetySettings"] = ss
	}

	// Tools.
	views := make([]tools.ToolView, 0, len(opts.Tools))
	for _, t := range opts.Tools {
		views = append(views, tools.ToolView{
			Type:             t.Type,
			ID:               t.ID,
			Name:             t.Name,
			Description:      t.Description,
			InputSchema:      t.InputSchema,
			ArgsSchema:       t.ArgsSchema,
			Strict:           t.Strict,
			ProviderExecuted: t.ProviderExecuted,
			Dynamic:          t.Dynamic,
		})
	}
	var toolChoiceView *tools.ToolChoiceView
	if opts.ToolChoice != nil {
		toolChoiceView = &tools.ToolChoiceView{Type: opts.ToolChoice.Type, ToolName: opts.ToolChoice.ToolName}
	}
	apiTools, toolCfg, toolWarnings, err := tools.PrepareTools(views, tools.PrepareToolsOpts{
		ModelID:                     m.modelID,
		IsVertexProvider:            args.IsVertex,
		StreamFunctionCallArguments: args.Options.StreamFunctionCallArguments,
		ToolChoice:                  toolChoiceView,
	})
	if err != nil {
		return nil, nil, nil, err
	}
	convertedWarnings := make([]Warning, 0, len(toolWarnings))
	for _, w := range toolWarnings {
		convertedWarnings = append(convertedWarnings, Warning{
			Type:    w.Type,
			Feature: w.Feature,
			Details: w.Details,
			Message: w.Message,
		})
	}
	args.Warnings = append(args.Warnings, convertedWarnings...)

	if len(apiTools) > 0 {
		toolBodies := make([]map[string]any, 0, len(apiTools))
		for _, t := range apiTools {
			toolBodies = append(toolBodies, t.Body)
		}
		body["tools"] = toolBodies
	}
	if toolCfg != nil {
		cfg := map[string]any{}
		if toolCfg.FunctionCallingConfig != nil {
			fcc := map[string]any{}
			if toolCfg.FunctionCallingConfig.Mode != "" {
				fcc["mode"] = toolCfg.FunctionCallingConfig.Mode
			}
			if len(toolCfg.FunctionCallingConfig.AllowedFunctionNames) > 0 {
				fcc["allowedFunctionNames"] = append([]string(nil), toolCfg.FunctionCallingConfig.AllowedFunctionNames...)
			}
			if toolCfg.FunctionCallingConfig.StreamFunctionCallArguments != nil {
				fcc["streamFunctionCallArguments"] = *toolCfg.FunctionCallingConfig.StreamFunctionCallArguments
			}
			if len(fcc) > 0 {
				cfg["functionCallingConfig"] = fcc
			}
		}
		if toolCfg.IncludeServerSideToolInvocations != nil {
			cfg["includeServerSideToolInvocations"] = *toolCfg.IncludeServerSideToolInvocations
		}
		if len(cfg) > 0 {
			body["toolConfig"] = cfg
		}
	}

	if args.Options.CachedContent != "" {
		body["cachedContent"] = args.Options.CachedContent
	}
	if len(args.Options.Labels) > 0 {
		body["labels"] = cloneStringMap(args.Options.Labels)
	}
	if args.Options.ServiceTier != "" && !args.IsVertex {
		body["serviceTier"] = args.Options.ServiceTier
	}

	// retrievalConfig: top-level (sibling of toolConfig).
	if args.Options.RetrievalConfig != nil && args.Options.RetrievalConfig.LatLng != nil {
		ll := args.Options.RetrievalConfig.LatLng
		toolConfig, _ := body["toolConfig"].(map[string]any)
		if toolConfig == nil {
			toolConfig = map[string]any{}
		}
		toolConfig["retrievalConfig"] = map[string]any{
			"latLng": map[string]any{
				"latitude":  ll.Latitude,
				"longitude": ll.Longitude,
			},
		}
		body["toolConfig"] = toolConfig
	}

	// Passthrough spread.
	for k, v := range args.Passthrough {
		if _, taken := body[k]; taken {
			continue
		}
		body[k] = v
	}

	// Vertex-only headers.
	headers := http.Header{}
	if args.IsVertex {
		if args.Options.SharedRequestType != "" {
			headers.Set("X-Vertex-AI-LLM-Shared-Request-Type", args.Options.SharedRequestType)
		}
		if args.Options.RequestType != "" {
			headers.Set("X-Vertex-AI-LLM-Request-Type", "shared")
		}
	}

	return body, headers, convertedWarnings, nil
}

func contentsToMaps(contents []internal.APIContent) []map[string]any {
	out := make([]map[string]any, 0, len(contents))
	for _, c := range contents {
		out = append(out, contentToMap(c))
	}
	return out
}

func contentToMap(c internal.APIContent) map[string]any {
	parts := make([]map[string]any, 0, len(c.Parts))
	for _, p := range c.Parts {
		parts = append(parts, partToMap(p))
	}
	return map[string]any{
		"role":  c.Role,
		"parts": parts,
	}
}

func partToMap(p internal.APIPart) map[string]any {
	out := map[string]any{}
	if p.Text != "" {
		out["text"] = p.Text
	}
	if p.Thought != nil {
		out["thought"] = *p.Thought
	}
	if p.ThoughtSignature != "" {
		out["thoughtSignature"] = p.ThoughtSignature
	}
	if p.InlineData != nil {
		out["inlineData"] = map[string]any{
			"mimeType": p.InlineData.MimeType,
			"data":     p.InlineData.Data,
		}
	}
	if p.FileData != nil {
		out["fileData"] = map[string]any{
			"mimeType": p.FileData.MimeType,
			"fileUri":  p.FileData.FileURI,
		}
	}
	if p.FunctionCall != nil {
		fc := map[string]any{"name": p.FunctionCall.Name}
		if p.FunctionCall.ID != "" {
			fc["id"] = p.FunctionCall.ID
		}
		if len(p.FunctionCall.Args) > 0 {
			fc["args"] = json.RawMessage(p.FunctionCall.Args)
		}
		if len(p.FunctionCall.PartialArgs) > 0 {
			pa := make([]any, 0, len(p.FunctionCall.PartialArgs))
			for _, a := range p.FunctionCall.PartialArgs {
				pa = append(pa, partialArgToMap(a))
			}
			fc["partialArgs"] = pa
		}
		if p.FunctionCall.WillContinue != nil {
			fc["willContinue"] = *p.FunctionCall.WillContinue
		}
		out["functionCall"] = fc
	}
	if p.FunctionResponse != nil {
		fr := map[string]any{
			"name":     p.FunctionResponse.Name,
			"response": json.RawMessage(p.FunctionResponse.Response),
		}
		if p.FunctionResponse.ID != "" {
			fr["id"] = p.FunctionResponse.ID
		}
		if len(p.FunctionResponse.Parts) > 0 {
			ps := make([]map[string]any, 0, len(p.FunctionResponse.Parts))
			for _, sub := range p.FunctionResponse.Parts {
				ps = append(ps, partToMap(sub))
			}
			fr["parts"] = ps
		}
		out["functionResponse"] = fr
	}
	if p.ToolCall != nil {
		tc := map[string]any{
			"toolType": p.ToolCall.ToolType,
			"id":       p.ToolCall.ID,
		}
		if len(p.ToolCall.Args) > 0 {
			tc["args"] = json.RawMessage(p.ToolCall.Args)
		}
		out["toolCall"] = tc
	}
	if p.ToolResponse != nil {
		tr := map[string]any{
			"toolType": p.ToolResponse.ToolType,
			"id":       p.ToolResponse.ID,
		}
		if len(p.ToolResponse.Response) > 0 {
			tr["response"] = json.RawMessage(p.ToolResponse.Response)
		}
		out["toolResponse"] = tr
	}
	if p.ExecutableCode != nil {
		out["executableCode"] = map[string]any{
			"language": p.ExecutableCode.Language,
			"code":     p.ExecutableCode.Code,
		}
	}
	if p.CodeExecutionResult != nil {
		out["codeExecutionResult"] = map[string]any{
			"outcome": p.CodeExecutionResult.Outcome,
			"output":  p.CodeExecutionResult.Output,
		}
	}
	return out
}

func partialArgToMap(a internal.APIPartialArg) map[string]any {
	out := map[string]any{"jsonPath": a.JSONPath}
	if a.WillContinue {
		out["willContinue"] = true
	}
	if a.StringValue != nil {
		out["stringValue"] = *a.StringValue
	}
	if a.NumberValue != nil {
		out["numberValue"] = *a.NumberValue
	}
	if a.BoolValue != nil {
		out["boolValue"] = *a.BoolValue
	}
	if a.NullValue != nil {
		out["nullValue"] = *a.NullValue
	}
	return out
}

// DoGenerate performs a non-streaming generation call.
func (m *googleLanguageModel) DoGenerate(ctx context.Context, opts GenerateOptions) (*GenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}

	args, err := m.getArgs(opts)
	if err != nil {
		return nil, err
	}

	contents, system, convertWarnings, err := ConvertPrompt(m.modelID, opts)
	if err != nil {
		return nil, err
	}
	args.Warnings = append(args.Warnings, convertWarnings...)

	body, extraHeaders, buildWarnings, err := m.buildChatRequest(args, opts, contents, system)
	if err != nil {
		return nil, err
	}
	args.Warnings = append(args.Warnings, buildWarnings...)

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	perCall := cloneHeader(opts.Headers)
	for k, values := range extraHeaders {
		perCall.Del(k)
		for _, v := range values {
			perCall.Add(k, v)
		}
	}

	resp, err := m.provider.executeJSON(ctx, "/"+getModelPath(m.modelID)+":generateContent", bodyBytes, perCall)
	if err != nil {
		return nil, err
	}

	parsed, parseWarnings, err := m.parseGenerateResponse(resp.Body, bodyBytes, resp.Headers)
	if err != nil {
		return nil, err
	}
	parsed.Warnings = append(args.Warnings, append(parseWarnings, parsed.Warnings...)...)
	parsed.Request = RequestMetadata{Body: append([]byte(nil), bodyBytes...)}
	respID := ""
	if gm, ok := parsed.ProviderMetadata["google"].(map[string]any); ok {
		if id, ok := gm["responseId"].(string); ok {
			respID = id
		}
	}
	parsed.Response = ResponseMetadata{
		ID:      respID,
		ModelID: m.modelID,
		Headers: cloneHeader(resp.Headers),
		Body:    append([]byte(nil), resp.Body...),
	}
	return parsed, nil
}

// parseGenerateResponse decodes a Google :generateContent response into a
// GenerateResult. The map-style API content envelopes are decoded via the
// internal typed structs.
func (m *googleLanguageModel) parseGenerateResponse(body []byte, requestBody []byte, headers http.Header) (*GenerateResult, []Warning, error) {
	var resp internal.APIGenerateContentResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, nil, InvalidResponseDataError{Message: err.Error(), Data: string(body)}
	}

	var warnings []Warning

	// Provider metadata at top level.
	pm := ProviderMetadata{}
	gm := map[string]any{}
	// Always set pm["google"] (even if currently empty) so subsequent writes
	// to gm (grounding, sources, unknownParts, etc.) are visible to callers.
	pm["google"] = gm
	if resp.ModelVersion != "" {
		gm["modelVersion"] = resp.ModelVersion
	}
	if resp.ResponseID != "" {
		gm["responseId"] = resp.ResponseID
	}

	// Pick the first candidate (matches upstream behavior).
	if len(resp.Candidates) == 0 {
		return &GenerateResult{
			ProviderMetadata: pm,
			Warnings:         warnings,
			Request:          RequestMetadata{Body: append([]byte(nil), requestBody...)},
			Response:         responseMetadata(headers, append([]byte(nil), body...), "", m.modelID),
		}, warnings, nil
	}
	cand := resp.Candidates[0]

	// Walk parts.
	contents, unknownParts, partWarnings := partsToContents(cand.Content.Parts)
	warnings = append(warnings, partWarnings...)

	// Finish reason.
	finish := mapGoogleFinishReason(cand.FinishReason, hasClientSideToolCalls(contents))

	// Provider metadata: grounding, safety, promptFeedback, urlContext,
	// citation, finishMessage, serviceTier, tokensDetails.
	if cand.GroundingMetadata != nil {
		gm["groundingMetadata"] = groundingMetadataToPublic(cand.GroundingMetadata)
		// Also surface extracted sources (de-duplicated by URL).
		sources := extractGroundingSources(cand.GroundingMetadata)
		if len(sources) > 0 {
			gm["sources"] = sourcesToPublic(sources)
		}
	}
	if cand.UrlContextMetadata != nil {
		gm["urlContextMetadata"] = urlContextMetadataToPublic(cand.UrlContextMetadata)
	}
	if cand.CitationMetadata != nil && len(cand.CitationMetadata.CitationSources) > 0 {
		gm["citationMetadata"] = citationMetadataToPublic(cand.CitationMetadata)
	}
	if len(cand.SafetyRatings) > 0 {
		gm["safetyRatings"] = safetyRatingsToPublic(cand.SafetyRatings)
	}
	if resp.PromptFeedback != nil {
		gm["promptFeedback"] = promptFeedbackToPublic(resp.PromptFeedback)
	}
	if len(unknownParts) > 0 {
		gm["unknownParts"] = unknownParts
		if !hasUnknownPartsWarning(warnings) {
			warnings = append(warnings, Warning{
				Type:    "other",
				Feature: "unknown-part",
				Message: "unrecognized part keys; preserved in providerMetadata",
			})
		}
	}

	// Usage.
	usage := convertGoogleUsage(resp.UsageMetadata)
	if resp.UsageMetadata != nil {
		if len(resp.UsageMetadata.PromptTokensDetails) > 0 {
			gm["promptTokensDetails"] = modalityDetailsToPublic(resp.UsageMetadata.PromptTokensDetails)
		}
		if len(resp.UsageMetadata.CandidatesTokensDetails) > 0 {
			gm["candidatesTokensDetails"] = modalityDetailsToPublic(resp.UsageMetadata.CandidatesTokensDetails)
		}
		if resp.UsageMetadata.TrafficType != "" {
			gm["serviceTier"] = resp.UsageMetadata.TrafficType
		}
	}

	result := &GenerateResult{
		Content:          contents,
		FinishReason:     finish,
		Usage:            usage,
		ProviderMetadata: pm,
	}
	return result, warnings, nil
}

func hasClientSideToolCalls(contents []Content) bool {
	for _, c := range contents {
		if tc, ok := c.(ToolCallContent); ok {
			// Server-side tool calls carry a serverToolType in
			// ProviderMetadata; client-side calls do not.
			if !isServerToolCallContent(tc) {
				return true
			}
		}
	}
	return false
}

func isServerToolCallContent(tc ToolCallContent) bool {
	if pm, ok := tc.ProviderMetadata["google"].(map[string]any); ok {
		if _, ok := pm["serverToolType"]; ok {
			return true
		}
	}
	return false
}

func hasUnknownPartsWarning(warnings []Warning) bool {
	for _, w := range warnings {
		if w.Feature == "unknown-part" {
			return true
		}
	}
	return false
}

// partsToContents walks a slice of wire parts and converts each into a typed
// google.Content. Unknown parts are returned via unknownParts for forward
// compat.
func partsToContents(parts []internal.APIPart) ([]Content, []map[string]any, []Warning) {
	var out []Content
	var unknowns []map[string]any
	var warnings []Warning
	for i, p := range parts {
		c, unknown, w := partToContent(p)
		if c != nil {
			out = append(out, c)
		}
		if unknown != nil {
			unknowns = append(unknowns, unknown)
		}
		warnings = append(warnings, w...)
		_ = i
	}
	return out, unknowns, warnings
}

func partToContent(p internal.APIPart) (Content, map[string]any, []Warning) {
	// Server tool call.
	if p.ToolCall != nil {
		pm := ProviderMetadata{
			"google": map[string]any{
				"serverToolType": p.ToolCall.ToolType,
			},
		}
		if p.ThoughtSignature != "" {
			pm["google"].(map[string]any)["thoughtSignature"] = p.ThoughtSignature
		}
		input := p.ToolCall.Args
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		return ToolCallContent{
			ToolCallID:       p.ToolCall.ID,
			ToolName:         p.ToolCall.ToolType,
			Input:            input,
			ProviderMetadata: pm,
		}, nil, nil
	}
	// Function call (client-side).
	if p.FunctionCall != nil {
		pm := ProviderMetadata{}
		if p.ThoughtSignature != "" {
			pm["google"] = map[string]any{"thoughtSignature": p.ThoughtSignature}
		}
		input := p.FunctionCall.Args
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		return ToolCallContent{
			ToolCallID:       p.FunctionCall.ID,
			ToolName:         p.FunctionCall.Name,
			Input:            input,
			ProviderMetadata: pm,
		}, nil, nil
	}
	// Reasoning (text with thought: true).
	if p.Text != "" && p.Thought != nil && *p.Thought {
		rc := ReasoningContent{Text: p.Text}
		if p.ThoughtSignature != "" {
			rc.ProviderOptions = ProviderOptions{"google": map[string]any{"thoughtSignature": p.ThoughtSignature}}
			rc.Signature = p.ThoughtSignature
		}
		return rc, nil, nil
	}
	// Text.
	if p.Text != "" {
		tc := TextContent{Text: p.Text}
		if p.ThoughtSignature != "" {
			tc.ProviderOptions = ProviderOptions{"google": map[string]any{"thoughtSignature": p.ThoughtSignature}}
		}
		return tc, nil, nil
	}
	// Inline data.
	if p.InlineData != nil {
		// Surface as ImageContent (assistant output may be image).
		return ImageContent{
			Source: ImageSource{
				Type:      "data",
				MediaType: p.InlineData.MimeType,
				Data:      p.InlineData.Data,
			},
		}, nil, nil
	}
	// File data.
	if p.FileData != nil {
		return ImageContent{
			Source: ImageSource{
				Type:      "url",
				MediaType: p.FileData.MimeType,
				URL:       p.FileData.FileURI,
			},
		}, nil, nil
	}
	// Executable code.
	if p.ExecutableCode != nil {
		return ExecutableCodeContent{
			Language: p.ExecutableCode.Language,
			Code:     p.ExecutableCode.Code,
		}, nil, nil
	}
	// Code execution result.
	if p.CodeExecutionResult != nil {
		return CodeExecutionResultContent{
			Outcome: p.CodeExecutionResult.Outcome,
			Output:  p.CodeExecutionResult.Output,
		}, nil, nil
	}
	// Unknown / forward-compat.
	return nil, partToMap(p), nil
}

// extractGroundingSources converts groundingMetadata into a flat list of
// Source entries (de-duplicated by URL).
func extractGroundingSources(gm *internal.APIGroundingMetadata) []Source {
	if gm == nil {
		return nil
	}
	var out []Source
	seen := map[string]struct{}{}
	for _, ch := range gm.GroundingChunks {
		if ch.Web != nil && ch.Web.URI != "" {
			if _, ok := seen[ch.Web.URI]; !ok {
				seen[ch.Web.URI] = struct{}{}
				out = append(out, Source{Type: "url", URL: ch.Web.URI, Title: ch.Web.Title})
			}
		}
		if ch.Image != nil && ch.Image.SourceURI != "" {
			if _, ok := seen[ch.Image.SourceURI]; !ok {
				seen[ch.Image.SourceURI] = struct{}{}
				out = append(out, Source{Type: "url", URL: ch.Image.SourceURI, Title: ch.Image.Title})
			}
		}
		if ch.RetrievedContext != nil && ch.RetrievedContext.URI != "" {
			if _, ok := seen[ch.RetrievedContext.URI]; !ok {
				seen[ch.RetrievedContext.URI] = struct{}{}
				src := Source{Type: "url", URL: ch.RetrievedContext.URI, Title: ch.RetrievedContext.Title}
				if strings.HasPrefix(src.URL, "gs://") {
					src.Type = "document"
					mt, fn := extensionToMediaType(src.URL)
					src.MediaType = mt
					src.Filename = fn
				}
				out = append(out, src)
			}
		}
		if ch.Maps != nil && ch.Maps.URI != "" {
			if _, ok := seen[ch.Maps.URI]; !ok {
				seen[ch.Maps.URI] = struct{}{}
				out = append(out, Source{Type: "url", URL: ch.Maps.URI, Title: ch.Maps.Title})
			}
		}
	}
	return out
}

func extensionToMediaType(uri string) (string, string) {
	idx := strings.LastIndex(uri, ".")
	if idx < 0 || idx < strings.LastIndex(uri, "/") {
		// No extension, or the dot is in the path prefix (e.g. gs://bucket.name/foo).
		base := uri[strings.LastIndex(uri, "/")+1:]
		return "", base
	}
	ext := strings.ToLower(uri[idx:])
	base := uri[strings.LastIndex(uri, "/")+1:]
	switch ext {
	case ".pdf":
		return "application/pdf", base
	case ".txt":
		return "text/plain", base
	case ".docx":
		return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", base
	case ".doc":
		return "application/msword", base
	case ".md", ".markdown":
		return "text/markdown", base
	}
	return "", base
}

func sourcesToPublic(srcs []Source) []map[string]any {
	out := make([]map[string]any, 0, len(srcs))
	for _, s := range srcs {
		entry := map[string]any{
			"type": s.Type,
		}
		if s.URL != "" {
			entry["url"] = s.URL
		}
		if s.Title != "" {
			entry["title"] = s.Title
		}
		if s.MediaType != "" {
			entry["mediaType"] = s.MediaType
		}
		if s.Filename != "" {
			entry["filename"] = s.Filename
		}
		out = append(out, entry)
	}
	return out
}

func groundingMetadataToPublic(gm *internal.APIGroundingMetadata) map[string]any {
	out := map[string]any{}
	if len(gm.WebSearchQueries) > 0 {
		out["webSearchQueries"] = append([]string(nil), gm.WebSearchQueries...)
	}
	if len(gm.ImageSearchQueries) > 0 {
		out["imageSearchQueries"] = append([]string(nil), gm.ImageSearchQueries...)
	}
	if len(gm.RetrievalQueries) > 0 {
		out["retrievalQueries"] = append([]string(nil), gm.RetrievalQueries...)
	}
	if gm.SearchEntryPoint != nil {
		out["searchEntryPoint"] = map[string]any{"renderedContent": gm.SearchEntryPoint.RenderedContent}
	}
	return out
}

func urlContextMetadataToPublic(um *internal.APIURLContextMetadata) map[string]any {
	if um == nil {
		return nil
	}
	return map[string]any{"urlMetadata": append([]string(nil), um.URLMetadata...)}
}

func citationMetadataToPublic(cm *internal.APICitationMetadata) map[string]any {
	if cm == nil {
		return nil
	}
	sources := make([]map[string]any, 0, len(cm.CitationSources))
	for _, s := range cm.CitationSources {
		entry := map[string]any{}
		if s.StartIndex != nil {
			entry["startIndex"] = *s.StartIndex
		}
		if s.EndIndex != nil {
			entry["endIndex"] = *s.EndIndex
		}
		if s.URI != "" {
			entry["uri"] = s.URI
		}
		if s.Title != "" {
			entry["title"] = s.Title
		}
		if s.License != "" {
			entry["license"] = s.License
		}
		sources = append(sources, entry)
	}
	return map[string]any{"citationSources": sources}
}

func safetyRatingsToPublic(rs []internal.APISafetyRating) []map[string]any {
	out := make([]map[string]any, 0, len(rs))
	for _, r := range rs {
		out = append(out, map[string]any{
			"category":    r.Category,
			"probability": r.Probability,
		})
	}
	return out
}

func promptFeedbackToPublic(pf *internal.APIPromptFeedback) map[string]any {
	if pf == nil {
		return nil
	}
	out := map[string]any{}
	if pf.BlockReason != "" {
		out["blockReason"] = pf.BlockReason
	}
	if len(pf.SafetyRatings) > 0 {
		out["safetyRatings"] = safetyRatingsToPublic(pf.SafetyRatings)
	}
	return out
}

func modalityDetailsToPublic(d []internal.APIModalityTokenCount) []map[string]any {
	out := make([]map[string]any, 0, len(d))
	for _, m := range d {
		out = append(out, map[string]any{
			"modality":   m.Modality,
			"tokenCount": m.TokenCount,
		})
	}
	return out
}
