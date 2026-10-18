package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
)

// DecisionModel answers structured questions against shared state.
type DecisionModel interface {
	ModelID() string
	// Provider returns "typesafe.decision".
	Provider() string
	SupportedQuestionTypes() []QuestionType
	DoDecide(ctx context.Context, opts DecisionCallOptions) (*DecisionResult, error)
}

type decisionModel struct {
	provider *typeSafeProvider
	modelID  string
}

func (m *decisionModel) ModelID() string  { return m.modelID }
func (m *decisionModel) Provider() string { return "typesafe.decision" }
func (m *decisionModel) SupportedQuestionTypes() []QuestionType {
	return []QuestionType{QuestionTypeChoice, QuestionTypeScore, QuestionTypeBoolean, QuestionTypeBox}
}

func (m *decisionModel) DoDecide(ctx context.Context, opts DecisionCallOptions) (*DecisionResult, error) {
	p := m.provider

	hasBox := false
	for id, q := range opts.Questions {
		if isNilQuestion(q) {
			return nil, &InvalidArgumentError{Argument: "questions." + id, Message: "TypeSafe questions must be non-nil."}
		}
		if c, ok := asChoice(q); ok && len(c.Criteria) > 255 {
			return nil, &InvalidArgumentError{Argument: "questions." + id + ".criteria", Message: "TypeSafe Choice questions support at most 255 options."}
		}
		if s, ok := asScore(q); ok && len(s.Criteria) > 10 {
			return nil, &InvalidArgumentError{Argument: "questions." + id + ".criteria", Message: "TypeSafe Score questions support at most 10 levels."}
		}
		if _, ok := asBox(q); ok {
			hasBox = true
		}
	}

	longContext := false
	var warnings []Warning
	if ts, ok := opts.ProviderOptions["typesafe"]; ok {
		for _, key := range slices.Sorted(maps.Keys(ts)) {
			if key == "longContext" {
				b, ok := ts[key].(bool)
				if !ok {
					return nil, &InvalidArgumentError{Argument: "providerOptions.typesafe.longContext", Message: "longContext must be a boolean."}
				}
				longContext = b
				continue
			}
			warnings = append(warnings, Warning{Type: WarningTypeUnsupported, Feature: "providerOptions.typesafe." + key})
		}
	}

	state, assets, images, err := p.buildState(opts.State)
	if err != nil {
		return nil, err
	}

	reqBody, err := json.Marshal(wireRequest{
		Model:     m.modelID,
		State:     state,
		Assets:    assets,
		Questions: opts.Questions,
	})
	if err != nil {
		return nil, &InvalidArgumentError{Argument: "questions", Message: err.Error()}
	}

	key := p.apiKey()
	if key == "" {
		return nil, ErrMissingAPIKey
	}
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+key)
	for k, values := range p.headers {
		headers.Del(k)
		for _, v := range values {
			headers.Add(k, v)
		}
	}
	headers.Set("User-Agent", "ai-sdk-go/typesafe/"+Version)
	if len(images) > 0 || hasBox {
		headers.Set("TypeSafe-Danger-Zone-Images", "true")
	}
	if longContext {
		headers.Set("TypeSafe-Danger-Zone-Long-Context", "true")
	}
	for k, values := range opts.Headers {
		headers.Del(k)
		for _, v := range values {
			headers.Add(k, v)
		}
	}

	result, err := p.executeDecision(ctx, reqBody, images, headers)
	if err != nil {
		return nil, err
	}

	var wire wireResponse
	if err := json.Unmarshal(result.body, &wire); err != nil {
		return nil, &InvalidResponseDataError{Message: "failed to decode TypeSafe response: " + err.Error(), Data: string(result.body)}
	}
	if wire.Answers == nil {
		return nil, &InvalidResponseDataError{Message: "TypeSafe response is missing answers", Data: string(result.body)}
	}
	answers := make(map[string]Answer, len(wire.Answers))
	confidence := map[string]float64{}
	for id, raw := range wire.Answers {
		decoded, err := decodeAnswer(raw)
		if err != nil {
			return nil, err
		}
		answers[id] = decoded.answer
		if decoded.confidence != nil {
			confidence[id] = *decoded.confidence
		}
	}
	modelID := m.modelID
	if wire.Model != nil {
		modelID = *wire.Model
	}
	var usage *Usage
	if wire.Usage != nil {
		usage = &Usage{InputTokens: wire.Usage.InputTokens, OutputTokens: wire.Usage.OutputTokens}
	}
	return &DecisionResult{
		Answers:  answers,
		Usage:    usage,
		Rounding: Rounding{ProbabilityDecimals: 2, ScoreDecimals: 2},
		Warnings: warnings,
		ProviderMetadata: map[string]map[string]any{
			"typesafe": {"confidence": confidence},
		},
		Response: Response{ModelID: modelID, Headers: result.headers, Body: json.RawMessage(bytes.Clone(result.body))},
	}, nil
}

// stateOut is one serialized state part: either text or raw JSON.
type stateOut struct {
	text   string
	raw    json.RawMessage
	isJSON bool
}

type imageAsset struct {
	name      string
	filename  string
	mediaType string
	data      []byte
}

// buildState validates state parts, resolves image references, and returns
// the wire state value, the assets map, and the multipart image payloads.
func (p *typeSafeProvider) buildState(state []StatePart) (json.RawMessage, map[string]wireAsset, []imageAsset, error) {
	var stateStrings []string
	for _, raw := range state {
		switch part := derefStatePart(raw).(type) {
		case TextStatePart:
			stateStrings = append(stateStrings, part.Text)
		case JSONStatePart:
			var v any
			if err := json.Unmarshal(part.Value, &v); err != nil {
				return nil, nil, nil, &InvalidArgumentError{Argument: "state", Message: "invalid JSON state part: " + err.Error()}
			}
			stateStrings = append(stateStrings, jsonStrings(v)...)
		}
	}

	var assets map[string]wireAsset
	var images []imageAsset
	outParts := make([]stateOut, 0, len(state))
	for i, raw := range state {
		part := derefStatePart(raw)
		file, ok := part.(FileStatePart)
		if !ok {
			switch p := part.(type) {
			case TextStatePart:
				outParts = append(outParts, stateOut{text: p.Text})
			case JSONStatePart:
				outParts = append(outParts, stateOut{raw: p.Value, isJSON: true})
			default:
				return nil, nil, nil, &InvalidArgumentError{Argument: "state", Message: fmt.Sprintf("unsupported state part type %T", raw)}
			}
			continue
		}
		if file.MediaType != "image/png" && file.MediaType != "image/jpeg" {
			return nil, nil, nil, &UnsupportedFunctionalityError{Functionality: "TypeSafe AI decision image media type: " + file.MediaType}
		}
		if len(file.Data) == 0 {
			return nil, nil, nil, &UnsupportedFunctionalityError{Functionality: "TypeSafe AI decision file input without inline data (" + file.MediaType + ")"}
		}
		reference := file.Reference
		if reference != "" && strings.TrimSpace(reference) == "" {
			return nil, nil, nil, &InvalidArgumentError{Argument: "providerOptions.typesafe.reference", Message: "TypeSafe image references must be nonempty."}
		}
		if file.InsertReference != nil && !*file.InsertReference && reference == "" {
			return nil, nil, nil, &InvalidArgumentError{Argument: "providerOptions.typesafe.reference", Message: "TypeSafe images with insertReference: false need an explicit reference."}
		}
		if strings.ContainsAny(file.Filename, "\r\n") {
			return nil, nil, nil, &InvalidArgumentError{Argument: "state.filename", Message: "TypeSafe image filenames must not contain CR or LF."}
		}
		if reference == "" {
			reference = p.generateID()
		}
		insert := true
		if file.InsertReference != nil {
			insert = *file.InsertReference
		} else {
			insert = !slices.ContainsFunc(stateStrings, func(s string) bool { return strings.Contains(s, reference) })
		}
		if assets != nil {
			if _, dup := assets[reference]; dup {
				return nil, nil, nil, &InvalidArgumentError{Argument: "providerOptions.typesafe.reference", Message: "Duplicate TypeSafe image reference: " + reference}
			}
		} else {
			assets = map[string]wireAsset{}
		}
		partName := fmt.Sprintf("image-%d", i)
		assets[reference] = wireAsset{Type: "image", Part: partName}
		filename := file.Filename
		if filename == "" {
			ext := "jpeg"
			if file.MediaType == "image/png" {
				ext = "png"
			}
			filename = partName + "." + ext
		}
		images = append(images, imageAsset{name: partName, filename: filename, mediaType: file.MediaType, data: file.Data})
		if insert {
			outParts = append(outParts, stateOut{text: reference})
		}
	}

	if len(outParts) == 1 && outParts[0].isJSON {
		return outParts[0].raw, assets, images, nil
	}
	var sb strings.Builder
	for i, part := range outParts {
		if i > 0 {
			sb.WriteByte('\n')
		}
		if part.isJSON {
			var buf bytes.Buffer
			if err := json.Compact(&buf, part.raw); err != nil {
				return nil, nil, nil, &InvalidArgumentError{Argument: "state", Message: "invalid JSON state part: " + err.Error()}
			}
			sb.Write(buf.Bytes())
		} else {
			sb.WriteString(part.text)
		}
	}
	stateJSON, err := json.Marshal(sb.String())
	if err != nil {
		return nil, nil, nil, &InvalidArgumentError{Argument: "state", Message: err.Error()}
	}
	return stateJSON, assets, images, nil
}

// jsonStrings returns the string values of v at any depth; object keys do
// not count.
func jsonStrings(v any) []string {
	switch t := v.(type) {
	case string:
		return []string{t}
	case []any:
		var out []string
		for _, e := range t {
			out = append(out, jsonStrings(e)...)
		}
		return out
	case map[string]any:
		out := make([]string, 0, len(t))
		for _, e := range t {
			out = append(out, jsonStrings(e)...)
		}
		return out
	}
	return nil
}

// isNilQuestion reports whether q is a nil pointer inside a non-nil
// interface, e.g. (*ChoiceQuestion)(nil).
func isNilQuestion(q Question) bool {
	switch v := q.(type) {
	case nil:
		return true
	case *ChoiceQuestion:
		return v == nil
	case *ScoreQuestion:
		return v == nil
	case *BooleanQuestion:
		return v == nil
	case *BoxQuestion:
		return v == nil
	}
	return false
}

func derefStatePart(p StatePart) StatePart {
	switch v := p.(type) {
	case *TextStatePart:
		if v != nil {
			return *v
		}
	case *JSONStatePart:
		if v != nil {
			return *v
		}
	case *FileStatePart:
		if v != nil {
			return *v
		}
	}
	return p
}

func asChoice(q Question) (ChoiceQuestion, bool) {
	switch v := q.(type) {
	case ChoiceQuestion:
		return v, true
	case *ChoiceQuestion:
		if v != nil {
			return *v, true
		}
	}
	return ChoiceQuestion{}, false
}

func asScore(q Question) (ScoreQuestion, bool) {
	switch v := q.(type) {
	case ScoreQuestion:
		return v, true
	case *ScoreQuestion:
		if v != nil {
			return *v, true
		}
	}
	return ScoreQuestion{}, false
}

func asBox(q Question) (BoxQuestion, bool) {
	switch v := q.(type) {
	case BoxQuestion:
		return v, true
	case *BoxQuestion:
		if v != nil {
			return *v, true
		}
	}
	return BoxQuestion{}, false
}
