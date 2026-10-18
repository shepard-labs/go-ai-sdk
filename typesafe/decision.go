package typesafe

import (
	"bytes"
	"encoding/json"
	"net/http"
)

// QuestionType identifies a supported decision question type.
type QuestionType string

const (
	QuestionTypeChoice  QuestionType = "choice"
	QuestionTypeScore   QuestionType = "score"
	QuestionTypeBoolean QuestionType = "boolean"
	QuestionTypeBox     QuestionType = "box"
)

// Input is structured question content: a string, arbitrary JSON, or null.
// The zero value marshals as null.
type Input struct {
	raw json.RawMessage
	err error
}

// StringInput returns an Input holding a JSON string.
func StringInput(s string) Input {
	b, _ := json.Marshal(s)
	return Input{raw: b}
}

// JSONInput returns an Input holding v marshaled to JSON. A marshal failure
// is surfaced as an *InvalidArgumentError when the request is built.
func JSONInput(v any) Input {
	b, err := json.Marshal(v)
	return Input{raw: b, err: err}
}

// RawInput returns an Input holding raw JSON. The bytes are cloned so
// later caller mutation cannot change a queued request.
func RawInput(raw json.RawMessage) Input {
	return Input{raw: bytes.Clone(raw)}
}

// NullInput returns an Input holding JSON null.
func NullInput() Input {
	return Input{raw: json.RawMessage("null")}
}

func (i Input) MarshalJSON() ([]byte, error) {
	if i.err != nil {
		return nil, i.err
	}
	if len(i.raw) == 0 {
		return []byte("null"), nil
	}
	return i.raw, nil
}

func (i *Input) UnmarshalJSON(b []byte) error {
	i.raw = bytes.Clone(b)
	i.err = nil
	return nil
}

// Question is a decision question: ChoiceQuestion, ScoreQuestion,
// BooleanQuestion, or BoxQuestion (values or pointers).
type Question interface {
	json.Marshaler
	sealedQuestion()
}

// ChoiceQuestion picks one named option. Criteria maps option names to
// optional structured descriptions; at most 255 options.
type ChoiceQuestion struct {
	Instructions *Input
	Criteria     map[string]*Input
}

func (q ChoiceQuestion) sealedQuestion() {}

func (q ChoiceQuestion) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         string            `json:"type"`
		Instructions *Input            `json:"instructions,omitempty"`
		Criteria     map[string]*Input `json:"criteria"`
	}{Type: "choice", Instructions: q.Instructions, Criteria: q.Criteria})
}

// ScoreQuestion rates state on ordered levels; at most 10 levels.
type ScoreQuestion struct {
	Instructions *Input
	Criteria     []*Input
}

func (q ScoreQuestion) sealedQuestion() {}

func (q ScoreQuestion) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         string   `json:"type"`
		Instructions *Input   `json:"instructions,omitempty"`
		Criteria     []*Input `json:"criteria"`
	}{Type: "score", Instructions: q.Instructions, Criteria: q.Criteria})
}

// BooleanCriteria describes the true and false outcomes of a BooleanQuestion.
type BooleanCriteria struct {
	True  *Input `json:"true,omitempty"`
	False *Input `json:"false,omitempty"`
}

// BooleanQuestion asks a yes/no question. It is sent to the API as type
// "noul" and answered with P(true).
type BooleanQuestion struct {
	Instructions *Input
	Criteria     *BooleanCriteria
}

func (q BooleanQuestion) sealedQuestion() {}

func (q BooleanQuestion) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         string           `json:"type"`
		Instructions *Input           `json:"instructions,omitempty"`
		Criteria     *BooleanCriteria `json:"criteria,omitempty"`
	}{Type: "noul", Instructions: q.Instructions, Criteria: q.Criteria})
}

// BoxQuestion locates a region in an image. It is sent to the API as exactly
// {"type":"box","instructions":...} and enables the images danger-zone
// header.
type BoxQuestion struct {
	Instructions Input
}

func (q BoxQuestion) sealedQuestion() {}

func (q BoxQuestion) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         string `json:"type"`
		Instructions Input  `json:"instructions"`
	}{Type: "box", Instructions: q.Instructions})
}

// StatePart is one ordered part of the shared decision state:
// TextStatePart, JSONStatePart, or FileStatePart (values or pointers).
type StatePart interface {
	sealedStatePart()
}

// TextStatePart is sent verbatim.
type TextStatePart struct {
	Text string
}

func (TextStatePart) sealedStatePart() {}

// JSONStatePart holds raw JSON. A state consisting of exactly one JSON part
// is sent as that JSON value; otherwise JSON parts are compact-encoded and
// joined with text parts using "\n".
type JSONStatePart struct {
	Value json.RawMessage
}

func (JSONStatePart) sealedStatePart() {}

// FileStatePart is an inline PNG or JPEG image. Reference names the asset in
// state and questions; when empty, one is generated. InsertReference
// controls whether the reference text is placed in state at the image's
// position: nil inserts it unless the reference already occurs in state text
// or JSON string values, a pointer to true forces insertion, and a pointer
// to false suppresses it (and requires an explicit Reference).
type FileStatePart struct {
	Filename        string
	MediaType       string
	Data            []byte
	Reference       string
	InsertReference *bool
}

func (FileStatePart) sealedStatePart() {}

// WarningTypeUnsupported marks provider options the API does not support.
const WarningTypeUnsupported = "unsupported"

// Warning reports a non-fatal issue with a decision call.
type Warning struct {
	Type    string
	Feature string
}

// Answer is a decision answer: ChoiceAnswer, ScoreAnswer, BooleanAnswer, or
// BoxAnswer.
type Answer interface {
	sealedAnswer()
}

// ChoiceAnswer is the chosen option and the full probability distribution.
type ChoiceAnswer struct {
	Choice        string
	Probabilities map[string]float64
}

func (ChoiceAnswer) sealedAnswer() {}

// ScoreAnswer is the fractional score and the per-level distribution.
type ScoreAnswer struct {
	Score         float64
	Probabilities map[string]float64
}

func (ScoreAnswer) sealedAnswer() {}

// BooleanAnswer is P(true).
type BooleanAnswer struct {
	Probability float64
}

func (BooleanAnswer) sealedAnswer() {}

// BoxAnswer locates a region in the image.
type BoxAnswer struct {
	Box Box
}

func (BoxAnswer) sealedAnswer() {}

// Box is a normalized rectangle measured from the image top-left.
type Box struct {
	Top    float64
	Bottom float64
	Left   float64
	Right  float64
}

// DecisionCallOptions configures one DoDecide call.
type DecisionCallOptions struct {
	State     []StatePart
	Questions map[string]Question
	// Headers override provider headers for this call.
	Headers http.Header
	// ProviderOptions carries the "typesafe" namespace; the only supported
	// key is longContext (bool). Unknown typesafe.* keys produce warnings.
	ProviderOptions map[string]map[string]any
}

// Usage carries token counts reported by the API. Absent values stay nil.
type Usage struct {
	InputTokens  *int64
	OutputTokens *int64
}

// Rounding declares the API's display rounding; values are not altered.
type Rounding struct {
	ProbabilityDecimals int
	ScoreDecimals       int
}

// Response carries response metadata from the API.
type Response struct {
	ModelID string
	Headers http.Header
	Body    json.RawMessage
}

// DecisionResult is the outcome of one DoDecide call.
type DecisionResult struct {
	Answers  map[string]Answer
	Usage    *Usage
	Rounding Rounding
	Warnings []Warning
	// ProviderMetadata["typesafe"]["confidence"] maps question IDs to the
	// API's per-answer confidence (boolean answers have no entry).
	ProviderMetadata map[string]map[string]any
	Response         Response
}
