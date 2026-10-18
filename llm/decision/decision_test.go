package decision

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestInputMarshaling(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input Input
		want  string
	}{
		{"zero value is null", Input{}, "null"},
		{"null", NullInput(), "null"},
		{"string", StringInput("Pick a department"), `"Pick a department"`},
		{"empty string", StringInput(""), `""`},
		{"json object", JSONInput(map[string]any{"meaning": "Explicit request"}), `{"meaning":"Explicit request"}`},
		{"json array", JSONInput([]any{"a", 1, nil}), `["a",1,null]`},
		{"raw is compacted", RawInput(json.RawMessage(` {"nested":true} `)), `{"nested":true}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.input)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != tc.want {
				t.Fatalf("marshal = %s, want %s", b, tc.want)
			}
			var round Input
			if err := json.Unmarshal(b, &round); err != nil {
				t.Fatal(err)
			}
			rb, err := json.Marshal(round)
			if err != nil {
				t.Fatal(err)
			}
			if string(rb) != tc.want {
				t.Fatalf("round trip = %s, want %s", rb, tc.want)
			}
		})
	}
}

func TestJSONInputMarshalErrorSurfaces(t *testing.T) {
	input := JSONInput(func() {})
	if _, err := json.Marshal(input); err == nil {
		t.Fatal("expected marshal error")
	}
}

func TestInputPointerSemantics(t *testing.T) {
	// *Input is nullable in questions; a nil pointer marshals as null and a
	// set pointer marshals its value.
	type holder struct {
		Instructions *Input `json:"instructions,omitempty"`
	}
	set := StringInput("hi")
	b, err := json.Marshal(holder{Instructions: &set})
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{"instructions":"hi"}` {
		t.Fatalf("marshal = %s", b)
	}
	b, err = json.Marshal(holder{})
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{}` {
		t.Fatalf("marshal = %s", b)
	}
}

func TestSealedInterfaces(t *testing.T) {
	// All concrete types satisfy their sealed interfaces, as values and as
	// pointers.
	questions := []Question{
		ChoiceQuestion{}, &ChoiceQuestion{},
		ScoreQuestion{}, &ScoreQuestion{},
		BooleanQuestion{}, &BooleanQuestion{},
		BoxQuestion{}, &BoxQuestion{},
	}
	if len(questions) != 8 {
		t.Fatalf("questions = %d", len(questions))
	}
	parts := []StatePart{
		TextStatePart{}, &TextStatePart{},
		JSONStatePart{}, &JSONStatePart{},
		FileStatePart{}, &FileStatePart{},
	}
	if len(parts) != 6 {
		t.Fatalf("state parts = %d", len(parts))
	}
	answers := []Answer{
		ChoiceAnswer{}, &ChoiceAnswer{},
		ScoreAnswer{}, &ScoreAnswer{},
		BooleanAnswer{}, &BooleanAnswer{},
		BoxAnswer{}, &BoxAnswer{},
	}
	if len(answers) != 8 {
		t.Fatalf("answers = %d", len(answers))
	}
}

func TestQuestionTypeConstants(t *testing.T) {
	got := []QuestionType{QuestionTypeChoice, QuestionTypeScore, QuestionTypeBoolean, QuestionTypeBox}
	want := []QuestionType{"choice", "score", "boolean", "box"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("question types = %v", got)
	}
	// The neutral vocabulary never uses provider wire names.
	for _, qt := range got {
		if qt == "noul" {
			t.Fatal("neutral question types must not contain wire names")
		}
	}
}

func TestUnsupportedQuestionTypeError(t *testing.T) {
	err := &UnsupportedQuestionTypeError{QuestionID: "q1", Type: QuestionTypeBoolean}
	if got := err.Error(); got != `decision: question "q1" has unsupported type "boolean"` {
		t.Fatalf("message = %q", got)
	}
	var target *UnsupportedQuestionTypeError
	if !errors.As(err, &target) {
		t.Fatal("errors.As failed")
	}
}
