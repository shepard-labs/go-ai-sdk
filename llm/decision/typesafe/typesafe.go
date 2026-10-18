// Package typesafe adapts TypeSafe AI decision models to the provider-neutral
// decision.Client interface.
//
// The package name collides with the TypeSafe provider module; when both are
// imported, alias one of them:
//
//	import (
//		"github.com/shepard-labs/go-ai-sdk/llm/decision"
//		decisiontypesafe "github.com/shepard-labs/go-ai-sdk/llm/decision/typesafe"
//		"github.com/shepard-labs/go-ai-sdk/typesafe"
//	)
//
// The decision capability is experimental.
package typesafe

import (
	"context"
	"fmt"
	"net/http"

	"github.com/shepard-labs/go-ai-sdk/llm/decision"
	typesafesdk "github.com/shepard-labs/go-ai-sdk/typesafe"
)

// adapter wraps a typesafe.DecisionModel as a decision.Client.
type adapter struct {
	model typesafesdk.DecisionModel
}

// NewAdapter wraps an existing TypeSafe decision model as a decision.Client.
func NewAdapter(model typesafesdk.DecisionModel) decision.Client {
	return &adapter{model: model}
}

// NewClient creates a TypeSafe-backed decision.Client from provider settings
// and a model ID. A missing API key is reported here, not at call time.
func NewClient(settings typesafesdk.ProviderSettings, modelID string) (decision.Client, error) {
	provider := typesafesdk.CreateTypeSafeAI(settings)
	if err := provider.Err(); err != nil {
		return nil, err
	}
	return NewAdapter(provider.DecisionModel(modelID)), nil
}

// SupportedQuestionTypes reports the question types the underlying model
// accepts. The neutral and provider vocabularies share string values.
func (a *adapter) SupportedQuestionTypes() []decision.QuestionType {
	supported := a.model.SupportedQuestionTypes()
	out := make([]decision.QuestionType, len(supported))
	for i, t := range supported {
		out[i] = decision.QuestionType(t)
	}
	return out
}

// Decide maps the neutral call onto the TypeSafe decision model. Questions
// whose type the model does not support are rejected before any I/O.
func (a *adapter) Decide(ctx context.Context, opts decision.CallOptions) (*decision.Result, error) {
	supported := make(map[decision.QuestionType]bool, 4)
	for _, t := range a.SupportedQuestionTypes() {
		supported[t] = true
	}

	questions := make(map[string]typesafesdk.Question, len(opts.Questions))
	for id, q := range opts.Questions {
		qtype, converted, err := convertQuestion(q)
		if err != nil {
			return nil, fmt.Errorf("decision: question %q: %w", id, err)
		}
		if !supported[qtype] {
			return nil, &decision.UnsupportedQuestionTypeError{QuestionID: id, Type: qtype}
		}
		questions[id] = converted
	}

	state := make([]typesafesdk.StatePart, 0, len(opts.State))
	for _, part := range opts.State {
		converted, err := convertStatePart(part)
		if err != nil {
			return nil, err
		}
		state = append(state, converted)
	}

	var headers http.Header
	if len(opts.Headers) > 0 {
		headers = http.Header{}
		for k, v := range opts.Headers {
			headers.Set(k, v)
		}
	}

	result, err := a.model.DoDecide(ctx, typesafesdk.DecisionCallOptions{
		State:           state,
		Questions:       questions,
		Headers:         headers,
		ProviderOptions: map[string]map[string]any(opts.ProviderOptions),
	})
	if err != nil {
		return nil, err
	}
	return convertResult(result)
}

func convertQuestion(q decision.Question) (decision.QuestionType, typesafesdk.Question, error) {
	switch q := derefQuestion(q).(type) {
	case decision.ChoiceQuestion:
		instructions, err := convertInputPtr(q.Instructions)
		if err != nil {
			return "", nil, err
		}
		criteria := make(map[string]*typesafesdk.Input, len(q.Criteria))
		for name, in := range q.Criteria {
			converted, err := convertInputPtr(in)
			if err != nil {
				return "", nil, err
			}
			criteria[name] = converted
		}
		return decision.QuestionTypeChoice, typesafesdk.ChoiceQuestion{Instructions: instructions, Criteria: criteria}, nil
	case decision.ScoreQuestion:
		instructions, err := convertInputPtr(q.Instructions)
		if err != nil {
			return "", nil, err
		}
		criteria := make([]*typesafesdk.Input, len(q.Criteria))
		for i, in := range q.Criteria {
			converted, err := convertInputPtr(in)
			if err != nil {
				return "", nil, err
			}
			criteria[i] = converted
		}
		return decision.QuestionTypeScore, typesafesdk.ScoreQuestion{Instructions: instructions, Criteria: criteria}, nil
	case decision.BooleanQuestion:
		instructions, err := convertInputPtr(q.Instructions)
		if err != nil {
			return "", nil, err
		}
		var criteria *typesafesdk.BooleanCriteria
		if q.Criteria != nil {
			trueInput, err := convertInputPtr(q.Criteria.True)
			if err != nil {
				return "", nil, err
			}
			falseInput, err := convertInputPtr(q.Criteria.False)
			if err != nil {
				return "", nil, err
			}
			criteria = &typesafesdk.BooleanCriteria{True: trueInput, False: falseInput}
		}
		return decision.QuestionTypeBoolean, typesafesdk.BooleanQuestion{Instructions: instructions, Criteria: criteria}, nil
	case decision.BoxQuestion:
		raw, err := q.Instructions.MarshalJSON()
		if err != nil {
			return "", nil, err
		}
		return decision.QuestionTypeBox, typesafesdk.BoxQuestion{Instructions: typesafesdk.RawInput(raw)}, nil
	default:
		return "", nil, fmt.Errorf("unknown question type %T", q)
	}
}

func convertStatePart(part decision.StatePart) (typesafesdk.StatePart, error) {
	switch p := derefStatePart(part).(type) {
	case decision.TextStatePart:
		return typesafesdk.TextStatePart{Text: p.Text}, nil
	case decision.JSONStatePart:
		return typesafesdk.JSONStatePart{Value: p.Value}, nil
	case decision.FileStatePart:
		return typesafesdk.FileStatePart{
			Filename:        p.Filename,
			MediaType:       p.MediaType,
			Data:            p.Data,
			Reference:       p.Reference,
			InsertReference: p.InsertReference,
		}, nil
	default:
		return nil, fmt.Errorf("decision: unknown state part type %T", part)
	}
}

// convertInputPtr converts a nullable neutral Input to a nullable provider
// Input, surfacing deferred marshal errors from decision.JSONInput.
func convertInputPtr(in *decision.Input) (*typesafesdk.Input, error) {
	if in == nil {
		return nil, nil
	}
	raw, err := in.MarshalJSON()
	if err != nil {
		return nil, err
	}
	out := typesafesdk.RawInput(raw)
	return &out, nil
}

func convertResult(result *typesafesdk.DecisionResult) (*decision.Result, error) {
	answers := make(map[string]decision.Answer, len(result.Answers))
	for id, a := range result.Answers {
		converted, err := convertAnswer(a)
		if err != nil {
			return nil, fmt.Errorf("decision: answer %q: %w", id, err)
		}
		answers[id] = converted
	}
	var usage *decision.Usage
	if result.Usage != nil {
		usage = &decision.Usage{InputTokens: result.Usage.InputTokens, OutputTokens: result.Usage.OutputTokens}
	}
	warnings := make([]decision.Warning, len(result.Warnings))
	for i, w := range result.Warnings {
		warnings[i] = decision.Warning{Type: w.Type, Feature: w.Feature}
	}
	return &decision.Result{
		Answers:          answers,
		Rounding:         decision.Rounding{ProbabilityDecimals: result.Rounding.ProbabilityDecimals, ScoreDecimals: result.Rounding.ScoreDecimals},
		Usage:            usage,
		Warnings:         warnings,
		ProviderMetadata: decision.ProviderMetadata(result.ProviderMetadata),
		Response: decision.ResponseMetadata{
			ModelID: result.Response.ModelID,
			Headers: result.Response.Headers,
			Body:    result.Response.Body,
		},
	}, nil
}

func convertAnswer(a typesafesdk.Answer) (decision.Answer, error) {
	switch a := a.(type) {
	case typesafesdk.ChoiceAnswer:
		return decision.ChoiceAnswer{Choice: a.Choice, Probabilities: a.Probabilities}, nil
	case *typesafesdk.ChoiceAnswer:
		if a == nil {
			return nil, fmt.Errorf("nil provider answer of type %T", a)
		}
		return decision.ChoiceAnswer{Choice: a.Choice, Probabilities: a.Probabilities}, nil
	case typesafesdk.ScoreAnswer:
		return decision.ScoreAnswer{Score: a.Score, Probabilities: a.Probabilities}, nil
	case *typesafesdk.ScoreAnswer:
		if a == nil {
			return nil, fmt.Errorf("nil provider answer of type %T", a)
		}
		return decision.ScoreAnswer{Score: a.Score, Probabilities: a.Probabilities}, nil
	case typesafesdk.BooleanAnswer:
		return decision.BooleanAnswer{Probability: a.Probability}, nil
	case *typesafesdk.BooleanAnswer:
		if a == nil {
			return nil, fmt.Errorf("nil provider answer of type %T", a)
		}
		return decision.BooleanAnswer{Probability: a.Probability}, nil
	case typesafesdk.BoxAnswer:
		return decision.BoxAnswer{Box: decision.Box{Top: a.Box.Top, Bottom: a.Box.Bottom, Left: a.Box.Left, Right: a.Box.Right}}, nil
	case *typesafesdk.BoxAnswer:
		if a == nil {
			return nil, fmt.Errorf("nil provider answer of type %T", a)
		}
		return decision.BoxAnswer{Box: decision.Box{Top: a.Box.Top, Bottom: a.Box.Bottom, Left: a.Box.Left, Right: a.Box.Right}}, nil
	default:
		return nil, fmt.Errorf("unknown provider answer type %T", a)
	}
}

func derefQuestion(q decision.Question) decision.Question {
	switch v := q.(type) {
	case *decision.ChoiceQuestion:
		if v != nil {
			return *v
		}
	case *decision.ScoreQuestion:
		if v != nil {
			return *v
		}
	case *decision.BooleanQuestion:
		if v != nil {
			return *v
		}
	case *decision.BoxQuestion:
		if v != nil {
			return *v
		}
	}
	return q
}

func derefStatePart(p decision.StatePart) decision.StatePart {
	switch v := p.(type) {
	case *decision.TextStatePart:
		if v != nil {
			return *v
		}
	case *decision.JSONStatePart:
		if v != nil {
			return *v
		}
	case *decision.FileStatePart:
		if v != nil {
			return *v
		}
	}
	return p
}
