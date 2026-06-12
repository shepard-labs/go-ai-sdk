// Package ai is the provider-agnostic router for the go-ai-sdk monorepo.
// It takes a user prompt, asks Claude Haiku which (provider, model) pair
// is best for that prompt, then dispatches the call to that provider.
//
// Sub-packages (anthropic, openai, google, cohere, openaicompatible,
// openrouter) remain available for power users who want raw access to
// a single provider.
package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/anthropic"
	"github.com/shepard-labs/go-ai-sdk/internal/adapters"
	"github.com/shepard-labs/go-ai-sdk/openai"
)

const (
	// DefaultRoutingModel is the model used for the routing decision.
	// claude-haiku-4-5 supports native structured output and is cheap
	// enough to call once per user request.
	DefaultRoutingModel = "claude-haiku-4-5-20251001"

	// routingMaxTokens bounds the routing-call response; the JSON
	// payload is small (a few words + one sentence).
	routingMaxTokens = 256

	// routingUserPromptTruncationBytes bounds the user prompt that
	// the routing model sees, so a 100KB paste doesn't cost 100KB of
	// input tokens on the routing call.
	routingUserPromptTruncationBytes = 4096
)

// Errors returned by Router.
var (
	// ErrEmptyCatalog is returned by CreateRouter when the catalog
	// has no entries.
	ErrEmptyCatalog = errors.New("ai: catalog is empty")
	// ErrNoProvidersConfigured is returned by CreateRouter when no
	// provider is set in RouterSettings.
	ErrNoProvidersConfigured = errors.New("ai: at least one provider must be configured")
	// ErrProviderNotSupported is returned when a chosen provider has
	// no adapter (e.g. google or cohere in v1).
	ErrProviderNotSupported = errors.New("ai: provider is configured but not yet supported")
	// ErrRoutingDecision is returned when the routing call returns
	// malformed JSON or an empty payload.
	ErrRoutingDecision = errors.New("ai: malformed routing decision")
	// ErrUnknownModel is returned when the routing call returns a
	// model that isn't in the catalog for the chosen provider.
	ErrUnknownModel = errors.New("ai: model not in catalog")
	// ErrUnknownProvider is returned when the routing call returns
	// a provider that the router was not configured for.
	ErrUnknownProvider = errors.New("ai: unknown provider from routing decision")
)

// NoSuchProviderError is returned when the routing decision named a
// provider that the router was not configured for, or that has no
// adapter.
type NoSuchProviderError struct {
	Provider string
	Known    []string
}

func (e *NoSuchProviderError) Error() string {
	return fmt.Sprintf("ai: provider %q not configured (known: %v)", e.Provider, e.Known)
}

// RouterSettings configures a Router. Anthropic is required (the
// router uses it for the routing call); the other providers are
// optional and only need to be set if the catalog references them.
type RouterSettings struct {
	// Anthropic is used for the routing call. Required.
	Anthropic anthropic.Provider
	// OpenAI is dispatched to when the routing decision picks "openai".
	// Optional — leave nil if the catalog has no openai entries.
	OpenAI openai.Provider

	// Catalog is the list of allowed models per provider. Required.
	// The provider keys must match the Provider's Name() return
	// (e.g. "anthropic", "openai").
	Catalog ProviderCatalog

	// RoutingModel overrides the routing model. Defaults to
	// DefaultRoutingModel ("claude-haiku-4-5-20251001").
	RoutingModel string

	// SystemPromptPrefix is appended to the auto-generated routing
	// system prompt. Use it to bias routing ("Prefer anthropic for
	// safety-sensitive tasks", etc.).
	SystemPromptPrefix string
}

// Router is the public router type. Mirrors the Provider shape used
// by every other package: a value with Err()/Name() and the call
// methods.
type Router struct {
	settings     RouterSettings
	anthropic    anthropic.Provider
	openai       openai.Provider
	catalog      ProviderCatalog
	routingModel anthropic.LanguageModel
	routingName  string
	err          error
}

// Name returns the package's display name.
func (r *Router) Name() string { return "ai" }

// Err returns the most recent configuration / startup error, or nil
// if the router is ready to use.
func (r *Router) Err() error { return r.err }

// CreateRouter validates settings, builds the routing model, and
// returns a ready-to-use *Router. Always check Err() on the result.
func CreateRouter(settings RouterSettings) *Router {
	r := &Router{
		settings:  settings,
		anthropic: settings.Anthropic,
		openai:    settings.OpenAI,
		catalog:   settings.Catalog,
	}
	if len(settings.Catalog) == 0 {
		r.err = ErrEmptyCatalog
		return r
	}
	if settings.Anthropic == nil && settings.OpenAI == nil {
		r.err = ErrNoProvidersConfigured
		return r
	}
	if settings.Anthropic == nil {
		r.err = ErrNoProvidersConfigured
		return r
	}
	if err := settings.Anthropic.Err(); err != nil {
		r.err = fmt.Errorf("ai: anthropic provider error: %w", err)
		return r
	}
	if settings.OpenAI != nil {
		if err := settings.OpenAI.Err(); err != nil {
			r.err = fmt.Errorf("ai: openai provider error: %w", err)
			return r
		}
	}
	routingName := settings.RoutingModel
	if routingName == "" {
		routingName = DefaultRoutingModel
	}
	routingModel := settings.Anthropic.Model(
		routingName,
		anthropic.ModelOptions{StructuredOutputMode: anthropic.StructuredOutputModeAuto},
	)
	r.routingModel = routingModel
	r.routingName = routingName
	return r
}

// Default is the package-level Router instance, configured from
// environment variables for the Anthropic provider only. Use
// CreateRouter for full control (e.g. to add OpenAI or to override
// the catalog).
var Default = CreateRouter(RouterSettings{
	Anthropic: anthropic.CreateAnthropic(anthropic.ProviderSettings{}),
})

// RouterResult is the sealed result of Router.Generate. The
// underlying value is one of AnthropicResult or OpenAIResult;
// type-switch on Kind and use the corresponding Value field.
type RouterResult struct {
	Kind       ResultKind
	Anthropic  *AnthropicResult
	OpenAI     *OpenAIResult
}

// ResultKind identifies which provider produced a RouterResult.
type ResultKind string

const (
	ResultKindAnthropic ResultKind = "anthropic"
	ResultKindOpenAI    ResultKind = "openai"
)

// AnthropicResult wraps an anthropic.GenerateResult. Defined as a
// distinct type (not a type alias) so the RouterResult interface
// stays local to this package and isRouterResult can attach to it.
type AnthropicResult struct{ Value anthropic.GenerateResult }

// OpenAIResult wraps an openai.GenerateResult.
type OpenAIResult struct{ Value openai.GenerateResult }

// StreamKind identifies which provider produced a StreamResult.
type StreamKind string

const (
	StreamKindAnthropic StreamKind = "anthropic"
	StreamKindOpenAI    StreamKind = "openai"
)

// StreamEnvelope carries the underlying provider's StreamResult. Only
// the field matching Kind is non-nil.
type StreamEnvelope struct {
	Kind      StreamKind
	Anthropic *anthropic.StreamResult
	OpenAI    *openai.StreamResult
}

// Generate dispatches the call to the provider/model chosen by the
// router. Returns a RouterResult carrying the chosen provider's
// GenerateResult, the Selection that was used, and any error.
func (r *Router) Generate(ctx context.Context, opts RouterOptions) (RouterResult, Selection, error) {
	sel, err := r.pick(ctx, opts)
	if err != nil {
		return RouterResult{}, sel, err
	}
	switch sel.Provider {
	case "anthropic":
		if r.anthropic == nil {
			return RouterResult{}, sel, &NoSuchProviderError{Provider: sel.Provider, Known: r.knownProviders()}
		}
		m := r.anthropic.Model(sel.Model)
		gen := adapters.BuildAnthropicGenerateOpts(opts.Prompt, opts.System, opts.Temperature, opts.MaxTokens)
		res, err := adapters.GenerateAnthropic(ctx, m, gen)
		if err != nil {
			return RouterResult{}, sel, err
		}
		return RouterResult{Kind: ResultKindAnthropic, Anthropic: &AnthropicResult{Value: res}}, sel, nil
	case "openai":
		if r.openai == nil {
			return RouterResult{}, sel, &NoSuchProviderError{Provider: sel.Provider, Known: r.knownProviders()}
		}
		m := r.openai.Chat(sel.Model) // .Chat() not .Model(); see internal/adapters/openai.go
		gen := adapters.BuildOpenAIGenerateOpts(opts.Prompt, opts.System, opts.Temperature, opts.MaxTokens)
		res, err := adapters.GenerateOpenAI(ctx, m, gen)
		if err != nil {
			return RouterResult{}, sel, err
		}
		return RouterResult{Kind: ResultKindOpenAI, OpenAI: &OpenAIResult{Value: res}}, sel, nil
	default:
		return RouterResult{}, sel, &NoSuchProviderError{Provider: sel.Provider, Known: r.knownProviders()}
	}
}

// Stream dispatches the call in streaming mode.
func (r *Router) Stream(ctx context.Context, opts RouterOptions) (*StreamEnvelope, Selection, error) {
	sel, err := r.pick(ctx, opts)
	if err != nil {
		return nil, sel, err
	}
	switch sel.Provider {
	case "anthropic":
		if r.anthropic == nil {
			return nil, sel, &NoSuchProviderError{Provider: sel.Provider, Known: r.knownProviders()}
		}
		m := r.anthropic.Model(sel.Model)
		streamOpts := adapters.BuildAnthropicStreamOpts(opts.Prompt, opts.System, opts.Temperature, opts.MaxTokens)
		res, err := adapters.StreamAnthropic(ctx, m, streamOpts)
		if err != nil {
			return nil, sel, err
		}
		return &StreamEnvelope{Kind: StreamKindAnthropic, Anthropic: res}, sel, nil
	case "openai":
		if r.openai == nil {
			return nil, sel, &NoSuchProviderError{Provider: sel.Provider, Known: r.knownProviders()}
		}
		m := r.openai.Chat(sel.Model)
		streamOpts := adapters.BuildOpenAIStreamOpts(opts.Prompt, opts.System, opts.Temperature, opts.MaxTokens)
		res, err := adapters.StreamOpenAI(ctx, m, streamOpts)
		if err != nil {
			return nil, sel, err
		}
		return &StreamEnvelope{Kind: StreamKindOpenAI, OpenAI: res}, sel, nil
	default:
		return nil, sel, &NoSuchProviderError{Provider: sel.Provider, Known: r.knownProviders()}
	}
}

// pick resolves the (provider, model) pair to use. Bypassed by
// opts.ForceProvider.
func (r *Router) pick(ctx context.Context, opts RouterOptions) (Selection, error) {
	if opts.ForceProvider != "" {
		if _, ok := r.catalog[opts.ForceProvider]; !ok {
			return Selection{Provider: opts.ForceProvider, Model: opts.ForceModel},
				&NoSuchProviderError{Provider: opts.ForceProvider, Known: r.knownProviders()}
		}
		return Selection{Provider: opts.ForceProvider, Model: opts.ForceModel, Reason: "forced"}, nil
	}
	return r.callHaiku(ctx, opts.Prompt)
}

type routingDecision struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	Reason   string `json:"reason"`
}

func (r *Router) callHaiku(ctx context.Context, userPrompt string) (Selection, error) {
	schema := routingDecisionSchema(r.knownProviders())
	gen := anthropic.GenerateOptions{
		Messages: []anthropic.Message{
			anthropic.SystemMessage{Content: r.routingSystemPrompt()},
			anthropic.UserMessage{Content: []anthropic.UserContent{
				anthropic.TextContent{Text: wrapRoutingUserPrompt(userPrompt)},
			}},
		},
		MaxTokens: routingMaxTokens,
		// No Temperature set — leave provider default (deterministic
		// for structured output).
		StructuredOutput: &anthropic.StructuredOutput{
			Name:        "routing_decision",
			Description: "Pick the best (provider, model) pair for the user's prompt.",
			Schema:      schema,
		},
	}
	res, err := r.routingModel.DoGenerate(ctx, gen)
	if err != nil {
		return Selection{}, fmt.Errorf("ai: routing call failed: %w", err)
	}
	if res == nil {
		return Selection{}, ErrRoutingDecision
	}
	var text string
	for _, c := range res.Content {
		if tc, ok := c.(anthropic.TextContent); ok {
			text += tc.Text
		}
	}
	if strings.TrimSpace(text) == "" {
		return Selection{}, ErrRoutingDecision
	}
	var d routingDecision
	if err := json.Unmarshal([]byte(text), &d); err != nil {
		return Selection{}, fmt.Errorf("%w: %v", ErrRoutingDecision, err)
	}
	if d.Provider == "" || d.Model == "" {
		return Selection{}, ErrRoutingDecision
	}
	allowed, ok := r.catalog[d.Provider]
	if !ok {
		return Selection{Provider: d.Provider, Model: d.Model, Reason: d.Reason},
			&NoSuchProviderError{Provider: d.Provider, Known: r.knownProviders()}
	}
	if !containsString(allowed, d.Model) {
		return Selection{Provider: d.Provider, Model: d.Model, Reason: d.Reason},
			fmt.Errorf("%w: provider=%q model=%q", ErrUnknownModel, d.Provider, d.Model)
	}
	return Selection{Provider: d.Provider, Model: d.Model, Reason: d.Reason}, nil
}

func (r *Router) knownProviders() []string {
	keys := make([]string, 0, len(r.catalog))
	for k := range r.catalog {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// routingDecisionSchema builds the JSON schema for the routing
// response, with the enum populated from the configured providers.
func routingDecisionSchema(providers []string) map[string]any {
	enum := make([]any, len(providers))
	for i, p := range providers {
		enum[i] = p
	}
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"provider": map[string]any{
				"type":        "string",
				"enum":        enum,
				"description": "The provider to dispatch to. Must be one of the configured providers.",
			},
			"model": map[string]any{
				"type":        "string",
				"description": "The model ID within the chosen provider.",
			},
			"reason": map[string]any{
				"type":        "string",
				"description": "One-sentence justification for this choice.",
			},
		},
		"required":             []string{"provider", "model", "reason"},
		"additionalProperties": false,
	}
}

// routingSystemPrompt is the system prompt sent to the routing model.
// It is generated per call so the catalog is always fresh.
func (r *Router) routingSystemPrompt() string {
	var sb strings.Builder
	sb.WriteString("You are a router for an LLM gateway. Pick the best (provider, model) pair for the user's prompt from the catalog below.\n\n")
	sb.WriteString("PROVIDERS AND MODELS:\n")
	keys := r.knownProviders()
	for _, k := range keys {
		fmt.Fprintf(&sb, "- %s: %s\n", k, strings.Join(r.catalog[k], ", "))
	}
	sb.WriteString("\nRULES:\n")
	sb.WriteString("- Return JSON with \"provider\", \"model\", \"reason\".\n")
	fmt.Fprintf(&sb, "- Provider must be one of: %s.\n", strings.Join(keys, ", "))
	sb.WriteString("- Model must be in the catalog for the chosen provider.\n")
	sb.WriteString("- Prefer cheaper/faster models for short, simple prompts.\n")
	sb.WriteString("- Prefer stronger models for reasoning, code, math, or long-context tasks.\n")
	sb.WriteString("- \"reason\" must be one sentence.\n")
	if r.settings.SystemPromptPrefix != "" {
		sb.WriteString("\n")
		sb.WriteString(r.settings.SystemPromptPrefix)
		sb.WriteString("\n")
	}
	sb.WriteString("\nRESPOND WITH ONLY THE JSON. NO PROSE, NO MARKDOWN FENCES.\n")
	return sb.String()
}

// wrapRoutingUserPrompt bounds the size of the user prompt sent to
// the routing model.
func wrapRoutingUserPrompt(prompt string) string {
	if len(prompt) > routingUserPromptTruncationBytes {
		prompt = prompt[:routingUserPromptTruncationBytes] + "\n\n[truncated]"
	}
	return "USER PROMPT:\n" + prompt
}

func containsString(xs []string, target string) bool {
	for _, x := range xs {
		if x == target {
			return true
		}
	}
	return false
}
