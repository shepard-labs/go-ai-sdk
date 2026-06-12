package openai

import (
	"testing"
	"time"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// --- pointer helpers ---

func TestIntPtr(t *testing.T) {
	v := intPtr(7)
	if v == nil || *v != 7 {
		t.Errorf("intPtr: %v", v)
	}
}

func TestFloatPtr(t *testing.T) {
	v := floatPtr(3.14)
	if v == nil || *v != 3.14 {
		t.Errorf("floatPtr: %v", v)
	}
}

func TestStringPtr(t *testing.T) {
	v := stringPtr("hi")
	if v == nil || *v != "hi" {
		t.Errorf("stringPtr: %v", v)
	}
}

func TestBoolPtr(t *testing.T) {
	v := boolPtr(true)
	if v == nil || !*v {
		t.Errorf("boolPtr: %v", v)
	}
}

func TestIntValueOrZero(t *testing.T) {
	if got := intValueOrZero(nil); got != 0 {
		t.Errorf("nil: %d", got)
	}
	if got := intValueOrZero(intPtr(11)); got != 11 {
		t.Errorf("ptr: %d", got)
	}
}

func TestDerefString(t *testing.T) {
	if got := derefString(nil, "fb"); got != "fb" {
		t.Errorf("nil: %q", got)
	}
	if got := derefString(stringPtr("x"), "fb"); got != "x" {
		t.Errorf("ptr: %q", got)
	}
}

// --- defaultOpenAIRetry ---

func TestDefaultOpenAIRetryNil(t *testing.T) {
	got := defaultOpenAIRetry(nil)
	if got.MaxRetries != 2 {
		t.Errorf("MaxRetries: %d", got.MaxRetries)
	}
	if got.Jitter != true {
		t.Errorf("Jitter: %v", got.Jitter)
	}
	if got.BaseDelay <= 0 {
		t.Errorf("BaseDelay: %v", got.BaseDelay)
	}
}

func TestDefaultOpenAIRetryOverrides(t *testing.T) {
	got := defaultOpenAIRetry(&RetryOptions{
		MaxRetries: 5,
		BaseDelay:  500 * time.Millisecond,
		MaxDelay:   10 * time.Second,
		Jitter:     false,
	})
	if got.MaxRetries != 5 {
		t.Errorf("MaxRetries: %d", got.MaxRetries)
	}
	if got.BaseDelay != 500*time.Millisecond {
		t.Errorf("BaseDelay: %v", got.BaseDelay)
	}
	if got.Jitter {
		t.Errorf("Jitter: %v", got.Jitter)
	}
}

func TestDefaultOpenAIRetryClampsNegativeRetries(t *testing.T) {
	got := defaultOpenAIRetry(&RetryOptions{MaxRetries: -1})
	if got.MaxRetries != 0 {
		t.Errorf("MaxRetries: %d", got.MaxRetries)
	}
}

func TestDefaultOpenAIRetryMaxDelayBumpsToBaseDelay(t *testing.T) {
	got := defaultOpenAIRetry(&RetryOptions{
		BaseDelay: 5 * time.Second,
		MaxDelay:  1 * time.Second,
	})
	if got.MaxDelay < got.BaseDelay {
		t.Errorf("MaxDelay < BaseDelay: %v < %v", got.MaxDelay, got.BaseDelay)
	}
}

// --- openaiDefaultIfZero ---

func TestOpenaiDefaultIfZero(t *testing.T) {
	if got := openaiDefaultIfZero(0, 5); got != 5 {
		t.Errorf("zero: %d", got)
	}
	if got := openaiDefaultIfZero(3, 5); got != 3 {
		t.Errorf("non-zero: %d", got)
	}
}

// --- retryAfterDelay ---

func TestRetryAfterDelayEmpty(t *testing.T) {
	if _, ok := retryAfterDelay(""); ok {
		t.Errorf("empty should return false")
	}
}

func TestRetryAfterDelaySeconds(t *testing.T) {
	d, ok := retryAfterDelay("5")
	if !ok {
		t.Fatal("expected ok")
	}
	if d != 5*time.Second {
		t.Errorf("duration: %v", d)
	}
}

func TestRetryAfterDelayHTTPTime(t *testing.T) {
	future := time.Now().Add(1 * time.Hour).UTC().Format(httpTimeFormat)
	d, ok := retryAfterDelay(future)
	if !ok {
		t.Fatal("expected ok for HTTP time")
	}
	if d <= 0 {
		t.Errorf("future: %v", d)
	}
}

func TestRetryAfterDelayPastHTTPTime(t *testing.T) {
	past := time.Now().Add(-1 * time.Hour).UTC().Format(httpTimeFormat)
	d, ok := retryAfterDelay(past)
	if !ok {
		t.Fatal("expected ok")
	}
	if d != 0 {
		t.Errorf("past: %v", d)
	}
}

func TestRetryAfterDelayUnparseable(t *testing.T) {
	if _, ok := retryAfterDelay("not-a-time"); ok {
		t.Errorf("unparseable should return false")
	}
}

const httpTimeFormat = "Mon, 02 Jan 2006 15:04:05 GMT"

// --- retryJitter ---

func TestRetryJitterDisabled(t *testing.T) {
	d := 5 * time.Second
	if got := retryJitter(d, false); got != d {
		t.Errorf("disabled: %v", got)
	}
}

func TestRetryJitterZero(t *testing.T) {
	if got := retryJitter(0, true); got != 0 {
		t.Errorf("zero: %v", got)
	}
}

func TestRetryJitterEnabled(t *testing.T) {
	d := 5 * time.Second
	got := retryJitter(d, true)
	if got < 0 || got > d {
		t.Errorf("out of range: %v not in [0, %v]", got, d)
	}
}

// --- openaiNoopLogger ---

func TestOpenaiNoopLogger(t *testing.T) {
	var l openaiNoopLogger
	l.Debug("a", 1)
	l.Info("b", 2)
	l.Warn("c", 3)
	l.Error("d", 4)
	// All no-ops, so this test just confirms none panic.
}

// --- openaiDefaultHTTPClient ---

func TestOpenaiDefaultHTTPClient(t *testing.T) {
	c := openaiDefaultHTTPClient()
	if c == nil {
		t.Fatal("expected client")
	}
	if c.Timeout != 0 {
		t.Errorf("timeout should be 0: %v", c.Timeout)
	}
}

// --- Type assertion methods ---

func TestAssistantContentAssertions(t *testing.T) {
	// TextContent
	tc := TextContent{Text: "x"}
	if _, ok := any(tc).(openaicompatible.AssistantContent); !ok {
		t.Errorf("TextContent should implement AssistantContent")
	}
	if _, ok := any(tc).(openaicompatible.Content); !ok {
		t.Errorf("TextContent should implement Content")
	}
	// ReasoningContent
	rc := ReasoningContent{Text: "y"}
	if _, ok := any(rc).(openaicompatible.AssistantContent); !ok {
		t.Errorf("ReasoningContent should implement AssistantContent")
	}
	if _, ok := any(rc).(openaicompatible.Content); !ok {
		t.Errorf("ReasoningContent should implement Content")
	}
	// ToolCallContent
	tcc := ToolCallContent{}
	if _, ok := any(tcc).(openaicompatible.AssistantContent); !ok {
		t.Errorf("ToolCallContent should implement AssistantContent")
	}
	if _, ok := any(tcc).(openaicompatible.Content); !ok {
		t.Errorf("ToolCallContent should implement Content")
	}
	// ToolApprovalResponse
	tar := ToolApprovalResponse{ApprovalID: "a"}
	if _, ok := any(tar).(openaicompatible.AssistantContent); !ok {
		t.Errorf("ToolApprovalResponse should implement AssistantContent")
	}
	// CustomContent
	cc := CustomContent{}
	if _, ok := any(cc).(openaicompatible.UserContent); !ok {
		t.Errorf("CustomContent should implement UserContent")
	}
	if _, ok := any(cc).(openaicompatible.AssistantContent); !ok {
		t.Errorf("CustomContent should implement AssistantContent")
	}
	if _, ok := any(cc).(openaicompatible.Content); !ok {
		t.Errorf("CustomContent should implement Content")
	}
	// SourceContent
	sc := SourceContent{}
	if _, ok := any(sc).(openaicompatible.Content); !ok {
		t.Errorf("SourceContent should implement Content")
	}
	if _, ok := any(sc).(openaicompatible.StreamPart); !ok {
		t.Errorf("SourceContent should implement StreamPart")
	}
	// CompactionContent
	cpc := CompactionContent{}
	if _, ok := any(cpc).(openaicompatible.Content); !ok {
		t.Errorf("CompactionContent should implement Content")
	}
}

func TestUserContentAssertions(t *testing.T) {
	ut := TextContent{}
	if _, ok := any(ut).(openaicompatible.UserContent); !ok {
		t.Errorf("TextContent should implement UserContent")
	}
}

func TestToolContentAssertions(t *testing.T) {
	tr := ToolResultContent{}
	if _, ok := any(tr).(openaicompatible.Content); !ok {
		t.Errorf("ToolResultContent should implement Content")
	}
	if _, ok := any(tr).(openaicompatible.ToolContent); !ok {
		t.Errorf("ToolResultContent should implement ToolContent")
	}
}

func TestStreamPartAssertions(t *testing.T) {
	parts := []openaicompatible.StreamPart{
		StreamTextStart{},
		StreamTextDelta{},
		StreamTextEnd{},
		StreamReasoningStart{},
		StreamReasoningDelta{},
		StreamReasoningEnd{},
		StreamToolInputStart{},
		StreamToolInputDelta{},
		StreamToolInputEnd{},
		StreamToolCall{},
		StreamCustomPart{},
		StreamToolApprovalRequest{},
		StreamCompactionEnd{},
		StreamRaw{},
		StreamResponseMetadata{},
		StreamStart{},
		StreamFinish{},
		StreamError{},
	}
	for _, p := range parts {
		if _, ok := any(p).(openaicompatible.StreamPart); !ok {
			t.Errorf("%T should implement StreamPart", p)
		}
	}

	// Delta variants
	deltas := []openaicompatible.StreamPart{
		StreamTextDelta{},
		StreamReasoningDelta{},
		StreamToolInputDelta{},
	}
	for _, d := range deltas {
		if _, ok := any(d).(openaicompatible.Delta); !ok {
			t.Errorf("%T should implement Delta", d)
		}
	}
}

// --- valueOrDefault ---

func TestValueOrDefaultEmpty(t *testing.T) {
	if got := valueOrDefault("", "fb"); got != "fb" {
		t.Errorf("empty: %q", got)
	}
}

func TestValueOrDefaultSet(t *testing.T) {
	if got := valueOrDefault("hi", "fb"); got != "hi" {
		t.Errorf("set: %q", got)
	}
}

// --- jsonFloat ---

func TestJSONFloatValid(t *testing.T) {
	if got := jsonFloat(3.14); got != "3.14" {
		t.Errorf("got: %q", got)
	}
}

func TestJSONFloatZero(t *testing.T) {
	if got := jsonFloat(0); got != "0" {
		t.Errorf("zero: %q", got)
	}
}
