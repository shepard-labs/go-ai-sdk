package openai

import (
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openaicompatible"
)

// Verifies that all openai-package content types implement the
// appropriate union marker methods. This is a compile-time test that
// also surfaces the interface membership at runtime for documentation.
func TestContentUnionMarkerMethods(t *testing.T) {
	// User content
	var uc openaicompatible.UserContent = TextContent{Text: "x"}
	if _, ok := uc.(openaicompatible.UserContent); !ok {
		t.Errorf("TextContent should implement UserContent")
	}
	var uc2 openaicompatible.UserContent = CustomContent{Kind: "k", Data: 1}
	if _, ok := uc2.(openaicompatible.UserContent); !ok {
		t.Errorf("CustomContent should implement UserContent")
	}

	// Assistant content
	var ac openaicompatible.AssistantContent = TextContent{Text: "x"}
	_ = ac
	var ac2 openaicompatible.AssistantContent = ReasoningContent{Text: "y"}
	_ = ac2
	var ac3 openaicompatible.AssistantContent = ToolCallContent{ToolCallContentEmbed: ToolCallContentEmbed{ToolCallID: "1", ToolName: "f"}}
	_ = ac3

	// Tool content
	var tc openaicompatible.ToolContent = ToolResultContent{}
	_ = tc

	// Content (openai-specific)
	var c openaicompatible.Content = CompactionContent{ItemID: "i", EncryptedContent: "e"}
	_ = c
	var c2 openaicompatible.Content = SourceContent{URL: "u"}
	_ = c2
}

// Verifies that ToolApprovalResponse implements AssistantContent
// (it's a special response to a stream ToolApprovalRequest).
func TestToolApprovalResponseImplementsAssistantContent(t *testing.T) {
	var ac openaicompatible.AssistantContent = ToolApprovalResponse{ApprovalID: "a", Approve: true}
	if _, ok := ac.(ToolApprovalResponse); !ok {
		t.Errorf("ToolApprovalResponse should implement AssistantContent")
	}
}
