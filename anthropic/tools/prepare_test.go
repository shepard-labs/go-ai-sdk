package tools

import (
	"testing"

	anthropic "github.com/shepard-labs/go-ai-sdk-anthropic/anthropic"
)

func TestPrepareToolsOptions(t *testing.T) {
	deferLoading := true
	eager := true
	prepared := PrepareTools([]anthropic.Tool{{Name: "tool"}}, anthropic.ToolOptions{DeferLoading: &deferLoading, EagerInputStreaming: &eager, AllowedCallers: []anthropic.ToolCallCaller{"caller"}})
	if prepared[0].DeferLoading == nil || !*prepared[0].DeferLoading || prepared[0].EagerInputStreaming == nil || !*prepared[0].EagerInputStreaming || prepared[0].AllowedCallers[0] != "caller" {
		t.Fatalf("prepared = %#v", prepared[0])
	}
}

func TestProviderToolConstructors(t *testing.T) {
	max := 10
	zoom := true
	tests := []anthropic.Tool{
		Bash_20241022(),
		Bash_20250124(),
		CodeExecution_20250522(),
		CodeExecution_20250825(),
		CodeExecution_20260120(),
		Computer_20251124(800, 600, 1, zoom),
		TextEditor_20250728(&max),
		Memory_20250818(),
		WebFetch_20260209(&max, []string{"example.com"}, nil, &anthropic.CitationsConfig{Enabled: true}, &max),
		WebSearch_20260209(&max, nil, []string{"blocked.com"}, &anthropic.UserLocation{Country: "US"}),
		ToolSearchRegex_20251119(),
		ToolSearchBm25_20251119(),
		Advisor_20260301("claude", &max, &anthropic.CachingConfig{Enabled: true}),
	}
	for _, tool := range tests {
		if tool.ID == "" || tool.Name == "" || !tool.ProviderExecuted {
			t.Fatalf("tool = %#v", tool)
		}
	}
	if *tests[5].EnableZoom != true || *tests[6].MaxCharacters != 10 || tests[8].AllowedDomains[0] != "example.com" || tests[12].AdvisorModel != "claude" {
		t.Fatalf("tools = %#v", tests)
	}
}

func TestToolNameMapping(t *testing.T) {
	if ToolNameMapping["anthropic.code_execution_20250522"] != "code_execution" || ToolNameMapping["anthropic.text_editor_20250728"] != "str_replace_based_edit_tool" || ToolNameMapping["anthropic.advisor_20260301"] != "advisor" {
		t.Fatalf("mapping = %#v", ToolNameMapping)
	}
}

func TestContainerIdForwarding(t *testing.T) {
	options, err := ForwardAnthropicContainerIdFromLastStep([]Step{
		{MessageMetadata: anthropic.MessageMetadata{"container": anthropic.ContainerInfo{ID: "old"}}},
		{ProviderMetadata: anthropic.ProviderMetadata{"containerID": "new"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if options.Container == nil || options.Container.ID != "new" {
		t.Fatalf("options = %#v", options)
	}
}
