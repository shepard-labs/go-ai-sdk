// Command validation demonstrates AgentLoopOptions.ToolPolicies: attaching a
// Validate function to a tool so the agent loop checks the model's tool input
// before accepting it. When validation fails, the loop feeds the error back to
// the model as a tool result so it can self-correct ("repair"); MaxToolRepairs
// caps how many such corrections are allowed before the loop gives up. This is
// how you enforce invariants the JSON Schema alone can't express.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"
	"github.com/shepard-labs/go-ai-sdk/llm/schema"

	// Blank-import the provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

// MeetingSlot is the structured output. The schema constrains the shape, but a
// cross-field rule — end must be after start — needs a Validate function.
type MeetingSlot struct {
	Day       string `json:"day" description:"day of week" enum:"Mon,Tue,Wed,Thu,Fri"`
	StartHour int    `json:"start_hour" description:"start hour, 24h" minimum:"0" maximum:"23"`
	EndHour   int    `json:"end_hour" description:"end hour, 24h" minimum:"0" maximum:"23"`
}

func main() {
	client, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	tool, err := schema.Tool("submit_slot", "Submit the chosen meeting slot.", MeetingSlot{})
	if err != nil {
		log.Fatalf("build tool schema: %v", err)
	}

	// Validate enforces the cross-field invariant the schema can't. Returning a
	// non-nil error makes the loop send the message back to the model as a tool
	// error, prompting it to retry with corrected input.
	validate := func(input json.RawMessage) error {
		var slot MeetingSlot
		if err := json.Unmarshal(input, &slot); err != nil {
			return fmt.Errorf("malformed input: %w", err)
		}
		if slot.EndHour <= slot.StartHour {
			return fmt.Errorf("end_hour (%d) must be greater than start_hour (%d)", slot.EndHour, slot.StartHour)
		}
		return nil
	}

	result, err := llm.AgentLoopResultWithOptions(context.Background(), client, llm.GenerateOptions{
		System: "You schedule meetings. Call submit_slot exactly once with a valid slot.",
		Messages: []llm.Message{{
			Role:    "user",
			Content: []llm.Content{llm.TextContent{Text: "Book a 1-hour meeting on Wednesday afternoon."}},
		}},
		Tools:     []llm.Tool{tool},
		MaxTokens: 1024,
	}, nil, llm.AgentLoopOptions{
		SubmitResultTool: tool.Name,
		MaxTurns:         6,
		ToolPolicies: map[string]llm.ToolPolicy{
			tool.Name: {Validate: validate},
		},
		MaxToolRepairs: 3, // allow up to 3 self-corrections before failing
	})
	if err != nil {
		log.Fatalf("agent loop: %v", err)
	}

	var slot MeetingSlot
	if err := json.Unmarshal(result.Input, &slot); err != nil {
		log.Fatalf("decode result: %v", err)
	}
	fmt.Printf("Booked: %s %02d:00–%02d:00\n", slot.Day, slot.StartHour, slot.EndHour)
	fmt.Printf("(loop took %d turn(s), %d repair(s))\n", result.Turns, result.Repairs)
}
