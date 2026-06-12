package tools

import (
	"testing"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

func TestFactoriesPropagateArgs(t *testing.T) {
	cases := []struct {
		name      string
		tool      openai.Tool
		checkType func(t *testing.T, args any)
	}{
		{
			name: "web_search",
			tool: WebSearch(openai.WebSearchArgs{
				SearchContextSize: strPtr("high"),
			}),
			checkType: func(t *testing.T, args any) {
				ws, ok := args.(openai.WebSearchArgs)
				if !ok {
					t.Fatalf("args: %T", args)
				}
				if ws.SearchContextSize == nil || *ws.SearchContextSize != "high" {
					t.Errorf("SearchContextSize: %v", ws.SearchContextSize)
				}
			},
		},
		{
			name: "web_search_preview",
			tool: WebSearchPreview(openai.WebSearchPreviewArgs{
				SearchContextSize: strPtr("medium"),
			}),
			checkType: func(t *testing.T, args any) {
				ws, ok := args.(openai.WebSearchPreviewArgs)
				if !ok {
					t.Fatalf("args: %T", args)
				}
				if ws.SearchContextSize == nil || *ws.SearchContextSize != "medium" {
					t.Errorf("SearchContextSize: %v", ws.SearchContextSize)
				}
			},
		},
		{
			name: "file_search",
			tool: FileSearch(openai.FileSearchArgs{
				VectorStoreIDs: []string{"vs_1", "vs_2"},
			}),
			checkType: func(t *testing.T, args any) {
				fs, ok := args.(openai.FileSearchArgs)
				if !ok {
					t.Fatalf("args: %T", args)
				}
				if len(fs.VectorStoreIDs) != 2 {
					t.Errorf("VectorStoreIDs: %v", fs.VectorStoreIDs)
				}
			},
		},
		{
			name: "shell",
			tool: Shell(openai.ShellArgs{
				Environment: &openai.ShellEnvironment{Type: "local"},
			}),
			checkType: func(t *testing.T, args any) {
				sh, ok := args.(openai.ShellArgs)
				if !ok {
					t.Fatalf("args: %T", args)
				}
				if sh.Environment == nil || sh.Environment.Type != "local" {
					t.Errorf("Environment: %v", sh.Environment)
				}
			},
		},
		{
			name: "image_generation",
			tool: ImageGeneration(openai.ImageGenerationArgs{
				Quality: strPtr("high"),
			}),
			checkType: func(t *testing.T, args any) {
				ig, ok := args.(openai.ImageGenerationArgs)
				if !ok {
					t.Fatalf("args: %T", args)
				}
				if ig.Quality == nil || *ig.Quality != "high" {
					t.Errorf("Quality: %v", ig.Quality)
				}
			},
		},
		{
			name: "mcp",
			tool: MCP(openai.MCPArgs{
				ServerLabel: "x",
				ServerURL:   strPtr("https://mcp.test"),
			}),
			checkType: func(t *testing.T, args any) {
				m, ok := args.(openai.MCPArgs)
				if !ok {
					t.Fatalf("args: %T", args)
				}
				if m.ServerLabel != "x" {
					t.Errorf("ServerLabel: %q", m.ServerLabel)
				}
			},
		},
		{
			name: "tool_search",
			tool: ToolSearch(openai.ToolSearchArgs{
				Execution: strPtr("server"),
			}),
			checkType: func(t *testing.T, args any) {
				ts, ok := args.(openai.ToolSearchArgs)
				if !ok {
					t.Fatalf("args: %T", args)
				}
				if ts.Execution == nil || *ts.Execution != "server" {
					t.Errorf("Execution: %v", ts.Execution)
				}
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if c.tool.Args == nil {
				t.Fatal("Args is nil")
			}
			c.checkType(t, c.tool.Args)
		})
	}
}

func TestFactoriesWithNoArgs(t *testing.T) {
	if ApplyPatch().Args == nil {
		t.Error("apply_patch args should be set")
	}
	if LocalShell().Args == nil {
		t.Error("local_shell args should be set")
	}
	if CustomTool("x", &openai.CustomToolFormat{Type: "text"}).Args == nil {
		t.Error("custom args should be set")
	}
}

func TestShellArgsEnvironmentPropagates(t *testing.T) {
	env := &openai.ShellEnvironment{Type: "container_auto"}
	shell := Shell(openai.ShellArgs{Environment: env})
	args, ok := shell.Args.(openai.ShellArgs)
	if !ok {
		t.Fatalf("args: %T", shell.Args)
	}
	if args.Environment == nil {
		t.Fatal("Environment should not be nil")
	}
	if args.Environment.Type != "container_auto" {
		t.Errorf("Type: %q", args.Environment.Type)
	}
}
