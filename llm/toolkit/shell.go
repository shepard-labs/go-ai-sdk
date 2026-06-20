package toolkit

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

// defaultShellTimeout caps a single command's run time.
const defaultShellTimeout = 30 * time.Second

// ShellConfig configures the Shell toolkit.
type ShellConfig struct {
	// Cwd is the working directory for commands.
	Cwd string
	// AllowedCmds, if non-empty, is the allowlist of base command names that may
	// be run (e.g. "ls", "cat"). An empty list permits any command.
	AllowedCmds []string
	// Timeout bounds each command's run time. Defaults to 30s when zero.
	Timeout time.Duration
}

// shellToolkit runs shell commands.
//
// Security: when AllowedCmds is non-empty, only those base command names are
// permitted; any other command is rejected before execution. Commands run with
// the configured Cwd and are bounded by Timeout via exec.CommandContext, which
// kills the process when the deadline elapses. Arguments are passed directly to
// exec (no shell interpretation), so shell metacharacters are not expanded.
type shellToolkit struct {
	cwd     string
	allowed map[string]bool
	timeout time.Duration
	tools   []llm.Tool
}

type runCommandInput struct {
	Command string   `json:"command" description:"the base command to run"`
	Args    []string `json:"args" description:"arguments passed to the command"`
}

// Shell creates a shell toolkit.
func Shell(config ShellConfig) Toolkit {
	timeout := config.Timeout
	if timeout <= 0 {
		timeout = defaultShellTimeout
	}
	var allowed map[string]bool
	if len(config.AllowedCmds) > 0 {
		allowed = make(map[string]bool, len(config.AllowedCmds))
		for _, cmd := range config.AllowedCmds {
			allowed[cmd] = true
		}
	}
	tk := &shellToolkit{cwd: config.Cwd, allowed: allowed, timeout: timeout}
	tk.tools = mustTools(
		schemaTool("run_command", "Run a shell command and return its stdout, stderr, and exit code.", runCommandInput{}),
	)
	return tk
}

func (t *shellToolkit) Tools() []llm.Tool { return t.tools }

func (t *shellToolkit) Dispatch(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	if name != "run_command" {
		return nil, fmt.Errorf("toolkit: unknown shell tool %q", name)
	}
	var in runCommandInput
	if err := json.Unmarshal(input, &in); err != nil {
		return nil, err
	}
	if in.Command == "" {
		return nil, fmt.Errorf("run_command: empty command")
	}
	if t.allowed != nil && !t.allowed[in.Command] {
		return nil, fmt.Errorf("run_command: command %q is not allowed", in.Command)
	}

	ctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, in.Command, in.Args...)
	cmd.Dir = t.cwd
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	exitCode := 0
	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) {
		exitCode = exitErr.ExitCode()
	} else if runErr != nil {
		// Command could not be started (not found, timeout, etc.).
		return jsonResult(map[string]any{
			"stdout":    stdout.String(),
			"stderr":    stderr.String(),
			"exit_code": -1,
			"error":     runErr.Error(),
		})
	}
	return jsonResult(map[string]any{
		"stdout":    stdout.String(),
		"stderr":    stderr.String(),
		"exit_code": exitCode,
	})
}

// ---- Git ----

// GitConfig configures the read-only Git toolkit.
type GitConfig struct {
	// Root is the repository root; all git operations run here.
	Root string
}

// gitToolkit exposes read-only git inspection.
//
// Security: only git_status, git_diff, and git_log are exposed — there are no
// write or history-mutating operations. Every invocation runs git with the
// fixed Root as its working directory and a hardcoded read-only subcommand;
// no user input is interpolated into the subcommand selection.
type gitToolkit struct {
	root    string
	timeout time.Duration
	tools   []llm.Tool
}

type gitLogInput struct {
	MaxCount *int `json:"max_count" description:"maximum number of commits to return"`
}

type gitDiffInput struct {
	Staged *bool `json:"staged" description:"show staged changes instead of the working tree"`
}

// Git creates a read-only git toolkit scoped to the configured repo root.
func Git(config GitConfig) Toolkit {
	tk := &gitToolkit{root: config.Root, timeout: defaultShellTimeout}
	tk.tools = mustTools(
		schemaTool("git_status", "Show the working tree status (git status).", struct{}{}),
		schemaTool("git_diff", "Show changes in the working tree or staging area (git diff).", gitDiffInput{}),
		schemaTool("git_log", "Show recent commit history (git log).", gitLogInput{}),
	)
	return tk
}

func (t *gitToolkit) Tools() []llm.Tool { return t.tools }

func (t *gitToolkit) Dispatch(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	switch name {
	case "git_status":
		return t.run(ctx, "status", "--short", "--branch")
	case "git_diff":
		var in gitDiffInput
		if len(input) > 0 {
			if err := json.Unmarshal(input, &in); err != nil {
				return nil, err
			}
		}
		args := []string{"diff"}
		if in.Staged != nil && *in.Staged {
			args = append(args, "--staged")
		}
		return t.run(ctx, args...)
	case "git_log":
		var in gitLogInput
		if len(input) > 0 {
			if err := json.Unmarshal(input, &in); err != nil {
				return nil, err
			}
		}
		maxCount := 20
		if in.MaxCount != nil && *in.MaxCount > 0 {
			maxCount = *in.MaxCount
		}
		return t.run(ctx, "log", "--oneline", fmt.Sprintf("--max-count=%d", maxCount))
	default:
		return nil, fmt.Errorf("toolkit: unknown git tool %q", name)
	}
}

func (t *gitToolkit) run(ctx context.Context, args ...string) (json.RawMessage, error) {
	ctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = t.root
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, stderr.String())
	}
	return jsonResult(map[string]any{"output": stdout.String()})
}
