package toolkit

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/schema"
)

// defaultMaxReadBytes caps read_file / search_files output at 1 MiB.
const defaultMaxReadBytes = 1 << 20

// defaultMaxSearchResults caps the number of matches returned by search_files.
const defaultMaxSearchResults = 500

// FilesConfig configures the Files toolkit.
type FilesConfig struct {
	// Roots are the allowed root directories. Every path argument must resolve
	// (after symlink evaluation) to a location inside one of these roots;
	// attempts to escape via "../" or symlinks are rejected. Roots must be
	// non-empty: Files returns an error when no roots are configured.
	Roots []string
	// MaxReadBytes caps the bytes returned by read_file and per-file matches in
	// search_files. Defaults to 1 MiB when zero.
	MaxReadBytes int
	// MaxSearchResults caps the number of matches search_files returns.
	// Defaults to 500 when zero.
	MaxSearchResults int
	// ReadOnly, when true, omits write_file from Tools() and rejects any
	// write_file dispatch. Use it to expose a strictly read-only filesystem view.
	ReadOnly bool
}

// filesToolkit implements filesystem tools scoped to a set of roots.
//
// Security: all path arguments are resolved with filepath.Abs and
// filepath.EvalSymlinks and then prefix-checked against the cleaned, symlink-
// resolved roots. A path that resolves outside every root — whether through
// "../" traversal or a symlink pointing elsewhere — is rejected before any I/O
// touches the target. Symlinks are followed but the resolved path must remain
// within a configured root. read_file and search_files enforce MaxReadBytes,
// and search_files additionally caps results at MaxSearchResults.
type filesToolkit struct {
	roots            []string
	maxReadBytes     int
	maxSearchResults int
	readOnly         bool
	tools            []llm.Tool
}

type readFileInput struct {
	Path string `json:"path" description:"path to the file to read"`
}

type writeFileInput struct {
	Path    string `json:"path" description:"path to the file to write"`
	Content string `json:"content" description:"content to write to the file"`
}

type listDirInput struct {
	Path string `json:"path" description:"directory to list"`
}

type searchFilesInput struct {
	Path    string `json:"path" description:"directory to search under"`
	Pattern string `json:"pattern" description:"substring to search for in file contents"`
}

// Files creates a filesystem toolkit scoped to the configured roots. It returns
// an error when Roots is empty, since an unrooted filesystem toolkit would have
// no boundary to enforce.
func Files(config FilesConfig) (Toolkit, error) {
	if len(config.Roots) == 0 {
		return nil, fmt.Errorf("toolkit: Files requires at least one root")
	}
	maxRead := config.MaxReadBytes
	if maxRead <= 0 {
		maxRead = defaultMaxReadBytes
	}
	maxResults := config.MaxSearchResults
	if maxResults <= 0 {
		maxResults = defaultMaxSearchResults
	}
	roots := make([]string, 0, len(config.Roots))
	for _, root := range config.Roots {
		if resolved, err := resolveExisting(root); err == nil {
			roots = append(roots, resolved)
		} else if abs, err := filepath.Abs(root); err == nil {
			roots = append(roots, filepath.Clean(abs))
		}
	}
	tk := &filesToolkit{roots: roots, maxReadBytes: maxRead, maxSearchResults: maxResults, readOnly: config.ReadOnly}
	tools := []toolResult{
		schemaTool("read_file", "Read the contents of a file.", readFileInput{}),
		schemaTool("list_dir", "List the entries of a directory.", listDirInput{}),
		schemaTool("search_files", "Search files under a directory for a substring.", searchFilesInput{}),
	}
	if !config.ReadOnly {
		tools = append(tools, schemaTool("write_file", "Write content to a file, creating or overwriting it.", writeFileInput{}))
	}
	tk.tools = mustTools(tools...)
	return tk, nil
}

func (t *filesToolkit) Tools() []llm.Tool { return t.tools }

func (t *filesToolkit) Dispatch(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	switch name {
	case "read_file":
		return t.readFile(input)
	case "write_file":
		if t.readOnly {
			return nil, fmt.Errorf("write_file: toolkit is read-only")
		}
		return t.writeFile(input)
	case "list_dir":
		return t.listDir(input)
	case "search_files":
		return t.searchFiles(input)
	default:
		return nil, fmt.Errorf("toolkit: unknown files tool %q", name)
	}
}

func (t *filesToolkit) readFile(input json.RawMessage) (json.RawMessage, error) {
	var in readFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return nil, err
	}
	path, err := t.resolveInRoot(in.Path)
	if err != nil {
		return nil, err
	}
	if info, err := os.Stat(path); err == nil && info.Size() > int64(t.maxReadBytes) {
		return nil, fmt.Errorf("read_file: file size %d exceeds limit %d bytes", info.Size(), t.maxReadBytes)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read_file: %w", err)
	}
	if len(data) > t.maxReadBytes {
		return nil, fmt.Errorf("read_file: file size %d exceeds limit %d bytes", len(data), t.maxReadBytes)
	}
	return jsonResult(map[string]any{"content": string(data)})
}

func (t *filesToolkit) writeFile(input json.RawMessage) (json.RawMessage, error) {
	var in writeFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return nil, err
	}
	// Resolve the parent directory: the target file may not exist yet, so its
	// own symlink resolution would fail.
	path, err := t.resolveNewInRoot(in.Path)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, []byte(in.Content), 0o644); err != nil {
		return nil, fmt.Errorf("write_file: %w", err)
	}
	return jsonResult(map[string]any{"bytes_written": len(in.Content)})
}

func (t *filesToolkit) listDir(input json.RawMessage) (json.RawMessage, error) {
	var in listDirInput
	if err := json.Unmarshal(input, &in); err != nil {
		return nil, err
	}
	path, err := t.resolveInRoot(in.Path)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, fmt.Errorf("list_dir: %w", err)
	}
	listed := make([]map[string]any, 0, len(entries))
	for _, entry := range entries {
		listed = append(listed, map[string]any{"name": entry.Name(), "is_dir": entry.IsDir()})
	}
	return jsonResult(map[string]any{"entries": listed})
}

func (t *filesToolkit) searchFiles(input json.RawMessage) (json.RawMessage, error) {
	var in searchFilesInput
	if err := json.Unmarshal(input, &in); err != nil {
		return nil, err
	}
	if in.Pattern == "" {
		return nil, fmt.Errorf("search_files: empty pattern")
	}
	root, err := t.resolveInRoot(in.Path)
	if err != nil {
		return nil, err
	}
	var matches []string
	truncated := false
	walkErr := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil || info.Size() > int64(t.maxReadBytes) {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		if strings.Contains(string(data), in.Pattern) {
			matches = append(matches, path)
			if len(matches) >= t.maxSearchResults {
				truncated = true
				return filepath.SkipAll
			}
		}
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("search_files: %w", walkErr)
	}
	return jsonResult(map[string]any{"matches": matches, "truncated": truncated})
}

// resolveInRoot resolves an existing path and verifies it lies within a root.
func (t *filesToolkit) resolveInRoot(path string) (string, error) {
	resolved, err := resolveExisting(path)
	if err != nil {
		return "", fmt.Errorf("toolkit: resolve %q: %w", path, err)
	}
	if !t.withinRoot(resolved) {
		return "", fmt.Errorf("toolkit: path %q escapes allowed roots", path)
	}
	return resolved, nil
}

// resolveNewInRoot resolves a path whose final element may not exist yet by
// resolving its parent directory's symlinks and checking the result is rooted.
func (t *filesToolkit) resolveNewInRoot(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	parent, err := resolveExisting(filepath.Dir(abs))
	if err != nil {
		return "", fmt.Errorf("toolkit: resolve parent of %q: %w", path, err)
	}
	full := filepath.Join(parent, filepath.Base(abs))
	if !t.withinRoot(full) {
		return "", fmt.Errorf("toolkit: path %q escapes allowed roots", path)
	}
	return full, nil
}

func (t *filesToolkit) withinRoot(resolved string) bool {
	for _, root := range t.roots {
		if resolved == root {
			return true
		}
		if strings.HasPrefix(resolved, root+string(os.PathSeparator)) {
			return true
		}
	}
	return false
}

// resolveExisting returns the absolute, symlink-evaluated form of an existing
// path.
func resolveExisting(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(abs)
}

// schemaTool builds a tool schema, returning the tool and any build error
// paired for mustTools.
func schemaTool(name, description string, input any) toolResult {
	tool, err := schema.Tool(name, description, input)
	return toolResult{tool: tool, err: err}
}

// toolResult pairs a built tool with its build error.
type toolResult struct {
	tool llm.Tool
	err  error
}

// mustTools collects tool/err pairs, panicking if any schema failed to build.
// Toolkit schemas are derived from fixed internal structs, so a failure is a
// programming error.
func mustTools(pairs ...toolResult) []llm.Tool {
	tools := make([]llm.Tool, 0, len(pairs))
	for _, pair := range pairs {
		if pair.err != nil {
			panic(fmt.Sprintf("toolkit: build tool schema: %v", pair.err))
		}
		tools = append(tools, pair.tool)
	}
	return tools
}
