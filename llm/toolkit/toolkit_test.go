package toolkit

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/shepard-labs/go-ai-sdk/llm"
)

func dispatch(t *testing.T, tk Toolkit, name string, input any) (json.RawMessage, error) {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	return tk.Dispatch(context.Background(), name, raw)
}

var _ Toolkit = Files(FilesConfig{})
var _ Toolkit = Shell(ShellConfig{})
var _ Toolkit = Git(GitConfig{})

// ---- Files ----

func TestFilesReadWriteList(t *testing.T) {
	root := t.TempDir()
	tk := Files(FilesConfig{Roots: []string{root}})

	if _, err := dispatch(t, tk, "write_file", writeFileInput{Path: filepath.Join(root, "a.txt"), Content: "hello"}); err != nil {
		t.Fatalf("write_file: %v", err)
	}
	raw, err := dispatch(t, tk, "read_file", readFileInput{Path: filepath.Join(root, "a.txt")})
	if err != nil {
		t.Fatalf("read_file: %v", err)
	}
	var read struct {
		Content   string `json:"content"`
		Truncated bool   `json:"truncated"`
	}
	json.Unmarshal(raw, &read)
	if read.Content != "hello" || read.Truncated {
		t.Fatalf("read = %#v", read)
	}

	raw, err = dispatch(t, tk, "list_dir", listDirInput{Path: root})
	if err != nil {
		t.Fatalf("list_dir: %v", err)
	}
	if !contains(string(raw), "a.txt") {
		t.Fatalf("list_dir output = %s", raw)
	}
}

func TestFilesMaxReadBytes(t *testing.T) {
	root := t.TempDir()
	tk := Files(FilesConfig{Roots: []string{root}, MaxReadBytes: 4})
	os.WriteFile(filepath.Join(root, "big.txt"), []byte("0123456789"), 0o644)
	raw, err := dispatch(t, tk, "read_file", readFileInput{Path: filepath.Join(root, "big.txt")})
	if err != nil {
		t.Fatalf("read_file: %v", err)
	}
	var read struct {
		Content   string `json:"content"`
		Truncated bool   `json:"truncated"`
	}
	json.Unmarshal(raw, &read)
	if read.Content != "0123" || !read.Truncated {
		t.Fatalf("read = %#v, want truncated 0123", read)
	}
}

func TestFilesRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "secret.txt")
	os.WriteFile(outside, []byte("secret"), 0o644)
	tk := Files(FilesConfig{Roots: []string{root}})

	if _, err := dispatch(t, tk, "read_file", readFileInput{Path: filepath.Join(root, "..", filepath.Base(filepath.Dir(outside)), "secret.txt")}); err == nil {
		t.Fatal("expected traversal rejection")
	}
	// Absolute path outside the root.
	if _, err := dispatch(t, tk, "read_file", readFileInput{Path: outside}); err == nil {
		t.Fatal("expected rejection of path outside root")
	}
	// Write outside root.
	if _, err := dispatch(t, tk, "write_file", writeFileInput{Path: outside, Content: "x"}); err == nil {
		t.Fatal("expected write rejection outside root")
	}
}

func TestFilesRejectsSymlinkEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks unreliable on windows")
	}
	root := t.TempDir()
	outsideDir := t.TempDir()
	os.WriteFile(filepath.Join(outsideDir, "secret.txt"), []byte("secret"), 0o644)
	// Symlink inside the root pointing outside it.
	link := filepath.Join(root, "escape")
	if err := os.Symlink(outsideDir, link); err != nil {
		t.Fatalf("symlink: %v", err)
	}
	tk := Files(FilesConfig{Roots: []string{root}})
	if _, err := dispatch(t, tk, "read_file", readFileInput{Path: filepath.Join(link, "secret.txt")}); err == nil {
		t.Fatal("expected symlink escape rejection")
	}
}

func TestFilesSearch(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "a.txt"), []byte("find me here"), 0o644)
	os.WriteFile(filepath.Join(root, "b.txt"), []byte("nothing"), 0o644)
	tk := Files(FilesConfig{Roots: []string{root}})
	raw, err := dispatch(t, tk, "search_files", searchFilesInput{Path: root, Pattern: "find me"})
	if err != nil {
		t.Fatalf("search_files: %v", err)
	}
	if !contains(string(raw), "a.txt") || contains(string(raw), "b.txt") {
		t.Fatalf("search output = %s", raw)
	}
}

// ---- Shell ----

func TestShellRunsAndCaptures(t *testing.T) {
	tk := Shell(ShellConfig{AllowedCmds: []string{"echo"}})
	raw, err := dispatch(t, tk, "run_command", runCommandInput{Command: "echo", Args: []string{"hi"}})
	if err != nil {
		t.Fatalf("run_command: %v", err)
	}
	var out struct {
		Stdout   string `json:"stdout"`
		ExitCode int    `json:"exit_code"`
	}
	json.Unmarshal(raw, &out)
	if out.Stdout != "hi\n" || out.ExitCode != 0 {
		t.Fatalf("out = %#v", out)
	}
}

func TestShellAllowlistRejects(t *testing.T) {
	tk := Shell(ShellConfig{AllowedCmds: []string{"echo"}})
	if _, err := dispatch(t, tk, "run_command", runCommandInput{Command: "rm", Args: []string{"-rf", "/"}}); err == nil {
		t.Fatal("expected allowlist rejection")
	}
}

func TestShellNonZeroExit(t *testing.T) {
	tk := Shell(ShellConfig{AllowedCmds: []string{"false"}})
	raw, err := dispatch(t, tk, "run_command", runCommandInput{Command: "false"})
	if err != nil {
		t.Fatalf("run_command: %v", err)
	}
	var out struct {
		ExitCode int `json:"exit_code"`
	}
	json.Unmarshal(raw, &out)
	if out.ExitCode == 0 {
		t.Fatalf("exit_code = %d, want non-zero", out.ExitCode)
	}
}

// ---- Git ----

func TestGitStatusAndLog(t *testing.T) {
	root := t.TempDir()
	for _, args := range [][]string{
		{"init"},
		{"config", "user.email", "t@e.st"},
		{"config", "user.name", "Tester"},
	} {
		runGit(t, root, args...)
	}
	os.WriteFile(filepath.Join(root, "f.txt"), []byte("x"), 0o644)
	runGit(t, root, "add", "f.txt")
	runGit(t, root, "commit", "-m", "init")

	tk := Git(GitConfig{Root: root})
	if _, err := dispatch(t, tk, "git_status", struct{}{}); err != nil {
		t.Fatalf("git_status: %v", err)
	}
	raw, err := dispatch(t, tk, "git_log", gitLogInput{})
	if err != nil {
		t.Fatalf("git_log: %v", err)
	}
	if !contains(string(raw), "init") {
		t.Fatalf("git_log output = %s", raw)
	}
}

// ---- Merge / Tools ----

func TestToolsAndMerge(t *testing.T) {
	root := t.TempDir()
	files := Files(FilesConfig{Roots: []string{root}})
	shell := Shell(ShellConfig{AllowedCmds: []string{"echo"}})
	all := Tools(files, shell)
	if len(all) != 5 {
		t.Fatalf("Tools count = %d, want 5", len(all))
	}
	dispatcher := Merge(files, shell)
	if _, err := dispatcher.Dispatch(context.Background(), "run_command", mustJSON(t, runCommandInput{Command: "echo", Args: []string{"ok"}})); err != nil {
		t.Fatalf("merged dispatch run_command: %v", err)
	}
	if _, err := dispatcher.Dispatch(context.Background(), "list_dir", mustJSON(t, listDirInput{Path: root})); err != nil {
		t.Fatalf("merged dispatch list_dir: %v", err)
	}
	if _, err := dispatcher.Dispatch(context.Background(), "nope", nil); err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestMergePanicsOnDuplicate(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on duplicate tool name")
		}
	}()
	root := t.TempDir()
	Merge(Files(FilesConfig{Roots: []string{root}}), Files(FilesConfig{Roots: []string{root}}))
}

var _ llm.ToolDispatcher = Merge()

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	tk := &gitToolkit{root: dir, timeout: defaultShellTimeout}
	if _, err := tk.run(context.Background(), args...); err != nil {
		// init/config/commit return output on stdout; tolerate benign stderr.
		if !contains(err.Error(), "nothing to commit") {
			t.Logf("git %v: %v", args, err)
		}
	}
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

func contains(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
