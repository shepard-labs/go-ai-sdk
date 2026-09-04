package openai

import (
	"testing"
)

// TestResponsesWebSearchBlockedDomains verifies that web_search filters
// serialize allowedDomains/blockedDomains to the snake_case wire names
// allowed_domains/blocked_domains.
func TestResponsesWebSearchBlockedDomains(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.webSearch", Args: WebSearchArgs{
			Filters: &WebSearchFilters{
				AllowedDomains: []string{"example.com"},
				BlockedDomains: []string{"blocked.com"},
			},
		}},
	})
	tools := extractTools(t, body)
	ws, ok := tools["web_search"]
	if !ok {
		t.Fatalf("no web_search in tools: %v", tools)
	}
	filters, ok := ws["filters"].(map[string]any)
	if !ok {
		t.Fatalf("filters: %v", ws["filters"])
	}
	if got := stringSlice(filters["allowed_domains"]); len(got) != 1 || got[0] != "example.com" {
		t.Errorf("allowed_domains: %v", filters["allowed_domains"])
	}
	if got := stringSlice(filters["blocked_domains"]); len(got) != 1 || got[0] != "blocked.com" {
		t.Errorf("blocked_domains: %v", filters["blocked_domains"])
	}
	if _, has := filters["allowedDomains"]; has {
		t.Errorf("camelCase allowedDomains leaked to wire: %v", filters)
	}
	if _, has := filters["blockedDomains"]; has {
		t.Errorf("camelCase blockedDomains leaked to wire: %v", filters)
	}
}

// TestResponsesWebSearchBlockedDomainsMapArgs verifies the same mapping
// when Args arrive as a plain map (e.g. decoded JSON).
func TestResponsesWebSearchBlockedDomainsMapArgs(t *testing.T) {
	body := runResponsesWithTools(t, []Tool{
		{Type: "provider", ID: "openai.webSearch", Args: map[string]any{
			"filters": map[string]any{"blockedDomains": []any{"blocked.com"}},
		}},
	})
	tools := extractTools(t, body)
	ws, ok := tools["web_search"]
	if !ok {
		t.Fatalf("no web_search in tools: %v", tools)
	}
	filters, ok := ws["filters"].(map[string]any)
	if !ok {
		t.Fatalf("filters: %v", ws["filters"])
	}
	if got := stringSlice(filters["blocked_domains"]); len(got) != 1 || got[0] != "blocked.com" {
		t.Errorf("blocked_domains: %v", filters["blocked_domains"])
	}
}

func stringSlice(v any) []string {
	switch t := v.(type) {
	case []string:
		return t
	case []any:
		var out []string
		for _, x := range t {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
