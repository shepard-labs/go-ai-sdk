// Command cache demonstrates llm.WithCache: wrapping a client with read-through
// response caching keyed by the request contents. Identical Generate calls hit
// the cache instead of the provider, saving latency and tokens. The wrapped
// client satisfies the same llm.Client interface, so call sites don't change.
// This example implements a minimal in-memory CacheBackend; a real one might be
// backed by Redis or another shared store.
package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/shepard-labs/go-ai-sdk/llm"
	"github.com/shepard-labs/go-ai-sdk/llm/registry"

	// Blank-import the provider adapter to register it with the registry.
	_ "github.com/shepard-labs/go-ai-sdk/llm/adapters/anthropic"
)

const apiKey = "sk-ant-api03-your-api-key"

// memCache is a minimal, concurrency-safe in-memory llm.CacheBackend. It tracks
// hits so the example can show caching actually happened.
type memCache struct {
	mu      sync.Mutex
	entries map[string]*llm.GenerateResult
	hits    int
}

func newMemCache() *memCache { return &memCache{entries: map[string]*llm.GenerateResult{}} }

func (c *memCache) Get(ctx context.Context, key string) (*llm.GenerateResult, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	result, ok := c.entries[key]
	if ok {
		c.hits++
	}
	return result, ok
}

func (c *memCache) Set(ctx context.Context, key string, result *llm.GenerateResult) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[key] = result
}

func main() {
	base, err := registry.NewClient("anthropic:claude-sonnet-4-6", registry.ProviderOptions{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	cache := newMemCache()
	client := llm.WithCache(base, cache)

	opts := llm.GenerateOptions{
		Messages: []llm.Message{{
			Role:    "user",
			Content: []llm.Content{llm.TextContent{Text: "Name a primary color."}},
		}},
		MaxTokens: 64,
	}

	// First call: cache miss — hits the provider and stores the result.
	// Second call: identical options, so it's served from the cache (no API call).
	for i := 1; i <= 2; i++ {
		start := time.Now()
		result, err := client.Generate(context.Background(), opts)
		if err != nil {
			log.Fatalf("generate %d: %v", i, err)
		}
		text := ""
		for _, content := range result.Content {
			if t, ok := content.(llm.TextContent); ok {
				text += t.Text
			}
		}
		fmt.Printf("call %d: %q (took %s, cache hits so far: %d)\n", i, text, time.Since(start).Round(time.Millisecond), cache.hits)
	}
}
