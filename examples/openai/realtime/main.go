package main

import (
	"context"
	"fmt"
	"log"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

func main() {
	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	rt := provider.ExperimentalRealtime().RealtimeModel("gpt-4o-realtime-preview")

	expires := 600
	secret, err := rt.DoCreateClientSecret(context.Background(), openai.ClientSecretOptions{
		ExpiresAfterSeconds: &expires,
		SessionConfig: &openai.SessionConfig{
			Instructions:     "You are a helpful voice assistant.",
			Voice:            "alloy",
			OutputModalities: []string{"audio", "text"},
		},
	})
	if err != nil {
		log.Fatalf("Error creating client secret: %v", err)
	}

	fmt.Println("Realtime client secret minted.")
	fmt.Printf("  WebSocket URL: %s\n", secret.URL)
	if secret.ExpiresAt != nil {
		fmt.Printf("  Expires at (unix): %d\n", *secret.ExpiresAt)
	}
	fmt.Printf("  Token prefix: %s...\n", truncate(secret.Token, 12))

	ws := rt.GetWebSocketConfig(openai.WebSocketConfigInput{
		Token: secret.Token,
		URL:   secret.URL,
	})
	fmt.Printf("\nWebSocket config:\n  URL: %s\n  Protocols: %v\n", ws.URL, ws.Protocols)

	// Helpers for a custom WebSocket client (no connection in this example).
	rawEvent := []byte(`{"type":"session.created","session_id":"sess_demo"}`)
	ev := rt.ParseServerEvent(rawEvent)
	fmt.Printf("\nParsed sample server event: type=%s session=%s\n", ev.Type, ev.SessionID)

	clientPayload, err := rt.SerializeClientEvent(openai.RealtimeClientEvent{
		Type:         openai.RealtimeClientResponseCreate,
		Instructions: "Say hello in one sentence.",
	})
	if err != nil {
		log.Fatalf("SerializeClientEvent: %v", err)
	}
	fmt.Printf("Sample client event JSON length: %d bytes\n", len(clientPayload))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
