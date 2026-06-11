package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/google"
)

const apiKey = "your-google-api-key"

func main() {
	provider := google.CreateGoogle(google.ProviderSettings{
		APIKey: apiKey,
	})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	// Replace this with a real file path. The Files API supports any media
	// type Gemini can ingest (PDF, images, audio, video).
	if len(os.Args) < 2 {
		log.Fatalf("usage: %s <path-to-file>\n", os.Args[0])
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatalf("failed to read %s: %v", os.Args[1], err)
	}

	files := provider.Files()
	result, err := files.Upload(context.Background(), data, google.FilesUploadOptions{
		MediaType: "application/pdf",
		ProviderOptions: google.ProviderOptions{
			"google": {
				"displayName": "example-upload",
			},
		},
	})
	if err != nil {
		log.Fatalf("Error uploading file: %v", err)
	}

	fmt.Printf("Upload complete.\n")
	fmt.Printf("  Media type: %s\n", result.MediaType)
	fmt.Printf("  Provider reference: %v\n", result.ProviderReference)
	if meta, ok := result.ProviderMetadata["google"]; ok {
		fmt.Printf("  Metadata: %+v\n", meta)
	}
}
