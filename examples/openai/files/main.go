package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/openai"
)

const apiKey = "your-openai-api-key"

func main() {
	if len(os.Args) < 2 {
		log.Fatalf("usage: %s <path-to-file>\n", os.Args[0])
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatalf("read file: %v", err)
	}

	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	result, err := provider.Files().UploadFile(context.Background(), openai.FilesUploadOptions{
		Data:      data,
		Filename:  os.Args[1],
		MediaType: "application/octet-stream",
		ProviderOptions: openai.ProviderOptions{
			"openai": {
				"purpose": "assistants",
			},
		},
	})
	if err != nil {
		log.Fatalf("Error uploading file: %v", err)
	}

	fmt.Println("Upload complete.")
	fmt.Printf("  Filename: %s\n", result.Filename)
	fmt.Printf("  Media type: %s\n", result.MediaType)
	fmt.Printf("  Provider reference: %v\n", result.ProviderReference)
	if meta, ok := result.ProviderMetadata["openai"]; ok {
		fmt.Printf("  Metadata: %+v\n", meta)
	}
}
