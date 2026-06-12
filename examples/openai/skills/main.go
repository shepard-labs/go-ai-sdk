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
		log.Fatalf("usage: %s <path-to-skill.zip>\n", os.Args[0])
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatalf("read file: %v", err)
	}

	provider := openai.CreateOpenAI(openai.ProviderSettings{APIKey: apiKey})
	if err := provider.Err(); err != nil {
		log.Fatalf("Error creating provider: %v", err)
	}

	result, err := provider.Skills().UploadSkill(context.Background(), openai.SkillsUploadOptions{
		Files: []openai.SkillsFile{{
			Path:      os.Args[1],
			Data:      data,
			MediaType: "application/zip",
		}},
	})
	if err != nil {
		log.Fatalf("Error uploading skill: %v", err)
	}

	fmt.Println("Skill upload complete.")
	fmt.Printf("  Name: %s\n", result.Name)
	fmt.Printf("  Description: %s\n", result.Description)
	fmt.Printf("  Latest version: %s\n", result.LatestVersion)
	fmt.Printf("  Provider reference: %v\n", result.ProviderReference)
	for _, w := range result.Warnings {
		fmt.Printf("  Warning: %s\n", w.Message)
	}
}
