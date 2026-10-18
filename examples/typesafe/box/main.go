// Command box asks a BoxQuestion — locate a named region in an image and get
// back normalized top/bottom/left/right coordinates. Any BoxQuestion (or any
// FileStatePart) enables the TypeSafe-Danger-Zone-Images header.
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/box
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shepard-labs/go-ai-sdk/examples/typesafe/imgdata"
	"github.com/shepard-labs/go-ai-sdk/typesafe"
)

func main() {
	if os.Getenv("TYPESAFE_AI_API_KEY") == "" {
		log.Fatal("TYPESAFE_AI_API_KEY is not set")
	}
	// Box requires jev-1.14 or newer — jev-latest currently maps to 1.13 and
	// rejects Box questions with 400 Invalid request.
	model := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{}).DecisionModel("jev-1.14.0")

	r, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.FileStatePart{MediaType: "image/jpeg", Data: imgdata.PhotoJPEG},
		},
		Questions: map[string]typesafe.Question{
			"subject": typesafe.BoxQuestion{
				Instructions: typesafe.StringInput("the main subject of the photo"),
			},
			// Box questions compose with the other types.
			"isLandscape": typesafe.BooleanQuestion{
				Instructions: ptr(typesafe.StringInput("Is this a landscape photo?")),
			},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}
	if a, ok := r.Answers["subject"].(typesafe.BoxAnswer); ok {
		fmt.Printf("subject box: top=%.2f bottom=%.2f left=%.2f right=%.2f\n",
			a.Box.Top, a.Box.Bottom, a.Box.Left, a.Box.Right)
	}
	if a, ok := r.Answers["isLandscape"].(typesafe.BooleanAnswer); ok {
		fmt.Printf("isLandscape: P(true)=%.2f\n", a.Probability)
	}
	if c, ok := r.ProviderMetadata["typesafe"]["confidence"].(map[string]float64); ok {
		fmt.Println("confidence:", c)
	}
}

func ptr[T any](v T) *T { return &v }
