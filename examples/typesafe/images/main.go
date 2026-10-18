// Command images shows how FileStatePart embeds inline PNG/JPEG data, how the
// wire request switches to multipart, and how Reference / InsertReference
// control where the asset is anchored in state.
//
//	TYPESAFE_AI_API_KEY=ts-... go run ./examples/typesafe/images
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
	// jev-1.14.0 is the first release that accepts image input; jev-latest
	// currently maps to jev-1.13 which rejects it.
	model := typesafe.CreateTypeSafeAI(typesafe.ProviderSettings{}).DecisionModel("jev-1.14.0")

	// Inline image: the SDK switches to multipart, uploads the JPEG as a named
	// asset, and places the generated reference into state automatically.
	r, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.FileStatePart{
				MediaType: "image/jpeg",
				Data:      imgdata.PhotoJPEG,
				Filename:  "photo.jpg",
				// Reference: "mug-image", // leave empty → a crypto-random ref is generated
				// InsertReference: ptr(true), // nil = auto-insert unless ref already in state
			},
		},
		Questions: map[string]typesafe.Question{
			"isOutdoors": typesafe.BooleanQuestion{
				Instructions: ptr(typesafe.StringInput("Is this an outdoor photograph?")),
			},
			"subject": typesafe.ChoiceQuestion{
				Instructions: ptr(typesafe.StringInput("What is the main subject?")),
				Criteria: map[string]*typesafe.Input{
					"mountain": ptr(typesafe.StringInput("Mountains or hills")),
					"water":    ptr(typesafe.StringInput("Lake, ocean, or river")),
					"sky":      ptr(typesafe.StringInput("Sky, clouds, or sunset")),
					"other":    ptr(typesafe.NullInput()),
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("decide: %v", err)
	}
	for id, a := range r.Answers {
		fmt.Printf("%s: %#v\n", id, a)
	}

	// Explicit Reference lets instructions point at the asset. The SDK
	// auto-inserts the reference into state text (InsertReference: nil) unless
	// the text already contains it.
	r2, err := model.DoDecide(context.Background(), typesafe.DecisionCallOptions{
		State: []typesafe.StatePart{
			typesafe.FileStatePart{
				MediaType: "image/jpeg",
				Data:      imgdata.PhotoJPEG,
				Reference: "photo-1", // named ref — goes into state as "photo-1"
			},
		},
		Questions: map[string]typesafe.Question{
			"match": typesafe.BooleanQuestion{Instructions: ptr(typesafe.StringInput("Does photo-1 show an outdoor scene?"))},
		},
	})
	if err != nil {
		log.Fatalf("decide (explicit ref): %v", err)
	}
	if a, ok := r2.Answers["match"].(typesafe.BooleanAnswer); ok {
		fmt.Printf("explicit ref → match: P(true)=%.2f\n", a.Probability)
	}
}

func ptr[T any](v T) *T { return &v }
