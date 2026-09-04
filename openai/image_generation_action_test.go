package openai

import (
	"testing"
)

// TestProviderToolImageGenerationAction verifies the image_generation
// provider tool forwards `action` and passes through `input_image_mask`.
func TestProviderToolImageGenerationAction(t *testing.T) {
	fileID := "file-abc123"
	tests := []struct {
		name string
		args any
	}{
		{
			name: "typed struct",
			args: ImageGenerationArgs{
				Action: strP("edit"),
				Model:  strP("gpt-image-1"),
				InputImageMask: &ImageGenInputMask{
					FileID: &fileID,
				},
			},
		},
		{
			name: "map args",
			args: map[string]any{
				"action":         "edit",
				"model":          "gpt-image-1",
				"inputImageMask": map[string]any{"file_id": "file-abc123"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, warnings, err := buildResponsesProviderTool(Tool{
				Type: "provider",
				ID:   "openai.imageGeneration",
				Args: tt.args,
			})
			if err != nil {
				t.Fatalf("buildResponsesProviderTool: %v", err)
			}
			if len(warnings) != 0 {
				t.Errorf("warnings: %v", warnings)
			}
			if tool["type"] != "image_generation" {
				t.Errorf("type: %v", tool["type"])
			}
			if tool["action"] != "edit" {
				t.Errorf("action: %v", tool["action"])
			}
			mask, ok := tool["input_image_mask"]
			if !ok || mask == nil {
				t.Fatalf("input_image_mask missing: %v", tool)
			}
		})
	}
}
