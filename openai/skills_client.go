package openai

import (
	"context"
	"encoding/json"
	"mime/multipart"
)

// openaiSkillsClient implements Skills.
type openaiSkillsClient struct {
	provider *openaiProvider
}

func (c *openaiSkillsClient) UploadSkill(ctx context.Context, opts SkillsUploadOptions) (*SkillsUploadResult, error) {
	if c.provider.err != nil {
		return nil, c.provider.err
	}
	if len(opts.Files) == 0 {
		return nil, InvalidPromptError{Message: "SkillsUploadOptions.Files must contain at least one entry"}
	}
	build := func(w *multipart.Writer) error {
		for i, f := range opts.Files {
			filename := f.Path
			if filename == "" {
				filename = "skill.zip"
			}
			field := "files[]"
			if i == 0 {
				// The first file is sent as "files[]" too (form encoding), but
				// some servers use a different field name. Match the spec.
				field = "files[]"
			}
			part, err := w.CreateFormFile(field, filename)
			if err != nil {
				return err
			}
			if _, err := part.Write(f.Data); err != nil {
				return err
			}
		}
		return nil
	}
	resp, err := c.provider.executeMultipart(ctx, endpointSkills, opts.Headers, build)
	if err != nil {
		return nil, err
	}
	var raw skillAPIResponse
	if err := json.Unmarshal(resp.Body, &raw); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse skills response: " + err.Error()}
	}
	result := &SkillsUploadResult{
		ProviderReference: ProviderReference{"openai": raw.ID},
		ProviderMetadata: ProviderMetadata{
			"openai": map[string]any{
				"createdAt": raw.CreatedAt,
			},
		},
		Warnings: []Warning{},
	}
	if raw.Name != nil {
		result.Name = *raw.Name
	}
	if raw.Description != nil {
		result.Description = *raw.Description
	}
	if raw.LatestVersion != nil {
		result.LatestVersion = *raw.LatestVersion
	}
	pm := result.ProviderMetadata["openai"].(map[string]any)
	if raw.DefaultVersion != nil {
		pm["defaultVersion"] = *raw.DefaultVersion
	}
	if raw.UpdatedAt != nil {
		pm["updatedAt"] = *raw.UpdatedAt
	}
	if opts.DisplayTitle != "" {
		result.Warnings = append(result.Warnings, Warning{Type: "unsupported", Message: "displayTitle is not supported"})
	}
	return result, nil
}

type skillAPIResponse struct {
	ID             string  `json:"id"`
	Name           *string `json:"name"`
	Description    *string `json:"description"`
	DefaultVersion *string `json:"default_version"`
	LatestVersion  *string `json:"latest_version"`
	CreatedAt      int64   `json:"created_at"`
	UpdatedAt      *int64  `json:"updated_at"`
}
