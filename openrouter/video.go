package openrouter

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"time"
)

type videoModel struct {
	provider *openRouterProvider
	modelID  string
	options  VideoOptions
}

func (m *videoModel) ModelID() string       { return m.modelID }
func (m *videoModel) Provider() string      { return "openrouter.video" }
func (m *videoModel) MaxVideosPerCall() int { return defaultMaxVideos }

func (m *videoModel) DoGenerate(ctx context.Context, opts VideoGenerateOptions) (*VideoGenerateResult, error) {
	warnings := []Warning{}
	if opts.N > 1 {
		warnings = append(warnings, Warning{Type: "unsupported-setting", Message: "OpenRouter returns one video per call"})
	}
	body := map[string]any{"model": m.modelID, "prompt": opts.Prompt}
	if opts.AspectRatio != "" {
		body["aspect_ratio"] = opts.AspectRatio
	}
	if opts.Resolution != "" {
		body["size"] = opts.Resolution
	}
	if opts.Duration != "" {
		body["duration"] = opts.Duration
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if m.options.GenerateAudio != nil {
		body["generate_audio"] = *m.options.GenerateAudio
	}
	if opts.Image != nil {
		body["frame_images"] = []map[string]any{{"image_url": videoImageURL(*opts.Image)}}
	}
	mergeBody(body, m.provider.extraBody)
	mergeBody(body, m.options.ExtraBody)
	mergeOpenRouterOptions(body, opts.ProviderOptions)
	var submit videoSubmitResponse
	_, h, err := m.provider.postJSON(ctx, "/videos", body, opts.Headers, &submit)
	if err != nil {
		return nil, err
	}
	pollInterval := m.options.PollInterval
	if pollInterval <= 0 {
		pollInterval = 2 * time.Second
	}
	maxPoll := m.options.MaxPollTime
	if maxPoll <= 0 {
		maxPoll = 10 * time.Minute
	}
	deadline := time.NewTimer(maxPoll)
	defer deadline.Stop()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		status, err := m.poll(ctx, submit.ID, opts.Headers)
		if err != nil {
			return nil, err
		}
		switch status.Status {
		case "completed":
			videos := make([]VideoData, 0, len(status.UnsignedURLs))
			for _, u := range status.UnsignedURLs {
				videos = append(videos, VideoData{Type: "url", URL: u, MediaType: "video/mp4"})
			}
			md := ProviderMetadata{"openrouter": map[string]any{"generationId": firstNonEmpty(status.GenerationID, submit.GenerationID), "cost": status.Cost}}
			return &VideoGenerateResult{Videos: videos, Warnings: warnings, ProviderMetadata: md, Response: ImageResponseMetadata{ID: submit.ID, Headers: h, Timestamp: time.Now()}}, nil
		case "failed", "dead", "cancelled", "expired":
			return nil, &APICallError{StatusCode: http.StatusInternalServerError, Message: "OpenRouter video generation failed with status " + status.Status, Retryable: false}
		default:
			select {
			case <-ticker.C:
				continue
			case <-deadline.C:
				return nil, &APICallError{StatusCode: http.StatusRequestTimeout, Message: "OpenRouter video generation timed out", Retryable: true}
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
}

func (m *videoModel) poll(ctx context.Context, id string, headers http.Header) (videoStatusResponse, error) {
	var out videoStatusResponse
	_, _, err := m.provider.getJSON(ctx, "/videos/"+id, headers, &out)
	return out, err
}
func videoImageURL(f VideoFile) string {
	media := f.MediaType
	if media == "" {
		media = "image/png"
	}
	switch f.Type {
	case "url":
		return f.URL
	case "base64":
		return "data:" + media + ";base64," + f.Base64
	default:
		return "data:" + media + ";base64," + base64.StdEncoding.EncodeToString(f.Data)
	}
}
func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

type videoSubmitResponse struct {
	ID           string `json:"id"`
	GenerationID string `json:"generation_id"`
	PollingURL   string `json:"polling_url"`
	Status       string `json:"status"`
}
type videoStatusResponse struct {
	ID           string   `json:"id"`
	GenerationID string   `json:"generation_id"`
	Status       string   `json:"status"`
	UnsignedURLs []string `json:"unsigned_urls"`
	Cost         float64  `json:"cost"`
}

var _ = json.Valid
