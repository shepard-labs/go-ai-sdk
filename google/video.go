package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/url"
	"strings"
	"time"

	"github.com/shepard-labs/go-ai-sdk/google/internal"
)

// googleVideoModel implements the Google Generative AI video model interface.
type googleVideoModel struct {
	provider *googleProvider
	modelID  string
}

// ModelID returns the model's ID string.
func (m *googleVideoModel) ModelID() string { return m.modelID }

// Provider returns the provider name suffix.
func (m *googleVideoModel) Provider() string { return m.provider.name + ".video" }

// MaxVideosPerCall returns the maximum number of videos per call (4).
func (m *googleVideoModel) MaxVideosPerCall() int { return 4 }

// DoGenerate performs a video generation call. It posts to :predictLongRunning
// and polls until the operation completes.
func (m *googleVideoModel) DoGenerate(ctx context.Context, opts VideoGenerateOptions) (*VideoGenerateResult, error) {
	if m.provider.err != nil {
		return nil, m.provider.err
	}

	rawOpts := cloneProviderOptions(opts.ProviderOptions)
	vertexLike := isVertexLike(m.provider.baseURL, m.provider.useVertexAIHeaders, rawOpts)
	merged := mergeGoogleNamespaces(rawOpts, vertexLike)
	videoOpts, recognizedKeys := videoModelOptionsFromProviderOptions(merged)

	var warnings []Warning
	if opts.Duration > 0 {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "duration",
			Details: "duration is not supported for video generation; use providerOptions.google.durationSeconds instead.",
		})
	}
	if opts.Image != nil && (opts.Image.Type == "url" || opts.Image.Type == "reference") {
		warnings = append(warnings, Warning{
			Type:    "unsupported",
			Feature: "image URL",
			Details: "image URL inputs are not supported; provide image data or base64.",
		})
	}

	// Build instances[0].
	instance := map[string]any{"prompt": opts.Prompt}

	// Add image as inlineData if provided (data type only).
	if opts.Image != nil && opts.Image.Type == "data" {
		mediaType := opts.Image.MediaType
		if mediaType == "" {
			mediaType = "image/png"
		}
		data := opts.Image.Base64
		if data == "" && len(opts.Image.Data) > 0 {
			data = base64.StdEncoding.EncodeToString(opts.Image.Data)
		}
		if data != "" {
			instance["image"] = map[string]any{
				"inlineData": map[string]any{
					"mimeType": mediaType,
					"data":     data,
				},
			}
		}
	}

	// Add referenceImages if provided via provider options.
	if len(videoOpts.ReferenceImages) > 0 {
		refImages := make([]map[string]any, len(videoOpts.ReferenceImages))
		for i, ri := range videoOpts.ReferenceImages {
			if ri.BytesBase64Encoded != "" {
				refImages[i] = map[string]any{
					"inlineData": map[string]any{
						"mimeType": "image/png",
						"data":     ri.BytesBase64Encoded,
					},
				}
			} else if ri.GcsUri != "" {
				refImages[i] = map[string]any{
					"gcsUri": ri.GcsUri,
				}
			}
		}
		instance["referenceImages"] = refImages
	}

	// Build parameters.
	parameters := map[string]any{}
	if opts.N > 0 {
		parameters["sampleCount"] = opts.N
	}
	if opts.AspectRatio != "" {
		parameters["aspectRatio"] = opts.AspectRatio
	}
	if opts.Resolution != "" {
		parameters["resolution"] = mapResolution(opts.Resolution)
	}
	if opts.Duration > 0 {
		parameters["durationSeconds"] = opts.Duration
	}
	if opts.Seed != nil {
		parameters["seed"] = *opts.Seed
	}
	if videoOpts.PersonGeneration != "" {
		parameters["personGeneration"] = videoOpts.PersonGeneration
	}
	if videoOpts.NegativePrompt != "" {
		parameters["negativePrompt"] = videoOpts.NegativePrompt
	}
	// Passthrough: forward unrecognized google namespace keys.
	recognized := map[string]struct{}{}
	for _, k := range recognizedKeys {
		recognized[k] = struct{}{}
	}
	for k, v := range merged {
		if _, ok := recognized[k]; ok {
			continue
		}
		if isVideoBodyReservedKey(k) {
			continue
		}
		parameters[k] = v
	}

	body := internal.APIVideoOperationRequest{
		Instances:  []map[string]any{instance},
		Parameters: parameters,
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	// POST :predictLongRunning
	postResp, err := m.provider.executeJSON(ctx, "/"+getModelPath(m.modelID)+":predictLongRunning", bodyBytes, cloneHeader(opts.Headers))
	if err != nil {
		return nil, err
	}

	var opResp internal.APIVideoOperationResponse
	if err := json.Unmarshal(postResp.Body, &opResp); err != nil {
		return nil, InvalidResponseDataError{Message: "failed to parse :predictLongRunning response", Data: string(postResp.Body)}
	}
	if opResp.Name == "" {
		return nil, &APICallError{
			Message:   "predictLongRunning response missing operation name",
			Type:      "GOOGLE_VIDEO_GENERATION_ERROR",
			Retryable: false,
			Status:    postResp.Status,
			Headers:   postResp.Headers,
			RequestID: postResp.RequestID,
			Body:      postResp.Body,
		}
	}

	// Poll until done.
	pollInterval := defaultPollIntervalMs
	if videoOpts.PollIntervalMs != nil {
		pollInterval = *videoOpts.PollIntervalMs
	}
	pollTimeout := defaultPollTimeoutMs
	if videoOpts.PollTimeoutMs != nil {
		pollTimeout = *videoOpts.PollTimeoutMs
	}

	deadline := time.NewTimer(time.Duration(pollTimeout) * time.Millisecond)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Duration(pollInterval) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline.C:
			return nil, &APICallError{
				Message:   "video generation timed out after " + formatMs(pollTimeout),
				Type:      "GOOGLE_VIDEO_GENERATION_TIMEOUT",
				Retryable: false,
			}
		case <-ticker.C:
			getResp, err := m.provider.executeGet(ctx, "/"+opResp.Name, cloneHeader(opts.Headers))
			if err != nil {
				return nil, err
			}
			var status internal.APIVideoOperationResponse
			if err := json.Unmarshal(getResp.Body, &status); err != nil {
				return nil, &APICallError{
					Message:   "failed to parse operation status response",
					Type:      "GOOGLE_VIDEO_GENERATION_ERROR",
					Retryable: false,
					Status:    getResp.Status,
					Headers:   getResp.Headers,
					RequestID: getResp.RequestID,
					Body:      getResp.Body,
				}
			}
			if status.Done {
				if status.Response == nil || len(status.Response.Predictions) == 0 {
					return nil, &APICallError{
						Message:   "operation completed but no generatedSamples returned",
						Type:      "GOOGLE_VIDEO_GENERATION_ERROR",
						Retryable: false,
						Status:    getResp.Status,
						Headers:   getResp.Headers,
						RequestID: getResp.RequestID,
						Body:      getResp.Body,
					}
				}
				uris := make([]string, len(status.Response.Predictions))
				for i, pred := range status.Response.Predictions {
					uris[i] = appendKey(pred.Video.URI, m.provider.apiKey)
				}
				return &VideoGenerateResult{
					Videos: uris,
					Warnings: warnings,
					ProviderMetadata: ProviderMetadata{
						"google": map[string]any{"operationName": opResp.Name},
					},
					Request:  RequestMetadata{Body: append([]byte(nil), bodyBytes...)},
					Response: responseMetadata(getResp.Headers, append([]byte(nil), getResp.Body...), "", m.modelID),
				}, nil
			}
		case <-ctx.Done():
			return nil, &APICallError{
				Message:   "video generation aborted",
				Type:      "GOOGLE_VIDEO_GENERATION_ABORTED",
				Retryable: false,
			}
		}
	}
}

// mapResolution converts pixel dimension strings to Google resolution labels.
func mapResolution(resolution string) string {
	switch resolution {
	case "1280x720":
		return "720p"
	case "1920x1080":
		return "1080p"
	case "3840x2160":
		return "4k"
	default:
		return resolution
	}
}

// appendKey appends ?key=<apiKey> (or &key= if uri already has a query) to uri.
func appendKey(uri, apiKey string) string {
	if uri == "" || apiKey == "" {
		return uri
	}
	u, err := url.Parse(uri)
	if err != nil {
		return uri
	}
	q := u.Query()
	q.Set("key", apiKey)
	u.RawQuery = q.Encode()
	return u.String()
}

// formatMs formats milliseconds as a human-readable string.
func formatMs(ms int) string {
	d := time.Duration(ms) * time.Millisecond
	s := d.String()
	s = strings.TrimSuffix(s, "0ms")
	s = strings.TrimSuffix(s, "0s")
	return s
}

// isVideoBodyReservedKey reports whether k is managed explicitly by this
// implementation and must not be spread from passthrough.
func isVideoBodyReservedKey(k string) bool {
	switch k {
	case "prompt", "image", "referenceImages", "sampleCount", "aspectRatio",
		"resolution", "durationSeconds", "seed", "personGeneration",
		"negativePrompt", "instances", "parameters":
		return true
	}
	return false
}

const defaultPollIntervalMs = 10_000
const defaultPollTimeoutMs = 600_000