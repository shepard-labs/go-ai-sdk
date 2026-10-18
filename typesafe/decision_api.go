package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
)

// wireRequest is the POST /systemone request body.
type wireRequest struct {
	Model     string               `json:"model"`
	State     json.RawMessage      `json:"state"`
	Assets    map[string]wireAsset `json:"assets,omitempty"`
	Questions map[string]Question  `json:"questions"`
}

type wireAsset struct {
	Type string `json:"type"`
	Part string `json:"part"`
}

// wireResponse is the TypeSafe decision response envelope.
type wireResponse struct {
	Model   *string                    `json:"model"`
	Answers map[string]json.RawMessage `json:"answers"`
	Usage   *struct {
		InputTokens  *int64 `json:"input_tokens"`
		OutputTokens *int64 `json:"output_tokens"`
	} `json:"usage"`
}

type apiResult struct {
	body    []byte
	headers http.Header
}

// executeDecision performs exactly one HTTP attempt, JSON or multipart.
func (p *typeSafeProvider) executeDecision(ctx context.Context, body []byte, images []imageAsset, headers http.Header) (*apiResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var reqBody io.Reader
	contentType := "application/json"
	if len(images) > 0 {
		var buf bytes.Buffer
		w := multipart.NewWriter(&buf)
		requestHeader := textproto.MIMEHeader{}
		requestHeader.Set("Content-Disposition", `form-data; name="request"; filename="request.json"`)
		requestHeader.Set("Content-Type", "application/json")
		pw, err := w.CreatePart(requestHeader)
		if err != nil {
			return nil, err
		}
		if _, err := pw.Write(body); err != nil {
			return nil, err
		}
		for _, img := range images {
			h := textproto.MIMEHeader{}
			// Escape the filename for Content-Disposition; \r and \n are
			// rejected during state validation.
			h.Set("Content-Disposition", `form-data; name="`+img.name+`"; filename="`+escapeMultipartFilename(img.filename)+`"`)
			h.Set("Content-Type", img.mediaType)
			pw, err := w.CreatePart(h)
			if err != nil {
				return nil, err
			}
			if _, err := pw.Write(img.data); err != nil {
				return nil, err
			}
		}
		if err := w.Close(); err != nil {
			return nil, err
		}
		reqBody = &buf
		contentType = w.FormDataContentType()
	} else {
		reqBody = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/systemone", reqBody)
	if err != nil {
		return nil, err
	}
	req.Header = headers
	req.Header.Set("Content-Type", contentType)
	resp, err := p.fetch.Do(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, err
	}
	if resp == nil {
		return nil, &APICallError{Message: "fetcher returned nil response and nil error", Retryable: true}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		errBody, truncated, readErr := readLimited(resp.Body, p.maxErrorResponseBytes)
		apiErr := &APICallError{
			Message:   errorMessage(errBody),
			Status:    resp.StatusCode,
			Retryable: retryableStatus(resp.StatusCode),
			Headers:   cloneHeader(resp.Header),
			Body:      errBody,
			Truncated: truncated,
			Cause:     readErr,
		}
		return nil, apiErr
	}
	respBody, truncated, err := readLimited(resp.Body, p.maxResponseBodyBytes)
	if err != nil {
		return nil, err
	}
	responseHeaders := cloneHeader(resp.Header)
	if truncated {
		return nil, &APICallError{
			Message:   "response body exceeded maximum size",
			Status:    resp.StatusCode,
			Headers:   responseHeaders,
			Body:      respBody,
			Truncated: true,
		}
	}
	return &apiResult{body: respBody, headers: responseHeaders}, nil
}

// escapeMultipartFilename escapes a filename for use inside a quoted
// Content-Disposition parameter. Callers reject \r and \n up front.
func escapeMultipartFilename(name string) string {
	return multipartFilenameEscaper.Replace(name)
}

var multipartFilenameEscaper = strings.NewReplacer(`\`, `\\`, `"`, `\"`)

func readLimited(r io.Reader, limit int64) ([]byte, bool, error) {
	if limit <= 0 {
		limit = defaultMaxResponseBodyBytes
	}
	body, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) > limit {
		return body[:limit], true, nil
	}
	return body, false, nil
}

type decodedAnswer struct {
	answer     Answer
	confidence *float64
}

func decodeAnswer(raw json.RawMessage) (decodedAnswer, error) {
	invalid := func(msg string) (decodedAnswer, error) {
		return decodedAnswer{}, &InvalidResponseDataError{Message: msg, Data: string(raw)}
	}
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return invalid("malformed answer: " + err.Error())
	}
	switch head.Type {
	case "choice":
		var a struct {
			Choice        *string            `json:"choice"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    *float64           `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return invalid("malformed choice answer: " + err.Error())
		}
		if a.Choice == nil || a.Probabilities == nil {
			return invalid("choice answer is missing choice or probabilities")
		}
		return decodedAnswer{answer: ChoiceAnswer{Choice: *a.Choice, Probabilities: a.Probabilities}, confidence: a.Confidence}, nil
	case "score":
		var a struct {
			Score         *float64           `json:"score"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    *float64           `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return invalid("malformed score answer: " + err.Error())
		}
		if a.Score == nil || a.Probabilities == nil {
			return invalid("score answer is missing score or probabilities")
		}
		return decodedAnswer{answer: ScoreAnswer{Score: *a.Score, Probabilities: a.Probabilities}, confidence: a.Confidence}, nil
	case "box":
		var a struct {
			Box *struct {
				Top    *float64 `json:"top"`
				Bottom *float64 `json:"bottom"`
				Left   *float64 `json:"left"`
				Right  *float64 `json:"right"`
			} `json:"box"`
			Confidence *float64 `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return invalid("malformed box answer: " + err.Error())
		}
		if a.Box == nil || a.Box.Top == nil || a.Box.Bottom == nil || a.Box.Left == nil || a.Box.Right == nil {
			return invalid("box answer is missing box coordinates")
		}
		return decodedAnswer{answer: BoxAnswer{Box: Box{
			Top:    *a.Box.Top,
			Bottom: *a.Box.Bottom,
			Left:   *a.Box.Left,
			Right:  *a.Box.Right,
		}}, confidence: a.Confidence}, nil
	case "noul":
		var a struct {
			Noul *float64 `json:"noul"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return invalid("malformed noul answer: " + err.Error())
		}
		if a.Noul == nil {
			return invalid("noul answer is missing noul")
		}
		return decodedAnswer{answer: BooleanAnswer{Probability: *a.Noul}}, nil
	default:
		return invalid("unknown answer type: " + head.Type)
	}
}
