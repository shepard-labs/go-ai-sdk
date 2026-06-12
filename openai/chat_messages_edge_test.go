package openai

import (
	"net/url"
	"strings"
	"testing"
)

// TestConvertUserFileContentImageURL verifies an image FileContent with a
// *url.URL data source emits a type: image_url entry that carries the URL.
func TestConvertUserFileContentImageURL(t *testing.T) {
	m := newTestChatModel()
	u, _ := url.Parse("https://example.com/cat.png")
	out, err := m.convertUserFileContent(FileContent{
		Data:      u,
		MediaType: "image/png",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["type"] != "image_url" {
		t.Errorf("type: %v", out["type"])
	}
	imageURL, ok := out["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("image_url: %T", out["image_url"])
	}
	if imageURL["url"] != u.String() {
		t.Errorf("url: %v", imageURL["url"])
	}
}

// TestConvertUserFileContentImageData verifies an image FileContent with
// raw bytes gets base64-encoded into a data URL.
func TestConvertUserFileContentImageData(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:      []byte{0xff, 0xd8, 0xff},
		MediaType: "image/jpeg",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	imageURL, ok := out["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("image_url: %T", out["image_url"])
	}
	urlStr, _ := imageURL["url"].(string)
	if !strings.HasPrefix(urlStr, "data:image/jpeg;base64,") {
		t.Errorf("url prefix: %q", urlStr)
	}
}

// TestConvertUserFileContentImageWithDetail verifies that the
// imageDetail provider option is forwarded to image_url.detail.
func TestConvertUserFileContentImageWithDetail(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:      []byte{0x01},
		MediaType: "image/png",
		ProviderOptions: ProviderMetadata{
			"openai": map[string]any{"imageDetail": "low"},
		},
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	imageURL, _ := out["image_url"].(map[string]any)
	if imageURL["detail"] != "low" {
		t.Errorf("detail: %v", imageURL["detail"])
	}
}

// TestConvertUserFileContentAudioURL verifies that an audio FileContent
// whose Data is a *url.URL throws UnsupportedFunctionalityError (the
// OpenAI chat API does not support audio URLs, only base64 data).
func TestConvertUserFileContentAudioURL(t *testing.T) {
	m := newTestChatModel()
	u, _ := url.Parse("https://example.com/song.mp3")
	_, err := m.convertUserFileContent(FileContent{
		Data:      u,
		MediaType: "audio/mp3",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// TestConvertUserFileContentAudioWav verifies an audio/wav FileContent
// with raw bytes emits an input_audio entry with format: "wav".
func TestConvertUserFileContentAudioWav(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:      []byte{0x52, 0x49, 0x46, 0x46},
		MediaType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["type"] != "input_audio" {
		t.Errorf("type: %v", out["type"])
	}
	ia, ok := out["input_audio"].(map[string]any)
	if !ok {
		t.Fatalf("input_audio: %T", out["input_audio"])
	}
	if ia["format"] != "wav" {
		t.Errorf("format: %v", ia["format"])
	}
	if _, ok := ia["data"].(string); !ok {
		t.Errorf("data not a string: %T", ia["data"])
	}
}

// TestConvertUserFileContentAudioMp3 verifies an audio/mpeg (or mp3)
// FileContent with raw bytes emits format: "mp3".
func TestConvertUserFileContentAudioMp3(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:      []byte{0xff, 0xfb, 0x90},
		MediaType: "audio/mpeg",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	ia, _ := out["input_audio"].(map[string]any)
	if ia["format"] != "mp3" {
		t.Errorf("format: %v", ia["format"])
	}
}

// TestConvertUserFileContentAudioFormatAliases verifies the various
// audio MIME aliases all map to the right format string.
func TestConvertUserFileContentAudioFormatAliases(t *testing.T) {
	cases := []struct {
		mediaType string
		want      string
	}{
		{"audio/wav", "wav"},
		{"audio/wave", "wav"},
		{"audio/x-wav", "wav"},
		{"audio/mp3", "mp3"},
		{"audio/mpeg", "mp3"},
	}
	for _, c := range cases {
		t.Run(c.mediaType, func(t *testing.T) {
			m := newTestChatModel()
			out, err := m.convertUserFileContent(FileContent{
				Data:      []byte{0x00},
				MediaType: c.mediaType,
			})
			if err != nil {
				t.Fatalf("convert: %v", err)
			}
			ia, _ := out["input_audio"].(map[string]any)
			if ia["format"] != c.want {
				t.Errorf("format: %v, want %s", ia["format"], c.want)
			}
		})
	}
}

// TestConvertUserFileContentAudioUnsupportedMediaType verifies an audio
// media type other than wav/mp3 throws UnsupportedFunctionalityError.
func TestConvertUserFileContentAudioUnsupportedMediaType(t *testing.T) {
	m := newTestChatModel()
	_, err := m.convertUserFileContent(FileContent{
		Data:      []byte{0x00},
		MediaType: "audio/ogg",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// TestConvertUserFileContentPDFURL verifies that a PDF FileContent whose
// Data is a *url.URL throws UnsupportedFunctionalityError.
func TestConvertUserFileContentPDFURL(t *testing.T) {
	m := newTestChatModel()
	u, _ := url.Parse("https://example.com/doc.pdf")
	_, err := m.convertUserFileContent(FileContent{
		Data:      u,
		MediaType: "application/pdf",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// TestConvertUserFileContentPDFData verifies a PDF FileContent with raw
// bytes emits a type: file entry with filename and file_data fields.
func TestConvertUserFileContentPDFData(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:     []byte("%PDF-1.4"),
		Filename: "doc.pdf",
		MediaType: "application/pdf",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["type"] != "file" {
		t.Errorf("type: %v", out["type"])
	}
	file, ok := out["file"].(map[string]any)
	if !ok {
		t.Fatalf("file: %T", out["file"])
	}
	if file["filename"] != "doc.pdf" {
		t.Errorf("filename: %v", file["filename"])
	}
	data, _ := file["file_data"].(string)
	if !strings.HasPrefix(data, "data:application/pdf;base64,") {
		t.Errorf("file_data prefix: %q", data)
	}
}

// TestConvertUserFileContentPDFDefaultFilename verifies a PDF without an
// explicit filename falls back to "part.pdf".
func TestConvertUserFileContentPDFDefaultFilename(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:      []byte("%PDF-1.4"),
		MediaType: "application/pdf",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	file, _ := out["file"].(map[string]any)
	if file["filename"] != "part.pdf" {
		t.Errorf("default filename: %v", file["filename"])
	}
}

// TestConvertUserFileContentTextFilePartUnsupported verifies text file
// parts (text/plain, text/html, etc.) are rejected.
func TestConvertUserFileContentTextFilePartUnsupported(t *testing.T) {
	m := newTestChatModel()
	_, err := m.convertUserFileContent(FileContent{
		Data:      []byte("hello"),
		MediaType: "text/plain",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// TestConvertUserFileContentUnknownMediaType verifies an unrecognized
// media type throws UnsupportedFunctionalityError.
func TestConvertUserFileContentUnknownMediaType(t *testing.T) {
	m := newTestChatModel()
	_, err := m.convertUserFileContent(FileContent{
		Data:      []byte{0x00},
		MediaType: "application/zip",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(UnsupportedFunctionalityError); !ok {
		t.Errorf("expected UnsupportedFunctionalityError, got %T: %v", err, err)
	}
}

// TestConvertUserFileContentReferenceRequiresOption verifies that
// Data: "reference" without a provider option is rejected with
// InvalidPromptError.
func TestConvertUserFileContentReferenceRequiresOption(t *testing.T) {
	m := newTestChatModel()
	_, err := m.convertUserFileContent(FileContent{
		Data:      "reference",
		MediaType: "image/png",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestConvertUserFileContentReferenceResolvesFileID verifies that
// Data: "reference" with an openai.reference option resolves to the
// file_id wire form.
func TestConvertUserFileContentReferenceResolvesFileID(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:      "reference",
		MediaType: "image/png",
		ProviderOptions: ProviderMetadata{
			"openai": map[string]any{"reference": "file-abc123"},
		},
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if out["type"] != "file" {
		t.Errorf("type: %v", out["type"])
	}
	file, _ := out["file"].(map[string]any)
	if file["file_id"] != "file-abc123" {
		t.Errorf("file_id: %v", file["file_id"])
	}
}

// TestConvertUserFileContentImageStarMediaType verifies the
// "image/*" media type falls back to image/jpeg when data is bytes.
func TestConvertUserFileContentImageStarMediaType(t *testing.T) {
	m := newTestChatModel()
	out, err := m.convertUserFileContent(FileContent{
		Data:      []byte{0xff, 0xd8},
		MediaType: "image/*",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	imageURL, _ := out["image_url"].(map[string]any)
	urlStr, _ := imageURL["url"].(string)
	if !strings.HasPrefix(urlStr, "data:image/jpeg;base64,") {
		t.Errorf("url prefix: %q", urlStr)
	}
}

// TestConvertUserContentUnknownType verifies that an unknown user
// content type throws InvalidPromptError.
func TestConvertUserContentUnknownType(t *testing.T) {
	m := newTestChatModel()
	type bogus struct{ UserContent }
	_, err := m.convertUserContent(bogus{})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestBase64DataUnsupportedType verifies an unsupported Data type (e.g.
// int) throws InvalidPromptError.
func TestBase64DataUnsupportedType(t *testing.T) {
	_, err := base64Data(42)
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}

// TestFileDataURLWithURL verifies fileDataURL with a *url.URL passes the
// string through unchanged.
func TestFileDataURLWithURL(t *testing.T) {
	u, _ := url.Parse("https://example.com/x.jpg")
	got, err := fileDataURL(u, "image/jpeg")
	if err != nil {
		t.Fatalf("fileDataURL: %v", err)
	}
	if got != u.String() {
		t.Errorf("got %q, want %q", got, u.String())
	}
}

// TestFileDataURLWithBytesStarType verifies that bytes with the image/*
// wildcard fall back to image/jpeg in the data URL.
func TestFileDataURLWithBytesStarType(t *testing.T) {
	got, err := fileDataURL([]byte{0xff, 0xd8}, "image/*")
	if err != nil {
		t.Fatalf("fileDataURL: %v", err)
	}
	if !strings.HasPrefix(got, "data:image/jpeg;base64,") {
		t.Errorf("prefix: %q", got)
	}
}

// TestFileDataURLUnsupportedType verifies an unsupported data type
// throws InvalidPromptError.
func TestFileDataURLUnsupportedType(t *testing.T) {
	_, err := fileDataURL(42, "image/png")
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(InvalidPromptError); !ok {
		t.Errorf("expected InvalidPromptError, got %T: %v", err, err)
	}
}
