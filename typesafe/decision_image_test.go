package typesafe

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"
)

var (
	pngData  = []byte{137, 80, 78, 71}
	jpegData = []byte{255, 216, 255, 224}
)

func imageRecorder(t *testing.T) *recorder {
	t.Helper()
	return &recorder{respond: func(*http.Request) *http.Response {
		return jsonResponse(200, `{"model":"jev-latest","answers":{}}`)
	}}
}

func TestImageExplicitReference(t *testing.T) {
	for _, reference := range []string{"[animal-photo]", "__proto__", "constructor", "request"} {
		t.Run(reference, func(t *testing.T) {
			rc := imageRecorder(t)
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			questions := map[string]Question{
				"cat": BooleanQuestion{
					Instructions: ptr(StringInput("Does " + reference + " show a cat?")),
					Criteria:     &BooleanCriteria{True: ptr(JSONInput(map[string]any{"image": reference}))},
				},
			}
			result, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
				State: []StatePart{
					FileStatePart{Filename: "unrelated-name.png", MediaType: "image/png", Data: pngData, Reference: reference},
					FileStatePart{MediaType: "image/jpeg", Data: jpegData},
				},
				Questions: questions,
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Warnings) != 0 {
				t.Fatalf("warnings = %+v", result.Warnings)
			}
			req := rc.requests[0]
			body := multipartBody(t, req)
			assets, ok := body["assets"].(map[string]any)
			if !ok || len(assets) != 2 {
				t.Fatalf("assets = %v", body["assets"])
			}
			if _, ok := assets[reference]; !ok {
				t.Fatalf("assets missing explicit reference %q: %v", reference, assets)
			}
			var generated string
			for key := range assets {
				if key != reference {
					generated = key
				}
			}
			if body["state"] != reference+"\n"+generated {
				t.Fatalf("state = %v", body["state"])
			}
			wantAsset := map[string]any{"type": "image", "part": "image-0"}
			if !reflect.DeepEqual(assets[reference], wantAsset) {
				t.Fatalf("assets[%q] = %v", reference, assets[reference])
			}
			parts := parseMultipart(t, req.Header.Get("Content-Type"), req.Body)
			image0 := findPart(t, parts, "image-0")
			if image0.Filename != "unrelated-name.png" {
				t.Fatalf("image-0 filename = %q", image0.Filename)
			}
			wantQuestion := map[string]any{
				"type":         "noul",
				"instructions": "Does " + reference + " show a cat?",
				"criteria":     map[string]any{"true": map[string]any{"image": reference}},
			}
			if !reflect.DeepEqual(body["questions"].(map[string]any)["cat"], wantQuestion) {
				t.Fatalf("question cat = %v", body["questions"])
			}
		})
	}
}

func TestImageWhitespaceReferenceRejected(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State:     []StatePart{FileStatePart{MediaType: "image/png", Data: pngData, Reference: " \t\n"}},
		Questions: map[string]Question{},
	})
	if iaErr, ok := errors.AsType[*InvalidArgumentError](err); !ok {
		t.Fatalf("err = %v (%T)", err, err)
	} else if iaErr.Argument != "providerOptions.typesafe.reference" {
		t.Fatalf("argument = %q", iaErr.Argument)
	}
	if len(rc.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(rc.requests))
	}
}

func TestImageJSONStateAlreadyContainsReference(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			JSONStatePart{Value: json.RawMessage(`{"photo":"[photo]","note":"Box arrived wet."}`)},
			FileStatePart{MediaType: "image/png", Data: pngData, Reference: "[photo]"},
		},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := rc.requests[0]
	body := multipartBody(t, req)
	wantState := map[string]any{"photo": "[photo]", "note": "Box arrived wet."}
	if !reflect.DeepEqual(body["state"], wantState) {
		t.Fatalf("state = %v", body["state"])
	}
	wantAssets := map[string]any{"[photo]": map[string]any{"type": "image", "part": "image-1"}}
	if !reflect.DeepEqual(body["assets"], wantAssets) {
		t.Fatalf("assets = %v", body["assets"])
	}
	parts := parseMultipart(t, req.Header.Get("Content-Type"), req.Body)
	var names []string
	for _, part := range parts {
		names = append(names, part.Name)
	}
	if !slices.Equal(names, []string{"request", "image-1"}) {
		t.Fatalf("parts = %v", names)
	}
	if got := req.Header.Get("TypeSafe-Danger-Zone-Images"); got != "true" {
		t.Fatalf("images header = %q", got)
	}
}

func TestImageTextStateAlreadyContainsReference(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			TextStatePart{Text: "Is [photo] damaged?"},
			FileStatePart{MediaType: "image/jpeg", Data: jpegData, Reference: "[photo]"},
		},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := multipartBody(t, rc.requests[0])
	if body["state"] != "Is [photo] damaged?" {
		t.Fatalf("state = %v", body["state"])
	}
	assets := body["assets"].(map[string]any)
	if len(assets) != 1 {
		t.Fatalf("assets = %v", assets)
	}
	if _, ok := assets["[photo]"]; !ok {
		t.Fatalf("assets = %v", assets)
	}
}

func TestImageReferenceInNestedJSONValuesNotKeys(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	stateJSON := `{"frames":[{"seconds_ago":0,"frame":"Latest frame: [frame]"}],"[label]":"A key is not a reference."}`
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			JSONStatePart{Value: json.RawMessage(stateJSON)},
			FileStatePart{MediaType: "image/png", Data: pngData, Reference: "[frame]"},
			FileStatePart{MediaType: "image/png", Data: pngData, Reference: "[label]"},
		},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := multipartBody(t, rc.requests[0])
	want := stateJSON + "\n[label]"
	if body["state"] != want {
		t.Fatalf("state = %v\nwant %v", body["state"], want)
	}
}

func TestImageInsertReferenceFalse(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			JSONStatePart{Value: json.RawMessage(`{"photo":"[found]"}`)},
			FileStatePart{MediaType: "image/jpeg", Data: jpegData, Reference: "[found]"},
			FileStatePart{MediaType: "image/jpeg", Data: jpegData, Reference: "[poster]", InsertReference: ptr(false)},
		},
		Questions: map[string]Question{
			"same": BooleanQuestion{Instructions: ptr(StringInput("Is the dog in `photo` the dog in [poster]?"))},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := multipartBody(t, rc.requests[0])
	wantState := map[string]any{"photo": "[found]"}
	if !reflect.DeepEqual(body["state"], wantState) {
		t.Fatalf("state = %v", body["state"])
	}
	assets := body["assets"].(map[string]any)
	if len(assets) != 2 {
		t.Fatalf("assets = %v", assets)
	}
	for _, key := range []string{"[found]", "[poster]"} {
		if _, ok := assets[key]; !ok {
			t.Fatalf("assets missing %q: %v", key, assets)
		}
	}
}

func TestImageInsertReferenceTrueForces(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			TextStatePart{Text: "Before: [photo]"},
			FileStatePart{MediaType: "image/png", Data: pngData, Reference: "[photo]", InsertReference: ptr(true)},
		},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := multipartBody(t, rc.requests[0])
	if body["state"] != "Before: [photo]\n[photo]" {
		t.Fatalf("state = %v", body["state"])
	}
}

func TestImageInsertsOnlyMissingReferences(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			TextStatePart{Text: "Compare [before] with the new photo."},
			FileStatePart{MediaType: "image/png", Data: pngData, Reference: "[before]", InsertReference: ptr(false)},
			FileStatePart{MediaType: "image/png", Data: pngData, Reference: "[after]"},
		},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := multipartBody(t, rc.requests[0])
	if body["state"] != "Compare [before] with the new photo.\n[after]" {
		t.Fatalf("state = %v", body["state"])
	}
	wantAssets := map[string]any{
		"[before]": map[string]any{"type": "image", "part": "image-1"},
		"[after]":  map[string]any{"type": "image", "part": "image-2"},
	}
	if !reflect.DeepEqual(body["assets"], wantAssets) {
		t.Fatalf("assets = %v", body["assets"])
	}
}

func TestImageInsertFalseWithoutReference(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State:     []StatePart{FileStatePart{MediaType: "image/png", Data: pngData, InsertReference: ptr(false)}},
		Questions: map[string]Question{},
	})
	if iaErr, ok := errors.AsType[*InvalidArgumentError](err); !ok {
		t.Fatalf("err = %v (%T)", err, err)
	} else if iaErr.Argument != "providerOptions.typesafe.reference" {
		t.Fatalf("argument = %q", iaErr.Argument)
	}
	if len(rc.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(rc.requests))
	}
}

func TestImageDuplicateReferences(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			FileStatePart{MediaType: "image/png", Data: pngData, Reference: "[photo]"},
			FileStatePart{MediaType: "image/jpeg", Data: jpegData, Reference: "[photo]"},
		},
		Questions: map[string]Question{},
	})
	if iaErr, ok := errors.AsType[*InvalidArgumentError](err); !ok {
		t.Fatalf("err = %v (%T)", err, err)
	} else if iaErr.Message != "Duplicate TypeSafe image reference: [photo]" {
		t.Fatalf("message = %q", iaErr.Message)
	}
	if len(rc.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(rc.requests))
	}
}

func TestImageMixedStateMultipleImages(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			TextStatePart{Text: "Compare these images."},
			FileStatePart{MediaType: "image/png", Data: pngData},
			JSONStatePart{Value: json.RawMessage(`{"label":"second"}`)},
			FileStatePart{MediaType: "image/jpeg", Data: jpegData},
		},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := rc.requests[0]
	body := multipartBody(t, req)
	assets := body["assets"].(map[string]any)
	if len(assets) != 2 {
		t.Fatalf("assets = %v", assets)
	}
	refs := make([]string, 0, 2)
	for key := range assets {
		refs = append(refs, key)
	}
	slices.Sort(refs)
	if refs[0] == refs[1] {
		t.Fatalf("duplicate generated references: %v", refs)
	}
	byPart := map[string]string{}
	for ref, asset := range assets {
		byPart[asset.(map[string]any)["part"].(string)] = ref
	}
	wantState := fmt.Sprintf("Compare these images.\n%s\n{\"label\":\"second\"}\n%s", byPart["image-1"], byPart["image-3"])
	if body["state"] != wantState {
		t.Fatalf("state = %v\nwant %v", body["state"], wantState)
	}
	parts := parseMultipart(t, req.Header.Get("Content-Type"), req.Body)
	var names []string
	for _, part := range parts {
		names = append(names, part.Name)
	}
	if !slices.Equal(names, []string{"request", "image-1", "image-3"}) {
		t.Fatalf("parts = %v", names)
	}
	requestPart := findPart(t, parts, "request")
	if requestPart.ContentType != "application/json" {
		t.Fatalf("request content type = %q", requestPart.ContentType)
	}
	if requestPart.Filename != "request.json" {
		t.Fatalf("request filename = %q", requestPart.Filename)
	}
	image1 := findPart(t, parts, "image-1")
	if image1.ContentType != "image/png" || image1.Filename != "image-1.png" || !bytes.Equal(image1.Data, pngData) {
		t.Fatalf("image-1 = %+v data %v", image1, image1.Data)
	}
	image3 := findPart(t, parts, "image-3")
	if image3.ContentType != "image/jpeg" || image3.Filename != "image-3.jpeg" || !bytes.Equal(image3.Data, jpegData) {
		t.Fatalf("image-3 = %+v data %v", image3, image3.Data)
	}
	if ct := req.Header.Get("Content-Type"); !strings.HasPrefix(ct, "multipart/form-data; boundary=") {
		t.Fatalf("content-type = %q", ct)
	}
	if got := req.Header.Get("TypeSafe-Danger-Zone-Images"); got != "true" {
		t.Fatalf("images header = %q", got)
	}
}

func TestImageFilenameEscaping(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State: []StatePart{
			FileStatePart{Filename: `vacation "beach".png`, MediaType: "image/png", Data: pngData},
		},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := parseMultipart(t, rc.requests[0].Header.Get("Content-Type"), rc.requests[0].Body)
	image0 := findPart(t, parts, "image-0")
	if image0.Filename != `vacation "beach".png` {
		t.Fatalf("filename = %q", image0.Filename)
	}
}

func TestImageFilenameCRLFRejected(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	for _, filename := range []string{"a\r\nb.png", "a\nb.png", "a\rb.png"} {
		_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
			State:     []StatePart{FileStatePart{Filename: filename, MediaType: "image/png", Data: pngData}},
			Questions: map[string]Question{},
		})
		if _, ok := errors.AsType[*InvalidArgumentError](err); !ok {
			t.Fatalf("filename %q: err = %v (%T)", filename, err, err)
		}
	}
	if len(rc.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(rc.requests))
	}
}

func TestTextOnlyRequestOmitsAssetsAndImagesHeader(t *testing.T) {
	rc := newFixtureRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), fixtureOptions()); err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(rc.requests[0].Body, &body); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["assets"]; ok {
		t.Fatalf("unexpected assets: %v", body["assets"])
	}
	if _, ok := rc.requests[0].Header["Typesafe-Danger-Zone-Images"]; ok {
		t.Fatalf("unexpected images header: %v", rc.requests[0].Header)
	}
}

func TestImageFreshReferencesPerRequest(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	opts := DecisionCallOptions{
		State:     []StatePart{FileStatePart{MediaType: "image/png", Data: pngData}},
		Questions: map[string]Question{},
	}
	for range 2 {
		if _, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), opts); err != nil {
			t.Fatal(err)
		}
	}
	first := multipartBody(t, rc.requests[0])["state"]
	second := multipartBody(t, rc.requests[1])["state"]
	if first == second {
		t.Fatalf("references not fresh: %v", first)
	}
}

func TestImageUnsupportedMediaTypes(t *testing.T) {
	for _, mediaType := range []string{"application/pdf", "image/webp", "image/gif", "image"} {
		t.Run(mediaType, func(t *testing.T) {
			rc := imageRecorder(t)
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
				State:     []StatePart{FileStatePart{MediaType: mediaType, Data: []byte{0x41}}},
				Questions: map[string]Question{},
			})
			if _, ok := errors.AsType[*UnsupportedFunctionalityError](err); !ok {
				t.Fatalf("err = %v (%T)", err, err)
			}
			if len(rc.requests) != 0 {
				t.Fatalf("requests = %d, want 0", len(rc.requests))
			}
		})
	}
}

func TestImageWithoutInlineData(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State:     []StatePart{FileStatePart{MediaType: "image/png"}},
		Questions: map[string]Question{},
	})
	if _, ok := errors.AsType[*UnsupportedFunctionalityError](err); !ok {
		t.Fatalf("err = %v (%T)", err, err)
	}
	if len(rc.requests) != 0 {
		t.Fatalf("requests = %d, want 0", len(rc.requests))
	}
}

func TestImageGeneratedReference(t *testing.T) {
	rc := imageRecorder(t)
	p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc, GenerateID: func() string { return "generated-ref" }})
	_, err := p.DecisionModel("jev-latest").DoDecide(t.Context(), DecisionCallOptions{
		State:     []StatePart{FileStatePart{MediaType: "image/png", Data: pngData}},
		Questions: map[string]Question{},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := multipartBody(t, rc.requests[0])
	if body["state"] != "generated-ref" {
		t.Fatalf("state = %v", body["state"])
	}
	if _, ok := body["assets"].(map[string]any)["generated-ref"]; !ok {
		t.Fatalf("assets = %v", body["assets"])
	}
}

func TestBoxQuestion(t *testing.T) {
	for _, withImage := range []bool{false, true} {
		t.Run(fmt.Sprintf("withImage=%v", withImage), func(t *testing.T) {
			rc := &recorder{respond: func(*http.Request) *http.Response {
				return jsonResponse(200, `{"model":"jev-latest","answers":{"target":{"type":"box","box":{"top":0.25,"bottom":0.82,"left":0.44,"right":0.84},"confidence":0.92}}}`)
			}}
			p := CreateTypeSafeAI(ProviderSettings{APIKey: "test-api-key", Fetch: rc})
			opts := DecisionCallOptions{
				State: []StatePart{TextStatePart{Text: "A photo."}},
				Questions: map[string]Question{
					"target": BoxQuestion{Instructions: StringInput("The mug")},
				},
			}
			if withImage {
				opts.State = []StatePart{FileStatePart{MediaType: "image/png", Data: []byte{1, 2, 3}}}
			}
			model := p.DecisionModel("jev-latest")
			if !slices.Contains(model.SupportedQuestionTypes(), QuestionTypeBox) {
				t.Fatal("box not in supported question types")
			}
			result, err := model.DoDecide(t.Context(), opts)
			if err != nil {
				t.Fatal(err)
			}
			boxAnswer, ok := result.Answers["target"].(BoxAnswer)
			if !ok {
				t.Fatalf("answer = %#v", result.Answers["target"])
			}
			wantBox := Box{Top: 0.25, Bottom: 0.82, Left: 0.44, Right: 0.84}
			if boxAnswer.Box != wantBox {
				t.Fatalf("box = %+v", boxAnswer.Box)
			}
			confidence := result.ProviderMetadata["typesafe"]["confidence"].(map[string]float64)
			if confidence["target"] != 0.92 {
				t.Fatalf("confidence = %v", confidence)
			}
			req := rc.requests[0]
			if got := req.Header.Get("TypeSafe-Danger-Zone-Images"); got != "true" {
				t.Fatalf("images header = %q", got)
			}
			var question any
			if withImage {
				question = multipartBody(t, req)["questions"].(map[string]any)["target"]
			} else {
				var body map[string]any
				if err := json.Unmarshal(req.Body, &body); err != nil {
					t.Fatal(err)
				}
				question = body["questions"].(map[string]any)["target"]
			}
			wantQuestion := map[string]any{"type": "box", "instructions": "The mug"}
			if !reflect.DeepEqual(question, wantQuestion) {
				t.Fatalf("question = %v", question)
			}
		})
	}
}
