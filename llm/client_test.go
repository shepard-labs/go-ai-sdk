package llm

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

func TestREQLLM001_ClientHasGenerateAndStream(t *testing.T) {
	clientType := reflect.TypeOf((*Client)(nil)).Elem()
	// spec §1.1: Client gains a Stream method alongside Generate.
	if clientType.NumMethod() != 2 {
		t.Fatalf("Client has %d methods, want 2", clientType.NumMethod())
	}
	gen, ok := clientType.MethodByName("Generate")
	if !ok {
		t.Fatal("Client missing Generate method")
	}
	wantGen := reflect.TypeOf(func(context.Context, GenerateOptions) (*GenerateResult, error) { return nil, nil })
	if gen.Type.NumIn() != wantGen.NumIn() || gen.Type.NumOut() != wantGen.NumOut() {
		t.Fatalf("Generate signature = %s, want Client.Generate(context.Context, GenerateOptions) (*GenerateResult, error)", gen.Type)
	}
	for i := 0; i < wantGen.NumIn(); i++ {
		if gen.Type.In(i) != wantGen.In(i) {
			t.Fatalf("Generate input %d = %s, want %s", i, gen.Type.In(i), wantGen.In(i))
		}
	}
	for i := 0; i < wantGen.NumOut(); i++ {
		if gen.Type.Out(i) != wantGen.Out(i) {
			t.Fatalf("Generate output %d = %s, want %s", i, gen.Type.Out(i), wantGen.Out(i))
		}
	}
	stream, ok := clientType.MethodByName("Stream")
	if !ok {
		t.Fatal("Client missing Stream method")
	}
	wantStream := reflect.TypeOf(func(context.Context, GenerateOptions) (<-chan StreamPart, error) { return nil, nil })
	for i := 0; i < wantStream.NumIn(); i++ {
		if stream.Type.In(i) != wantStream.In(i) {
			t.Fatalf("Stream input %d = %s, want %s", i, stream.Type.In(i), wantStream.In(i))
		}
	}
	for i := 0; i < wantStream.NumOut(); i++ {
		if stream.Type.Out(i) != wantStream.Out(i) {
			t.Fatalf("Stream output %d = %s, want %s", i, stream.Type.Out(i), wantStream.Out(i))
		}
	}
}

func TestREQLLM003_PublicTypesNoGoAISDKReexport(t *testing.T) {
	types := []reflect.Type{
		reflect.TypeOf(Message{}),
		reflect.TypeOf(TextContent{}),
		reflect.TypeOf(ToolUseContent{}),
		reflect.TypeOf(ToolResultContent{}),
		reflect.TypeOf(Tool{}),
		reflect.TypeOf(GenerateResult{}),
		reflect.TypeOf(Usage{}),
	}
	for _, typ := range types {
		if typ.PkgPath() != "github.com/shepard-labs/go-ai-sdk/llm" {
			t.Fatalf("%s package = %q", typ.Name(), typ.PkgPath())
		}
	}

	var contents []Content = []Content{
		TextContent{Text: "hello"},
		ToolUseContent{ID: "call", Name: "tool", Input: json.RawMessage(`{}`)},
		ToolResultContent{ToolUseID: "call", Text: "ok"},
	}
	if len(contents) != 3 {
		t.Fatalf("content implementations = %d, want 3", len(contents))
	}
}

// TestNewContentVariantsSatisfyContent verifies the new ReasoningContent and
// ImageContent types added in spec §1.2 satisfy the Content interface and live
// in the llm package.
func TestNewContentVariantsSatisfyContent(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeOf(ReasoningContent{}),
		reflect.TypeOf(ImageContent{}),
		reflect.TypeOf(ImageURLSource{}),
		reflect.TypeOf(ImageInlineSource{}),
	} {
		if typ.PkgPath() != "github.com/shepard-labs/go-ai-sdk/llm" {
			t.Fatalf("%s package = %q", typ.Name(), typ.PkgPath())
		}
	}
	var contents []Content = []Content{
		ReasoningContent{Text: "thinking"},
		ImageContent{Source: ImageURLSource{URL: "https://example.com/a.png"}, MIME: "image/png"},
		ImageContent{Source: ImageInlineSource{Data: []byte{0x89}}, MIME: "image/png"},
	}
	if len(contents) != 3 {
		t.Fatalf("new content implementations = %d, want 3", len(contents))
	}
}

func TestREQLLM004_ToolInputSchemaRawMessage(t *testing.T) {
	field, ok := reflect.TypeOf(Tool{}).FieldByName("InputSchema")
	if !ok {
		t.Fatal("Tool missing InputSchema")
	}
	if field.Type != reflect.TypeOf(json.RawMessage{}) {
		t.Fatalf("Tool.InputSchema type = %s, want json.RawMessage", field.Type)
	}
}
