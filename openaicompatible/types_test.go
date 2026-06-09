package openaicompatible

import "testing"

func TestPublicTypesAndMarkers(t *testing.T) {
	var _ Message = SystemMessage{}
	var _ Message = UserMessage{}
	var _ Message = AssistantMessage{}
	var _ Message = ToolMessage{}
	var _ UserContent = TextContent{}
	var _ UserContent = FileContent{}
	var _ AssistantContent = TextContent{}
	var _ AssistantContent = ReasoningContent{}
	var _ AssistantContent = ToolCallContent{}
	var _ ToolContent = ToolResultContent{}
	var _ Content = TextContent{}
	var _ Content = ReasoningContent{}
	var _ Content = ToolCallContent{}
	var _ StreamPart = StreamStart{}
	var _ StreamPart = StreamResponseMetadata{}
	var _ StreamPart = StreamTextStart{}
	var _ StreamPart = StreamTextDelta{}
	var _ StreamPart = StreamTextEnd{}
	var _ StreamPart = StreamReasoningStart{}
	var _ StreamPart = StreamReasoningDelta{}
	var _ StreamPart = StreamReasoningEnd{}
	var _ StreamPart = StreamToolInputStart{}
	var _ StreamPart = StreamToolInputDelta{}
	var _ StreamPart = StreamToolInputEnd{}
	var _ StreamPart = StreamToolCall{}
	var _ StreamPart = StreamFinish{}
	var _ StreamPart = StreamError{}
	var _ StreamPart = StreamRaw{}
	var _ Delta = StreamTextDelta{}
	var _ Delta = StreamReasoningDelta{}
	var _ Delta = StreamToolInputDelta{}
	_ = GenerateOptions{}
	_ = StreamOptions{}
	_ = EmbedOptions{}
	_ = ImageGenerateOptions{}
	_ = Tool{}
	_ = ToolChoice{}
	_ = ResponseFormat{}
	_ = StructuredOutput{}
	_ = GenerateResult{}
	_ = StreamResult{}
	_ = EmbedResult{}
	_ = ImageGenerateResult{}
	_ = RequestMetadata{}
	_ = ResponseMetadata{}
	_ = StreamResponse{}
	_ = ImageResponseMetadata{}
	_ = ProviderMetadata{}
	_ = MessageMetadata{}
	_ = Warning{}
	_ = FinishReason{}
	_ = Usage{}
	_ = EmbeddingUsage{}
	_ = ChatOptions{}
	_ = CompletionOptions{}
	_ = EmbeddingOptions{}
	_ = ErrorParseInput{}
	_ = ProviderErrorStructure{}
	_ = APIError{}
	_ = APICallError{}
	_ = UnsupportedFunctionalityError{}
	_ = InvalidPromptError{}
	_ = InvalidResponseDataError{}
	_ = TooManyEmbeddingValuesForCallError{}
	t.Log("public type surface compiles")
}
