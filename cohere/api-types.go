package cohere

type cohereDocument struct {
	Data map[string]any `json:"data"`
}
type coherePrompt struct {
	Messages  []map[string]any
	Documents []cohereDocument
	Warnings  []Warning
}

type cohereChatResponse struct {
	GenerationID *string               `json:"generation_id"`
	Message      cohereResponseMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
	Usage        struct {
		BilledUnits *CohereUsageTokens `json:"billed_units"`
		Tokens      *CohereUsageTokens `json:"tokens"`
	} `json:"usage"`
}
type cohereResponseMessage struct {
	Role      string                   `json:"role"`
	Content   []cohereResponseContent  `json:"content"`
	ToolCalls []cohereResponseToolCall `json:"tool_calls"`
	Citations []cohereCitation         `json:"citations"`
}
type cohereResponseContent struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}
type cohereResponseToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}
type cohereCitation struct {
	Start   int                    `json:"start"`
	End     int                    `json:"end"`
	Text    string                 `json:"text"`
	Sources []cohereCitationSource `json:"sources"`
	Type    *string                `json:"type"`
}
type cohereCitationSource struct {
	Type     *string `json:"type"`
	ID       *string `json:"id"`
	Document struct {
		ID    *string `json:"id"`
		Text  string  `json:"text"`
		Title string  `json:"title"`
	} `json:"document"`
}

type cohereStreamType struct {
	Type string `json:"type"`
}
type cohereStreamContentChunk struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Message struct {
			Content struct {
				Type     string  `json:"type"`
				Text     *string `json:"text"`
				Thinking *string `json:"thinking"`
			} `json:"content"`
		} `json:"message"`
	} `json:"delta"`
}
type cohereStreamMessageStart struct {
	Type string `json:"type"`
	ID   string `json:"id"`
}
type cohereStreamMessageEnd struct {
	Type  string `json:"type"`
	Delta struct {
		FinishReason string `json:"finish_reason"`
		Usage        struct {
			Tokens *CohereUsageTokens `json:"tokens"`
		} `json:"usage"`
	} `json:"delta"`
}
type cohereStreamToolCallChunk struct {
	Type  string `json:"type"`
	Delta struct {
		Message struct {
			ToolCalls struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"delta"`
}

type cohereEmbeddingResponse struct {
	Embeddings struct {
		Float [][]float64 `json:"float"`
	} `json:"embeddings"`
	Meta struct {
		BilledUnits struct {
			InputTokens int `json:"input_tokens"`
		} `json:"billed_units"`
	} `json:"meta"`
}
type cohereRerankResponse struct {
	ID      string `json:"id"`
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
	Meta any `json:"meta"`
}
