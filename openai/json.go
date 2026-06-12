package openai

import "encoding/json"

func jsonMarshal(v any) ([]byte, error) {
	return json.Marshal(v)
}

func jsonUnmarshal(data []byte, v any) error {
	return json.Unmarshal(data, v)
}

func jsonUnmarshalStrict(data []byte, v any) error {
	return json.Unmarshal(data, v)
}

func cloneProviderOptions(opts ProviderOptions) ProviderOptions {
	if opts == nil {
		return nil
	}
	out := make(ProviderOptions, len(opts))
	for key, values := range opts {
		cloned := make(map[string]any, len(values))
		for k, v := range values {
			cloned[k] = v
		}
		out[key] = cloned
	}
	return out
}
