package tools

import anthropic "github.com/shepard-labs/go-ai-sdk/anthropic"

type Step struct {
	ProviderMetadata anthropic.ProviderMetadata
	MessageMetadata  anthropic.MessageMetadata
}

type ProviderOptions struct {
	Container *anthropic.Container
}

func ForwardAnthropicContainerIdFromLastStep(steps []Step) (ProviderOptions, error) {
	for i := len(steps) - 1; i >= 0; i-- {
		if id, ok := containerIDFromMetadata(steps[i].MessageMetadata); ok {
			return ProviderOptions{Container: &anthropic.Container{ID: id}}, nil
		}
		if id, ok := containerIDFromMetadata(steps[i].ProviderMetadata); ok {
			return ProviderOptions{Container: &anthropic.Container{ID: id}}, nil
		}
	}
	return ProviderOptions{}, nil
}

func containerIDFromMetadata(metadata map[string]any) (string, bool) {
	if metadata == nil {
		return "", false
	}
	if id, ok := metadata["containerID"].(string); ok && id != "" {
		return id, true
	}
	if container, ok := metadata["container"].(*anthropic.ContainerInfo); ok && container.ID != "" {
		return container.ID, true
	}
	if container, ok := metadata["container"].(anthropic.ContainerInfo); ok && container.ID != "" {
		return container.ID, true
	}
	return "", false
}
