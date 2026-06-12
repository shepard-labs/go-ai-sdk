package openai

// Voice identifiers for the Speech (TTS) API. Each constant maps to the
// wire-format string the OpenAI speech endpoint expects.
const (
	VoiceAlloy   = "alloy"
	VoiceAsh     = "ash"
	VoiceBallad  = "ballad"
	VoiceCoral   = "coral"
	VoiceEcho    = "echo"
	VoiceFable   = "fable"
	VoiceNova    = "nova"
	VoiceOnyx    = "onyx"
	VoiceSage    = "sage"
	VoiceShimmer = "shimmer"
	VoiceVerse   = "verse"
)

// DefaultVoice is used when the caller does not specify a voice.
const DefaultVoice = VoiceAlloy

// validVoices is the closed set of supported voice identifiers.
var validVoices = map[string]struct{}{
	VoiceAlloy:   {},
	VoiceAsh:     {},
	VoiceBallad:  {},
	VoiceCoral:   {},
	VoiceEcho:    {},
	VoiceFable:   {},
	VoiceNova:    {},
	VoiceOnyx:    {},
	VoiceSage:    {},
	VoiceShimmer: {},
	VoiceVerse:   {},
}

// validOutputFormats is the closed set of supported audio output formats.
var validOutputFormats = map[string]struct{}{
	"mp3":  {},
	"opus": {},
	"aac":  {},
	"flac": {},
	"wav":  {},
	"pcm":  {},
}

// defaultOutputFormat is the fallback audio output format.
const defaultOutputFormat = "mp3"

// isValidVoice returns true if v is a known voice identifier.
func isValidVoice(v string) bool {
	_, ok := validVoices[v]
	return ok
}

// isValidOutputFormat returns true if f is a known audio output format.
func isValidOutputFormat(f string) bool {
	_, ok := validOutputFormats[f]
	return ok
}

// mediaTypeToExtension maps a media type to its file extension for use in
// the filename of multipart uploads (e.g. "audio/mpeg" → "mp3",
// "audio/mp4" → "m4a"). Per spec table.
func mediaTypeToExtension(mediaType string) string {
	switch mediaType {
	case "audio/wav", "audio/wave", "audio/x-wav":
		return "wav"
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/mp4", "audio/m4a", "audio/x-m4a":
		return "m4a"
	case "audio/webm":
		return "webm"
	case "audio/mpga":
		return "mpga"
	case "audio/ogg":
		return "ogg"
	case "audio/flac", "audio/x-flac":
		return "flac"
	case "audio/aac":
		return "aac"
	case "audio/opus":
		return "opus"
	case "audio/pcm":
		return "pcm"
	}
	return "mp3"
}
