package openai

import "testing"

func TestIsValidVoiceAcceptsAllKnownVoices(t *testing.T) {
	for _, v := range []string{
		VoiceAlloy, VoiceAsh, VoiceBallad, VoiceCoral, VoiceEcho,
		VoiceFable, VoiceNova, VoiceOnyx, VoiceSage, VoiceShimmer, VoiceVerse,
	} {
		if !isValidVoice(v) {
			t.Errorf("isValidVoice(%q) = false, want true", v)
		}
	}
}

func TestIsValidVoiceRejectsUnknown(t *testing.T) {
	for _, v := range []string{"", "junk", "ALLoy", "ALLOY"} {
		if isValidVoice(v) {
			t.Errorf("isValidVoice(%q) = true, want false", v)
		}
	}
}

func TestIsValidOutputFormat(t *testing.T) {
	for _, f := range []string{"mp3", "opus", "aac", "flac", "wav", "pcm"} {
		if !isValidOutputFormat(f) {
			t.Errorf("isValidOutputFormat(%q) = false, want true", f)
		}
	}
	for _, f := range []string{"", "ogg", "wma", "Mp3"} {
		if isValidOutputFormat(f) {
			t.Errorf("isValidOutputFormat(%q) = true, want false", f)
		}
	}
}

func TestDefaultVoiceIsAlloy(t *testing.T) {
	if DefaultVoice != "alloy" {
		t.Errorf("DefaultVoice = %q, want alloy", DefaultVoice)
	}
}

func TestDefaultOutputFormatIsMp3(t *testing.T) {
	if defaultOutputFormat != "mp3" {
		t.Errorf("defaultOutputFormat = %q, want mp3", defaultOutputFormat)
	}
}
