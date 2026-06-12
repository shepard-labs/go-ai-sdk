package openai

// languageNameToISO maps the 57 full-language names the OpenAI transcription
// API returns (e.g. "english", "chinese") to ISO-639-1 codes ("en", "zh").
// Unrecognized strings map to "" (empty).
var languageNameToISO = map[string]string{
	"afrikaans":         "af",
	"albanian":          "sq",
	"amharic":           "am",
	"arabic":            "ar",
	"armenian":          "hy",
	"assamese":          "as",
	"azerbaijani":       "az",
	"bashkir":           "ba",
	"basque":            "eu",
	"belarusian":        "be",
	"bengali":           "bn",
	"bosnian":           "bs",
	"breton":            "br",
	"bulgarian":         "bg",
	"catalan":           "ca",
	"chinese":           "zh",
	"croatian":          "hr",
	"czech":             "cs",
	"danish":            "da",
	"dutch":             "nl",
	"english":           "en",
	"estonian":          "et",
	"faroese":           "fo",
	"finnish":           "fi",
	"french":            "fr",
	"galician":          "gl",
	"georgian":          "ka",
	"german":            "de",
	"greek":             "el",
	"gujarati":          "gu",
	"haitian":           "ht",
	"haitian creole":    "ht",
	"hebrew":            "he",
	"hindi":             "hi",
	"hungarian":         "hu",
	"icelandic":         "is",
	"indonesian":        "id",
	"italian":           "it",
	"japanese":          "ja",
	"kannada":           "kn",
	"kazakh":            "kk",
	"korean":            "ko",
	"latvian":           "lv",
	"lithuanian":        "lt",
	"luxembourgish":     "lb",
	"macedonian":        "mk",
	"malay":             "ms",
	"malayalam":         "ml",
	"maltese":           "mt",
	"marathi":           "mr",
	"mongolian":         "mn",
	"norwegian":         "no",
	"nynorsk":           "nn",
	"pashto":            "ps",
	"persian":           "fa",
	"polish":            "pl",
	"portuguese":        "pt",
	"punjabi":           "pa",
	"romanian":          "ro",
	"russian":           "ru",
	"serbian":           "sr",
	"sinhala":           "si",
	"slovak":            "sk",
	"slovenian":         "sl",
	"spanish":           "es",
	"swahili":           "sw",
	"swedish":           "sv",
	"tagalog":           "tl",
	"tamil":             "ta",
	"tatar":             "tt",
	"telugu":            "te",
	"thai":              "th",
	"turkish":           "tr",
	"ukrainian":         "uk",
	"urdu":              "ur",
	"uzbek":             "uz",
	"vietnamese":        "vi",
	"welsh":             "cy",
	"yiddish":           "yi",
	"yoruba":            "yo",
}

// mapLanguageToISO normalizes a full-name language string from the OpenAI
// transcription response to its ISO-639-1 code. Unrecognized values map
// to "" so callers can detect "unknown" easily.
func mapLanguageToISO(name string) string {
	if name == "" {
		return ""
	}
	if v, ok := languageNameToISO[name]; ok {
		return v
	}
	return ""
}
