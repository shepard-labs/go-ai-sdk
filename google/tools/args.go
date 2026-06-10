package tools

// Args types for the Google provider tools.
//
// These are defined in this subpackage to avoid an import cycle (the
// parent google package imports this subpackage for the PrepareTools
// dispatcher). The google package re-exports them as type aliases so
// callers see a single vocabulary.

import "sort"

// GoogleSearchArgs carries optional arguments for the googleSearch provider tool.
type GoogleSearchArgs struct {
	SearchTypes     *GoogleSearchTypes
	TimeRangeFilter *TimeRangeFilter
}

// GoogleSearchTypes specifies which search types to enable.
type GoogleSearchTypes struct {
	WebSearch   map[string]any
	ImageSearch map[string]any
}

// TimeRangeFilter restricts search results to a time range (RFC 3339 strings).
type TimeRangeFilter struct {
	StartTime string
	EndTime   string
}

// FileSearchArgs carries arguments for the fileSearch provider tool.
type FileSearchArgs struct {
	FileSearchStoreNames []string
	TopK                 *int
	MetadataFilter       string
}

// VertexRagStoreArgs carries arguments for the vertexRagStore provider tool.
type VertexRagStoreArgs struct {
	RagCorpus string // "projects/{p}/locations/{l}/ragCorpora/{c}"
	TopK      *int
}

// sortStrings is a small helper used by PrepareTools.
func sortStrings(s []string) {
	sort.Strings(s)
}
