// Package imgdata embeds a real photograph used by the image/box examples so
// they don't depend on files outside the module. The TypeSafe API requires
// decodable images — tiny synthetic fixtures are rejected with "Invalid
// image dimensions" / "Invalid request", so a real JPEG is bundled instead.
package imgdata

import (
	_ "embed"
)

// PhotoJPEG is a real 800×600 photograph (a mountain landscape) that the
// TypeSafe API accepts as image state.
//
//go:embed photo.jpg
var PhotoJPEG []byte
