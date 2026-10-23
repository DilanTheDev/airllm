// genicon draws the app icon (the same ringed square as the titlebar mark) and writes
// build/appicon.png plus a multi-size build/windows/icon.ico.
package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
)

var (
	bg     = color.NRGBA{0x14, 0x15, 0x18, 0xff}
	accent = color.NRGBA{0x6d, 0x9e, 0xff, 0xff}
)

// roundedRectCoverage returns how much of pixel (x,y) lies inside a rounded rect (anti-aliased).
func inside(px, py, x0, y0, x1, y1, r float64) float64 {
	cx := math.Max(x0+r, math.Min(px, x1-r))
	cy := math.Max(y0+r, math.Min(py, y1-r))
	d := math.Hypot(px-cx, py-cy) - r
	return math.Max(0, math.Min(1, 0.5-d))
}

func blend(a, b color.NRGBA, t float64) color.NRGBA {
	l := func(x, y uint8) uint8 { return uint8(float64(x)*(1-t) + float64(y)*t + 0.5) }
	return color.NRGBA{l(a.R, b.R), l(a.G, b.G), l(a.B, b.B), l(a.A, b.A)}
}

func draw(n int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	s := float64(n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			px, py := float64(x)+0.5, float64(y)+0.5
			c := color.NRGBA{}
			// tile
			if t := inside(px, py, 0, 0, s, s, s*0.22); t > 0 {
				c = blend(c, bg, t)
			}
			// ring
			m := s * 0.22
			if t := inside(px, py, m, m, s-m, s-m, s*0.14); t > 0 {
				c = blend(c, accent, t)
			}
			g := s * 0.30
			if t := inside(px, py, g, g, s-g, s-g, s*0.07); t > 0 {
				c = blend(c, bg, t)
			}
			k := s * 0.36
			if t := inside(px, py, k, k, s-k, s-k, s*0.04); t > 0 {
				c = blend(c, accent, t)
			}
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

func main() {
	out := "build"
	if len(os.Args) > 1 {
		out = os.Args[1]
	}
	must := func(err error) {
		if err != nil {
			log.Fatal(err)
		}
	}
	must(os.MkdirAll(filepath.Join(out, "windows"), 0o755))
	var big bytes.Buffer
	must(png.Encode(&big, draw(1024)))
	must(os.WriteFile(filepath.Join(out, "appicon.png"), big.Bytes(), 0o644))

	sizes := []int{16, 24, 32, 48, 64, 128, 256}
	var pngs [][]byte
	for _, sz := range sizes {
		var b bytes.Buffer
		must(png.Encode(&b, draw(sz)))
		pngs = append(pngs, b.Bytes())
	}
	var ico bytes.Buffer
	binary.Write(&ico, binary.LittleEndian, []uint16{0, 1, uint16(len(sizes))})
	offset := 6 + 16*len(sizes)
	for i, sz := range sizes {
		w := uint8(sz)
		if sz >= 256 {
			w = 0
		}
		binary.Write(&ico, binary.LittleEndian, struct {
			W, H, Colors, Reserved uint8
			Planes, Bits           uint16
			Size, Offset           uint32
		}{w, w, 0, 0, 1, 32, uint32(len(pngs[i])), uint32(offset)})
		offset += len(pngs[i])
	}
	for _, p := range pngs {
		ico.Write(p)
	}
	must(os.WriteFile(filepath.Join(out, "windows", "icon.ico"), ico.Bytes(), 0o644))
}
