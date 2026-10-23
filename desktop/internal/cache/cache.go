// Package cache lists and removes the models AirLLM has downloaded and split into layer shards.
package cache

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Split struct {
	Path        string `json:"path"`
	Compression string `json:"compression"` // "", "4bit", "8bit"
	Bytes       int64  `json:"bytes"`
	Layers      int    `json:"layers"`
}

type Entry struct {
	Name      string  `json:"name"` // repo id, or folder name for custom locations
	Path      string  `json:"path"`
	Bytes     int64   `json:"bytes"` // whole folder on disk (originals + splits)
	Splits    []Split `json:"splits"`
	HubRepo   bool    `json:"hubRepo"`
	Snapshots int     `json:"snapshots"`
}

// HubDir resolves the Hugging Face hub cache the same way huggingface_hub does.
func HubDir(hfHomeOverride string) string {
	if hfHomeOverride != "" {
		return filepath.Join(hfHomeOverride, "hub")
	}
	if v := os.Getenv("HF_HUB_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HUGGINGFACE_HUB_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// dirSize sums regular files without following symlinks, so hub snapshot links are not
// double counted against their blobs.
func dirSize(root string) int64 {
	var n int64
	_ = filepath.WalkDir(root, func(_ string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.Type().IsRegular() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}

func splitCompression(name string) string {
	switch {
	case strings.HasSuffix(name, ".4bit"):
		return "4bit"
	case strings.HasSuffix(name, ".8bit"):
		return "8bit"
	}
	return ""
}

func isSplitDir(name string) bool { return strings.HasPrefix(name, "splitted_model") }

func readSplit(path string) Split {
	s := Split{Path: path, Compression: splitCompression(filepath.Base(path)), Bytes: dirSize(path)}
	if ents, err := os.ReadDir(path); err == nil {
		for _, e := range ents {
			if strings.HasSuffix(e.Name(), ".safetensors") {
				s.Layers++
			}
		}
	}
	return s
}

// findSplits looks for splitted_model* folders up to depth levels below root.
func findSplits(root string, depth int) []Split {
	var out []Split
	ents, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		p := filepath.Join(root, e.Name())
		if isSplitDir(e.Name()) {
			out = append(out, readSplit(p))
		} else if depth > 0 {
			out = append(out, findSplits(p, depth-1)...)
		}
	}
	return out
}

// Scan lists hub repos (models--org--name) plus any extra folders holding split models.
func Scan(hub string, extra []string) []Entry {
	var out []Entry
	if ents, err := os.ReadDir(hub); err == nil {
		for _, e := range ents {
			if !e.IsDir() || !strings.HasPrefix(e.Name(), "models--") {
				continue
			}
			p := filepath.Join(hub, e.Name())
			name := strings.ReplaceAll(strings.TrimPrefix(e.Name(), "models--"), "--", "/")
			entry := Entry{Name: name, Path: p, HubRepo: true, Bytes: dirSize(p)}
			if snaps, err := os.ReadDir(filepath.Join(p, "snapshots")); err == nil {
				entry.Snapshots = len(snaps)
				for _, s := range snaps {
					entry.Splits = append(entry.Splits, findSplits(filepath.Join(p, "snapshots", s.Name()), 0)...)
				}
			}
			out = append(out, entry)
		}
	}
	for _, root := range extra {
		if root == "" {
			continue
		}
		splits := findSplits(root, 2)
		if isSplitDir(filepath.Base(root)) {
			splits = append(splits, readSplit(root))
		}
		if len(splits) == 0 {
			continue
		}
		out = append(out, Entry{Name: filepath.Base(root), Path: root, Bytes: dirSize(root), Splits: splits})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Bytes > out[j].Bytes })
	return out
}

// Delete removes path, but only if it is a hub repo folder or a split folder found by Scan.
// This keeps a bad call from the UI from ever deleting an arbitrary directory.
func Delete(path, hub string, extra []string) error {
	clean, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	for _, e := range Scan(hub, extra) {
		if e.HubRepo && samePath(e.Path, clean) {
			return os.RemoveAll(clean)
		}
		for _, s := range e.Splits {
			if samePath(s.Path, clean) {
				return os.RemoveAll(clean)
			}
		}
	}
	return errors.New("refusing to delete: not a known model cache folder")
}

func samePath(a, b string) bool {
	aa, err1 := filepath.Abs(a)
	bb, err2 := filepath.Abs(b)
	if err1 != nil || err2 != nil {
		return false
	}
	if os.PathSeparator == '\\' {
		return strings.EqualFold(aa, bb)
	}
	return aa == bb
}
