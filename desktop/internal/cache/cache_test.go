package cache

import (
	"os"
	"path/filepath"
	"testing"
)

func write(t *testing.T, p string, n int) {
	t.Helper()
	os.MkdirAll(filepath.Dir(p), 0o755)
	if err := os.WriteFile(p, make([]byte, n), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestScanAndDelete(t *testing.T) {
	hub := t.TempDir()
	repo := filepath.Join(hub, "models--Qwen--Qwen3-8B")
	write(t, filepath.Join(repo, "blobs", "abc"), 1000)
	snap := filepath.Join(repo, "snapshots", "deadbeef")
	write(t, filepath.Join(snap, "config.json"), 10)
	write(t, filepath.Join(snap, "splitted_model", "model.layers.0.safetensors"), 100)
	write(t, filepath.Join(snap, "splitted_model", "model.layers.1.safetensors"), 100)
	write(t, filepath.Join(snap, "splitted_model.4bit", "model.layers.0.safetensors"), 30)
	write(t, filepath.Join(hub, "datasets--x--y", "f"), 5) // not a model

	custom := t.TempDir()
	write(t, filepath.Join(custom, "splitted_model", "lm_head.safetensors"), 50)
	empty := t.TempDir()

	entries := Scan(hub, []string{custom, empty, ""})
	if len(entries) != 2 {
		t.Fatalf("entries: %+v", entries)
	}
	e := entries[0]
	if e.Name != "Qwen/Qwen3-8B" || !e.HubRepo || e.Bytes != 1240 || len(e.Splits) != 2 || e.Snapshots != 1 {
		t.Fatalf("repo: %+v", e)
	}
	comp := map[string]Split{}
	for _, s := range e.Splits {
		comp[s.Compression] = s
	}
	if comp[""].Layers != 2 || comp["4bit"].Bytes != 30 {
		t.Fatalf("splits: %+v", e.Splits)
	}
	if entries[1].HubRepo || len(entries[1].Splits) != 1 {
		t.Fatalf("custom: %+v", entries[1])
	}

	// Refuses anything that is not a scanned model folder.
	for _, bad := range []string{hub, snap, filepath.Join(snap, "config.json"), custom, "/"} {
		if err := Delete(bad, hub, []string{custom}); err == nil {
			t.Fatalf("deleted %s", bad)
		}
	}
	if err := Delete(filepath.Join(snap, "splitted_model.4bit"), hub, []string{custom}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(snap, "splitted_model.4bit")); err == nil {
		t.Fatal("split not deleted")
	}
	if err := Delete(repo, hub, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(repo); err == nil {
		t.Fatal("repo not deleted")
	}
	if _, err := os.Stat(filepath.Join(hub, "datasets--x--y")); err != nil {
		t.Fatal("unrelated folder touched")
	}
}

func TestHubDir(t *testing.T) {
	if got := HubDir("/m"); got != filepath.Join("/m", "hub") {
		t.Fatal(got)
	}
	t.Setenv("HF_HUB_CACHE", "/x/hub")
	if got := HubDir(""); got != "/x/hub" {
		t.Fatal(got)
	}
}
