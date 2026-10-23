package settings

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRoundTripAndDefaults(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sub", "settings.json")
	s := Open(p)
	if s.Get().Model.MaxSeqLen != 512 || !s.Get().Model.Prefetching {
		t.Fatal("defaults not applied")
	}
	v := s.Get()
	v.Model.Model = "meta-llama/Llama-3.1-8B"
	v.Generation.DoSample = true
	v.Model.Prefetching = false
	if err := s.Save(v); err != nil {
		t.Fatal(err)
	}
	r := Open(p).Get()
	if r.Model.Model != "meta-llama/Llama-3.1-8B" || !r.Generation.DoSample || r.Model.Prefetching {
		t.Fatalf("%+v", r)
	}

	// A partial file keeps defaults for the fields it lacks.
	os.WriteFile(p, []byte(`{"model":{"model":"x"}}`), 0o600)
	r = Open(p).Get()
	if r.Model.Model != "x" || r.Generation.MaxNewTokens != 128 || r.Train.LoraR != 16 {
		t.Fatalf("%+v", r)
	}
	// A corrupt file falls back to defaults.
	os.WriteFile(p, []byte(`{not json`), 0o600)
	if Open(p).Get().Model.Model != Defaults().Model.Model {
		t.Fatal("corrupt file not ignored")
	}
}

func TestDataDir(t *testing.T) {
	d := t.TempDir()
	got, err := DataDir(d)
	if err != nil || got != d {
		t.Fatalf("%s %v", got, err)
	}
}
