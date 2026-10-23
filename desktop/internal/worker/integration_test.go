package worker

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestRealWorker drives the real airllm_worker.py with torch + airllm through the Go client,
// the same path the app uses. Opt in with AIRLLM_TEST_PYTHON=<python with torch/transformers>.
func TestRealWorker(t *testing.T) {
	py := os.Getenv("AIRLLM_TEST_PYTHON")
	if py == "" {
		t.Skip("set AIRLLM_TEST_PYTHON to run")
	}
	repo, _ := filepath.Abs("../../..")
	script := filepath.Join(repo, "desktop", "worker", "airllm_worker.py")

	var mu sync.Mutex
	byID := map[string][]map[string]any{}
	done := map[string]chan struct{}{}
	w := New(Hooks{OnMessage: func(raw json.RawMessage) {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Errorf("bad message %s", raw)
			return
		}
		id, _ := m["id"].(string)
		mu.Lock()
		byID[id] = append(byID[id], m)
		if m["type"] != "event" && done[id] != nil {
			close(done[id])
		}
		mu.Unlock()
	}})
	err := w.Start(Config{Python: py, Script: script, Dir: t.TempDir(), Env: []string{
		"PYTHONPATH=" + filepath.Join(repo, "air_llm"), "AIRLLM_WORKER_USE_BUNDLED=0", "PYTHONUTF8=1",
	}}, 2*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Stop(10 * time.Second)

	job := func(cmd string, args map[string]any) map[string]any {
		t.Helper()
		mu.Lock()
		id, err := w.Job(cmd, args)
		if err != nil {
			mu.Unlock()
			t.Fatal(err)
		}
		ch := make(chan struct{})
		done[id] = ch
		mu.Unlock()
		select {
		case <-ch:
		case <-time.After(10 * time.Minute):
			t.Fatalf("%s timed out", cmd)
		}
		mu.Lock()
		defer mu.Unlock()
		msgs := byID[id]
		last := msgs[len(msgs)-1]
		if last["type"] != "result" {
			t.Fatalf("%s failed: %v", cmd, last)
		}
		last["_events"] = msgs[:len(msgs)-1]
		return last
	}

	if _, err := w.Call("env", nil, time.Minute); err != nil {
		t.Fatal(err)
	}
	load := job("load", map[string]any{"model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
		"device": "cpu", "dtype": "float32", "max_seq_len": 64, "prefetching": true,
		"layer_shards_saving_path": t.TempDir()})
	if load["data"].(map[string]any)["architecture"] != "LlamaForCausalLM" {
		t.Fatalf("load: %v", load)
	}
	gen := job("generate", map[string]any{"prompt": "Hello", "max_new_tokens": 8})
	var streamed strings.Builder
	for _, e := range gen["_events"].([]map[string]any) {
		if e["event"] == "token" {
			streamed.WriteString(e["text"].(string))
		}
	}
	text := gen["data"].(map[string]any)["text"].(string)
	if strings.TrimSpace(streamed.String()) != strings.TrimSpace(text) || gen["data"].(map[string]any)["new_tokens"].(float64) == 0 {
		t.Fatalf("stream %q vs %q", streamed.String(), text)
	}
	st, err := w.Call("status", nil, 10*time.Second)
	if err != nil || !strings.Contains(string(st), "tiny-random") {
		t.Fatalf("status %s %v", st, err)
	}
}
