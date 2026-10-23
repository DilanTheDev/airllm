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

func TestParseProgress(t *testing.T) {
	p, ok := ParseProgress("Fetching 3 files:  33%|███▎      | 1/3 [00:01<00:02,  1.2s/it]")
	if !ok || p.Pct != 33 || p.Label != "Fetching 3 files" || !strings.HasPrefix(p.Text, "1/3") {
		t.Fatalf("got %+v %v", p, ok)
	}
	p, ok = ParseProgress("100%|██████████| 34/34 [00:10<00:00]")
	if !ok || p.Pct != 100 || p.Label != "" {
		t.Fatalf("got %+v %v", p, ok)
	}
	if _, ok := ParseProgress("loading layer model.layers.3"); ok {
		t.Fatal("plain line parsed as progress")
	}
	if _, ok := ParseProgress("bogus 400%| x"); ok {
		t.Fatal("pct > 100 accepted")
	}
}

func TestSplitCRLF(t *testing.T) {
	in := "a\rb\nc\r\nd"
	var got []string
	data := []byte(in)
	for len(data) > 0 {
		adv, tok, _ := splitCRLF(data, true)
		got = append(got, string(tok))
		data = data[adv:]
	}
	want := []string{"a", "b", "c", "", "d"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %q", got)
	}
}

// fakeWorker is a tiny python script that speaks the protocol, so the Go side is tested
// without torch. It needs python3 on PATH.
const fakeWorker = `
import json, sys, time
print(json.dumps({"id": None, "type": "event", "event": "ready"}), flush=True)
print("some log\rprogress 50%|##   | 1/2", file=sys.stderr, flush=True)
for raw in sys.stdin:
    m = json.loads(raw)
    if m["cmd"] == "echo":
        print(json.dumps({"id": m["id"], "type": "result", "data": m["args"]}), flush=True)
    elif m["cmd"] == "fail":
        print(json.dumps({"id": m["id"], "type": "error", "error": "boom"}), flush=True)
    elif m["cmd"] == "job":
        for i in range(3):
            print(json.dumps({"id": m["id"], "type": "event", "event": "token", "text": str(i)}), flush=True)
        print(json.dumps({"id": m["id"], "type": "result", "data": {"done": True}}), flush=True)
    elif m["cmd"] == "hang":
        time.sleep(600)
    elif m["cmd"] == "shutdown":
        break
`

func python(t *testing.T) string {
	for _, p := range []string{os.Getenv("AIRLLM_TEST_PYTHON"), "/usr/bin/python3", "python3", "python"} {
		if p == "" {
			continue
		}
		if filepath.IsAbs(p) {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	t.Skip("no python available")
	return ""
}

func startFake(t *testing.T, h Hooks) *Worker {
	dir := t.TempDir()
	script := filepath.Join(dir, "w.py")
	if err := os.WriteFile(script, []byte(fakeWorker), 0o644); err != nil {
		t.Fatal(err)
	}
	w := New(h)
	if err := w.Start(Config{Python: python(t), Script: script, Dir: dir}, 20*time.Second); err != nil {
		t.Fatal(err)
	}
	return w
}

func TestCallJobAndKill(t *testing.T) {
	var mu sync.Mutex
	var msgs []map[string]any
	var logs []string
	var progs []Progress
	exited := make(chan struct{})
	w := startFake(t, Hooks{
		OnMessage: func(raw json.RawMessage) {
			var m map[string]any
			_ = json.Unmarshal(raw, &m)
			mu.Lock()
			msgs = append(msgs, m)
			mu.Unlock()
		},
		OnLog:      func(l []string) { mu.Lock(); logs = append(logs, l...); mu.Unlock() },
		OnProgress: func(p Progress) { mu.Lock(); progs = append(progs, p); mu.Unlock() },
		OnExit:     func(error) { close(exited) },
	})

	res, err := w.Call("echo", map[string]any{"x": 1}, 5*time.Second)
	if err != nil || string(res) != `{"x": 1}` && string(res) != `{"x":1}` {
		t.Fatalf("echo: %s %v", res, err)
	}
	if _, err := w.Call("fail", nil, 5*time.Second); err == nil || err.Error() != "boom" {
		t.Fatalf("fail: %v", err)
	}

	id, err := w.Job("job", nil)
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		n := len(msgs)
		mu.Unlock()
		if n >= 4 || time.Now().After(deadline) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	mu.Lock()
	if len(msgs) != 4 || msgs[0]["id"] != id || msgs[3]["type"] != "result" {
		t.Fatalf("job msgs: %v", msgs)
	}
	mu.Unlock()

	// A hung call is unblocked by Kill: pending calls fail instead of waiting forever.
	errc := make(chan error, 1)
	go func() { _, err := w.Call("hang", nil, time.Minute); errc <- err }()
	time.Sleep(200 * time.Millisecond)
	w.Kill()
	select {
	case err := <-errc:
		if err == nil {
			t.Fatal("hung call returned nil after kill")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("hung call not released by kill")
	}
	<-exited
	if w.Running() {
		t.Fatal("still running after kill")
	}
	if _, err := w.Call("echo", nil, time.Second); err != ErrNotRunning {
		t.Fatalf("call after exit: %v", err)
	}
	time.Sleep(200 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	if len(logs) == 0 || logs[0] != "some log" {
		t.Fatalf("logs: %q", logs)
	}
	if len(progs) == 0 || progs[0].Pct != 50 {
		t.Fatalf("progress: %+v", progs)
	}
}

func TestStopIsGraceful(t *testing.T) {
	exited := make(chan struct{})
	w := startFake(t, Hooks{OnExit: func(error) { close(exited) }})
	start := time.Now()
	w.Stop(5 * time.Second)
	<-exited
	if time.Since(start) > 4*time.Second {
		t.Fatal("stop fell through to kill")
	}
	w.Stop(time.Second) // stopping twice is a no-op
}

func TestStartupFailure(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "bad.py")
	os.WriteFile(script, []byte("raise SystemExit(3)"), 0o644)
	w := New(Hooks{})
	if err := w.Start(Config{Python: python(t), Script: script, Dir: dir}, 10*time.Second); err == nil {
		t.Fatal("expected startup error")
	}
}
