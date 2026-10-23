// Package worker runs airllm_worker.py as a child process and speaks its JSON-lines protocol.
//
// The model lives in that process, not in the app, so the window never blocks on torch, a CUDA
// crash cannot take the UI down, and "force stop" (killing the process) always frees VRAM.
package worker

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Message is one protocol line from the worker.
type Message struct {
	ID    *string         `json:"id"`
	Type  string          `json:"type"`
	Event string          `json:"event,omitempty"`
	Data  json.RawMessage `json:"data,omitempty"`
	Error string          `json:"error,omitempty"`
	Raw   json.RawMessage `json:"-"`
}

// Progress is a tqdm bar parsed from the worker's stderr (downloads, layer splitting).
type Progress struct {
	Label string  `json:"label"`
	Pct   float64 `json:"pct"`
	Text  string  `json:"text"`
}

type Hooks struct {
	OnMessage  func(raw json.RawMessage) // every async event/result, forwarded to the UI as is
	OnLog      func(lines []string)      // batched stderr lines
	OnProgress func(p Progress)
	OnExit     func(err error)
}

type Config struct {
	Python string
	Script string
	Dir    string
	Env    []string
}

type pending struct {
	ch chan Message
}

type Worker struct {
	hooks Hooks

	mu      sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	pending map[string]*pending
	ready   chan struct{}
	exited  chan struct{}
	alive   bool

	seq atomic.Int64
}

func New(h Hooks) *Worker {
	return &Worker{hooks: h, pending: map[string]*pending{}}
}

var ErrNotRunning = errors.New("engine is not running")

func (w *Worker) Running() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.alive
}

// Start launches the worker and waits until it reports ready (python + imports up).
func (w *Worker) Start(cfg Config, timeout time.Duration) error {
	w.mu.Lock()
	if w.alive {
		w.mu.Unlock()
		return nil
	}
	cmd := exec.Command(cfg.Python, "-u", cfg.Script)
	cmd.Dir = cfg.Dir
	cmd.Env = append(os.Environ(), cfg.Env...)
	hideWindow(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		w.mu.Unlock()
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		w.mu.Unlock()
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		w.mu.Unlock()
		return err
	}
	if err := cmd.Start(); err != nil {
		w.mu.Unlock()
		return fmt.Errorf("start python: %w", err)
	}
	w.cmd, w.stdin, w.alive = cmd, stdin, true
	w.ready = make(chan struct{})
	w.exited = make(chan struct{})
	ready, exited := w.ready, w.exited
	w.mu.Unlock()

	var readers sync.WaitGroup
	readers.Add(2)
	go func() { defer readers.Done(); w.readStdout(stdout) }()
	go func() { defer readers.Done(); w.readStderr(stderr) }()
	go func() {
		readers.Wait()
		err := cmd.Wait()
		w.mu.Lock()
		w.alive = false
		for id, p := range w.pending {
			p.ch <- Message{Type: "error", Error: "engine stopped"}
			delete(w.pending, id)
		}
		w.mu.Unlock()
		close(exited)
		if w.hooks.OnExit != nil {
			w.hooks.OnExit(err)
		}
	}()

	select {
	case <-ready:
		return nil
	case <-exited:
		return errors.New("engine exited during startup; see the log for the Python error")
	case <-time.After(timeout):
		w.Kill()
		return errors.New("engine did not start in time")
	}
}

func (w *Worker) readStdout(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 64*1024*1024)
	for sc.Scan() {
		line := append([]byte(nil), sc.Bytes()...)
		var m Message
		if err := json.Unmarshal(line, &m); err != nil {
			w.emitLog([]string{"[engine] " + string(line)})
			continue
		}
		if m.ID == nil {
			if m.Event == "ready" {
				w.mu.Lock()
				if w.ready != nil {
					close(w.ready)
					w.ready = nil
				}
				w.mu.Unlock()
			}
			continue
		}
		w.mu.Lock()
		p := w.pending[*m.ID]
		if p != nil && m.Type != "event" {
			delete(w.pending, *m.ID)
		}
		w.mu.Unlock()
		if p != nil {
			if m.Type != "event" {
				p.ch <- m
			}
			continue
		}
		if w.hooks.OnMessage != nil {
			w.hooks.OnMessage(line)
		}
	}
}

var tqdmRe = regexp.MustCompile(`^(.*?):?\s*(\d{1,3})%\|`)

// ParseProgress recognises a tqdm status line such as "Fetching 3 files:  33%|███ | 1/3 [...]".
func ParseProgress(line string) (Progress, bool) {
	m := tqdmRe.FindStringSubmatch(line)
	if m == nil {
		return Progress{}, false
	}
	pct, err := strconv.Atoi(m[2])
	if err != nil || pct > 100 {
		return Progress{}, false
	}
	label := strings.TrimSpace(m[1])
	text := line
	if i := strings.LastIndex(line, "|"); i >= 0 && i+1 < len(line) {
		text = strings.TrimSpace(line[i+1:])
	}
	return Progress{Label: label, Pct: float64(pct), Text: text}, true
}

// splitCRLF splits on \n and on bare \r, which tqdm uses to redraw a bar in place.
func splitCRLF(data []byte, atEOF bool) (int, []byte, error) {
	for i, b := range data {
		if b == '\n' || b == '\r' {
			return i + 1, data[:i], nil
		}
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

func (w *Worker) readStderr(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	sc.Split(splitCRLF)

	// Batch log lines so a chatty download doesn't flood the UI with one IPC call per line.
	lines := make(chan string, 1024)
	done := make(chan struct{})
	go func() {
		defer close(done)
		var buf []string
		tick := time.NewTicker(120 * time.Millisecond)
		defer tick.Stop()
		var lastProg Progress
		var haveProg bool
		flush := func() {
			if len(buf) > 0 {
				w.emitLog(buf)
				buf = nil
			}
			if haveProg && w.hooks.OnProgress != nil {
				w.hooks.OnProgress(lastProg)
				haveProg = false
			}
		}
		for {
			select {
			case l, ok := <-lines:
				if !ok {
					flush()
					return
				}
				if p, isProg := ParseProgress(l); isProg {
					lastProg, haveProg = p, true
					if p.Pct < 100 {
						continue // redraws of the same bar don't belong in the log
					}
				}
				buf = append(buf, l)
				if len(buf) >= 500 {
					flush()
				}
			case <-tick.C:
				flush()
			}
		}
	}()
	for sc.Scan() {
		if s := strings.TrimRight(sc.Text(), " "); s != "" {
			lines <- s
		}
	}
	close(lines)
	<-done
}

func (w *Worker) emitLog(lines []string) {
	if w.hooks.OnLog != nil && len(lines) > 0 {
		w.hooks.OnLog(lines)
	}
}

func (w *Worker) write(id, cmd string, args any) error {
	b, err := json.Marshal(map[string]any{"id": id, "cmd": cmd, "args": args})
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.alive {
		return ErrNotRunning
	}
	_, err = w.stdin.Write(append(b, '\n'))
	return err
}

func (w *Worker) nextID() string { return "q" + strconv.FormatInt(w.seq.Add(1), 10) }

// Call sends a command and waits for its result. Events for it (if any) are dropped.
func (w *Worker) Call(cmd string, args any, timeout time.Duration) (json.RawMessage, error) {
	id := w.nextID()
	p := &pending{ch: make(chan Message, 1)}
	w.mu.Lock()
	if !w.alive {
		w.mu.Unlock()
		return nil, ErrNotRunning
	}
	w.pending[id] = p
	w.mu.Unlock()
	if err := w.write(id, cmd, args); err != nil {
		w.mu.Lock()
		delete(w.pending, id)
		w.mu.Unlock()
		return nil, err
	}
	select {
	case m := <-p.ch:
		if m.Type == "error" {
			return nil, errors.New(m.Error)
		}
		return m.Data, nil
	case <-time.After(timeout):
		w.mu.Lock()
		delete(w.pending, id)
		w.mu.Unlock()
		return nil, fmt.Errorf("%s timed out", cmd)
	}
}

// Start a long job; its events and final result arrive through Hooks.OnMessage.
func (w *Worker) Job(cmd string, args any) (string, error) {
	id := w.nextID()
	return id, w.write(id, cmd, args)
}

// Stop asks the worker to exit, killing it if it doesn't within the grace period.
func (w *Worker) Stop(grace time.Duration) {
	w.mu.Lock()
	alive, exited := w.alive, w.exited
	w.mu.Unlock()
	if !alive {
		return
	}
	_ = w.write("bye", "shutdown", nil)
	select {
	case <-exited:
	case <-time.After(grace):
		w.Kill()
	}
}

// Kill terminates the worker immediately (used for force stop during a load or a hung job).
func (w *Worker) Kill() {
	w.mu.Lock()
	cmd, alive, exited := w.cmd, w.alive, w.exited
	w.mu.Unlock()
	if !alive || cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
	}
}
