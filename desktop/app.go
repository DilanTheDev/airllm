package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"sync"
	"time"

	"airllm-desktop/internal/cache"
	"airllm-desktop/internal/engine"
	"airllm-desktop/internal/settings"
	"airllm-desktop/internal/worker"

	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

const AppVersion = "1.0.0"

type App struct {
	ctx     context.Context
	dataDir string
	store   *settings.Store
	paths   engine.Paths
	files   fs.FS
	w       *worker.Worker

	mu          sync.Mutex
	setupCancel context.CancelFunc
	startMu     sync.Mutex // held while the worker starts; concurrent callers wait for it
}

func NewApp(files fs.FS) *App {
	a := &App{files: files}
	a.w = worker.New(worker.Hooks{
		OnMessage:  func(raw json.RawMessage) { a.emit("engine:msg", raw) },
		OnLog:      func(lines []string) { a.emit("engine:log", lines) },
		OnProgress: func(p worker.Progress) { a.emit("engine:progress", p) },
		OnExit: func(err error) {
			msg := ""
			if err != nil {
				msg = err.Error()
			}
			a.emit("engine:exit", map[string]any{"error": msg})
		},
	})
	return a
}

func (a *App) emit(name string, data any) {
	if a.ctx != nil {
		runtime.EventsEmit(a.ctx, name, data)
	}
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	exe, err := os.Executable()
	exeDir := "."
	if err == nil {
		exeDir = filepath.Dir(exe)
	}
	a.dataDir, err = settings.DataDir(exeDir)
	if err != nil {
		a.dataDir = exeDir
	}
	a.store = settings.Open(filepath.Join(a.dataDir, "settings.json"))
	a.paths = engine.NewPaths(a.dataDir)
}

// secondInstance brings the running window forward when the exe is launched again.
func (a *App) secondInstance(options.SecondInstanceData) {
	if a.ctx == nil {
		return
	}
	runtime.WindowUnminimise(a.ctx)
	runtime.Show(a.ctx)
}

func (a *App) shutdown(ctx context.Context) {
	a.mu.Lock()
	if a.setupCancel != nil {
		a.setupCancel()
	}
	a.mu.Unlock()
	a.w.Stop(3 * time.Second)
}

// ---- state ------------------------------------------------------------------------------------

type EngineState struct {
	Installed  bool         `json:"installed"`
	External   bool         `json:"external"`
	Python     string       `json:"python"`
	Info       *engine.Info `json:"info"`
	Running    bool         `json:"running"`
	SettingUp  bool         `json:"settingUp"`
	Supported  bool         `json:"supported"`
	RuntimeDir string       `json:"runtimeDir"`
}

type Boot struct {
	Version      string            `json:"version"`
	Platform     string            `json:"platform"`
	DataDir      string            `json:"dataDir"`
	SettingsPath string            `json:"settingsPath"`
	Settings     settings.Settings `json:"settings"`
	Defaults     settings.Settings `json:"defaults"`
	Engine       EngineState       `json:"engine"`
	Variants     []engine.Variant  `json:"variants"`
	HubDir       string            `json:"hubDir"`
}

func (a *App) Bootstrap() Boot {
	s := a.store.Get()
	return Boot{
		Version: AppVersion, Platform: goruntime.GOOS, DataDir: a.dataDir,
		SettingsPath: a.store.Path(), Settings: s, Defaults: settings.Defaults(),
		Engine: a.EngineStatus(), Variants: engine.Variants, HubDir: cache.HubDir(s.Engine.ModelCache),
	}
}

func (a *App) pythonExe() (string, bool) {
	if p := a.store.Get().Engine.PythonPath; p != "" {
		return p, true
	}
	return a.paths.PythonExe(), false
}

func (a *App) EngineStatus() EngineState {
	py, external := a.pythonExe()
	st := EngineState{Python: py, External: external, Running: a.w.Running(),
		Supported: goruntime.GOOS == "windows", RuntimeDir: a.paths.Root}
	if external {
		_, err := os.Stat(py)
		st.Installed = err == nil
	} else {
		st.Installed = a.paths.Installed()
		st.Info, _ = a.paths.ReadInfo()
	}
	a.mu.Lock()
	st.SettingUp = a.setupCancel != nil
	a.mu.Unlock()
	return st
}

type Hardware struct {
	GPUs        []engine.GPU `json:"gpus"`
	Recommended string       `json:"recommended"`
}

func (a *App) DetectHardware() Hardware {
	g := engine.DetectGPUs(a.ctx)
	return Hardware{GPUs: g, Recommended: engine.RecommendVariant(g)}
}

func (a *App) SaveSettings(s settings.Settings) error { return a.store.Save(s) }

// ---- engine setup -----------------------------------------------------------------------------

func (a *App) SetupEngine(variant string, bitsandbytes bool) error {
	a.mu.Lock()
	if a.setupCancel != nil {
		a.mu.Unlock()
		return errors.New("setup is already running")
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.setupCancel = cancel
	a.mu.Unlock()

	a.w.Stop(3 * time.Second)
	s := a.store.Get()
	go func() {
		defer func() {
			a.mu.Lock()
			a.setupCancel = nil
			a.mu.Unlock()
			cancel()
		}()
		err := engine.Setup(ctx, a.paths, engine.Options{
			Variant: variant, Bitsandbytes: bitsandbytes, PyPIAirLLM: !s.Engine.UseBundled,
		}, engine.Reporter{
			Stage: func(i, n int, label string) {
				a.emit("setup:stage", map[string]any{"index": i, "total": n, "label": label})
			},
			Bytes: func(done, total int64) { a.emit("setup:bytes", map[string]any{"done": done, "total": total}) },
			Log:   func(l string) { a.emit("engine:log", []string{l}) },
		})
		msg := ""
		if err != nil {
			msg = err.Error()
			if errors.Is(err, context.Canceled) {
				msg = "setup cancelled"
			}
		}
		a.emit("setup:done", map[string]any{"error": msg})
	}()
	return nil
}

func (a *App) CancelSetup() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.setupCancel != nil {
		a.setupCancel()
	}
}

// ---- worker -----------------------------------------------------------------------------------

func (a *App) StartEngine() error {
	a.startMu.Lock()
	defer a.startMu.Unlock()
	if a.w.Running() {
		return nil
	}
	a.mu.Lock()
	settingUp := a.setupCancel != nil
	a.mu.Unlock()
	if settingUp {
		return errors.New("wait for engine setup to finish")
	}

	py, external := a.pythonExe()
	if _, err := os.Stat(py); err != nil {
		if external {
			return fmt.Errorf("python not found at %s", py)
		}
		return errors.New("the engine is not set up yet; open Engine and run setup")
	}
	script, err := engine.SyncApp(a.paths, a.files)
	if err != nil {
		return fmt.Errorf("prepare engine files: %w", err)
	}
	s := a.store.Get()
	a.emit("engine:state", map[string]any{"starting": true})
	err = a.w.Start(worker.Config{
		Python: py, Script: script, Dir: a.paths.App,
		Env: engine.Env(s.Engine.UseBundled, s.Engine.ModelCache),
	}, 3*time.Minute)
	a.emit("engine:state", map[string]any{"starting": false, "running": a.w.Running()})
	return err
}

func (a *App) StopEngine() { a.w.Stop(5 * time.Second) }

func (a *App) RestartEngine() error {
	a.w.Stop(5 * time.Second)
	return a.StartEngine()
}

// EngineCall runs a quick command and returns its result.
func (a *App) EngineCall(cmd string, args map[string]any) (json.RawMessage, error) {
	if err := a.StartEngine(); err != nil {
		return nil, err
	}
	return a.w.Call(cmd, args, 2*time.Minute)
}

// StartJob starts load / generate / train. Progress and the result arrive as engine:msg events.
func (a *App) StartJob(cmd string, args map[string]any) (string, error) {
	switch cmd {
	case "load", "generate", "train":
	default:
		return "", fmt.Errorf("not a job: %s", cmd)
	}
	if err := a.StartEngine(); err != nil {
		return "", err
	}
	return a.w.Job(cmd, args)
}

func (a *App) CancelJob() error {
	if !a.w.Running() {
		return nil
	}
	_, err := a.w.Call("cancel", nil, 10*time.Second)
	return err
}

// ForceStop kills the engine process. Anything loaded is gone and VRAM is released.
func (a *App) ForceStop() { a.w.Kill() }

// ---- cache ------------------------------------------------------------------------------------

func (a *App) cacheRoots() (string, []string) {
	s := a.store.Get()
	extra := append([]string(nil), s.Engine.ExtraCache...)
	if s.Model.LayerShardsSavingPath != "" {
		extra = append(extra, s.Model.LayerShardsSavingPath)
	}
	return cache.HubDir(s.Engine.ModelCache), extra
}

func (a *App) ScanCache() []cache.Entry {
	hub, extra := a.cacheRoots()
	return cache.Scan(hub, extra)
}

func (a *App) DeleteCache(path string) error {
	hub, extra := a.cacheRoots()
	return cache.Delete(path, hub, extra)
}

// ---- dialogs & shell --------------------------------------------------------------------------

func (a *App) PickFile(title, pattern string) (string, error) {
	opts := runtime.OpenDialogOptions{Title: title}
	if pattern != "" {
		opts.Filters = []runtime.FileFilter{{DisplayName: pattern, Pattern: pattern}}
	}
	return runtime.OpenFileDialog(a.ctx, opts)
}

func (a *App) PickFolder(title string) (string, error) {
	return runtime.OpenDirectoryDialog(a.ctx, runtime.OpenDialogOptions{Title: title, CanCreateDirectories: true})
}

func (a *App) SaveFile(title, name, pattern string) (string, error) {
	opts := runtime.SaveDialogOptions{Title: title, DefaultFilename: name}
	if pattern != "" {
		opts.Filters = []runtime.FileFilter{{DisplayName: pattern, Pattern: pattern}}
	}
	return runtime.SaveFileDialog(a.ctx, opts)
}

func (a *App) OpenFolder(path string) error {
	if path == "" {
		return errors.New("no path")
	}
	if fi, err := os.Stat(path); err == nil && !fi.IsDir() {
		path = filepath.Dir(path)
	}
	var cmd *exec.Cmd
	switch goruntime.GOOS {
	case "windows":
		cmd = exec.Command("explorer", path)
	case "darwin":
		cmd = exec.Command("open", path)
	default:
		cmd = exec.Command("xdg-open", path)
	}
	return cmd.Start()
}

func (a *App) OpenURL(url string) { runtime.BrowserOpenURL(a.ctx, url) }
