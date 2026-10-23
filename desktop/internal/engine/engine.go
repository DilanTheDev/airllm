// Package engine sets up the private Python + PyTorch runtime the worker runs in.
//
// Layout under <dataDir>/runtime:
//
//	python/        embeddable CPython (Windows) with pip and the packages
//	app/           the worker script and the bundled airllm source, rewritten when they change
//	engine.json    what was installed (variant, versions), written after a successful setup
package engine

import (
	"archive/zip"
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"airllm-desktop/internal/worker"
)

const (
	PythonVersion = "3.12.10"
	pythonTag     = "312"
	getPipURL     = "https://bootstrap.pypa.io/get-pip.py"
)

// Variants of PyTorch the setup can install, keyed by the PyTorch wheel index suffix.
var Variants = []Variant{
	{ID: "cu126", Label: "NVIDIA GPU · CUDA 12.6", Note: "GTX 10 series through RTX 40 series"},
	{ID: "cu128", Label: "NVIDIA GPU · CUDA 12.8", Note: "Required for RTX 50 series; also runs older cards"},
	{ID: "cu118", Label: "NVIDIA GPU · CUDA 11.8", Note: "Older drivers (below 525)"},
	{ID: "cpu", Label: "CPU only", Note: "No GPU acceleration; slow, but works everywhere"},
}

type Variant struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Note  string `json:"note"`
}

// BasePackages mirrors air_llm/setup.py install_requires.
var BasePackages = []string{
	"transformers>=4.49,<6", "accelerate>=1.0", "safetensors", "huggingface-hub", "scipy",
	"sentencepiece", "tqdm", "protobuf",
}

type Info struct {
	Variant     string    `json:"variant"`
	Python      string    `json:"python"`
	InstalledAt time.Time `json:"installedAt"`
	Packages    []string  `json:"packages"`
}

type Paths struct {
	Root   string
	Python string // python dir
	App    string
}

func NewPaths(dataDir string) Paths {
	root := filepath.Join(dataDir, "runtime")
	return Paths{Root: root, Python: filepath.Join(root, "python"), App: filepath.Join(root, "app")}
}

func (p Paths) PythonExe() string {
	if runtime.GOOS == "windows" {
		return filepath.Join(p.Python, "python.exe")
	}
	return filepath.Join(p.Python, "bin", "python3")
}

func (p Paths) infoFile() string { return filepath.Join(p.Root, "engine.json") }

func (p Paths) ReadInfo() (*Info, error) {
	b, err := os.ReadFile(p.infoFile())
	if err != nil {
		return nil, err
	}
	var i Info
	if err := json.Unmarshal(b, &i); err != nil {
		return nil, err
	}
	return &i, nil
}

// Installed reports whether a completed setup exists.
func (p Paths) Installed() bool {
	if _, err := os.Stat(p.PythonExe()); err != nil {
		return false
	}
	_, err := p.ReadInfo()
	return err == nil
}

// SyncApp writes the embedded worker files into p.App when their content changed. It returns
// the path of the worker script.
func SyncApp(p Paths, files fs.FS) (string, error) {
	h := sha256.New()
	var names []string
	err := fs.WalkDir(files, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || strings.Contains(path, "__pycache__") || strings.HasSuffix(path, ".pyc") {
			return nil
		}
		names = append(names, path)
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Strings(names)
	for _, n := range names {
		b, err := fs.ReadFile(files, n)
		if err != nil {
			return "", err
		}
		h.Write([]byte(n))
		h.Write(b)
	}
	sum := hex.EncodeToString(h.Sum(nil))
	stamp := filepath.Join(p.App, ".stamp")
	script := filepath.Join(p.App, "airllm_worker.py")
	if b, err := os.ReadFile(stamp); err == nil && string(b) == sum {
		if _, err := os.Stat(script); err == nil {
			return script, nil
		}
	}
	if err := os.RemoveAll(p.App); err != nil {
		return "", err
	}
	for _, n := range names {
		dst := filepath.Join(p.App, filepath.FromSlash(n))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return "", err
		}
		b, _ := fs.ReadFile(files, n)
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			return "", err
		}
	}
	return script, os.WriteFile(stamp, []byte(sum), 0o644)
}

// Env returns environment variables the worker and pip run with.
func Env(useBundled bool, modelCache string) []string {
	env := []string{
		"PYTHONUTF8=1", "PYTHONIOENCODING=utf-8", "PYTHONUNBUFFERED=1",
		"HF_HUB_DISABLE_SYMLINKS_WARNING=1", "TRANSFORMERS_NO_ADVISORY_WARNINGS=1",
		"PYTHONNOUSERSITE=1",
	}
	if useBundled {
		env = append(env, "AIRLLM_WORKER_USE_BUNDLED=1")
	} else {
		env = append(env, "AIRLLM_WORKER_USE_BUNDLED=0")
	}
	if modelCache != "" {
		env = append(env, "HF_HOME="+modelCache)
	}
	return env
}

// ---- setup ------------------------------------------------------------------------------------

type Reporter struct {
	Stage func(index, total int, label string)
	Bytes func(done, total int64)
	Log   func(line string)
}

type Options struct {
	Variant      string
	Bitsandbytes bool
	PyPIAirLLM   bool // install airllm from PyPI as well (used when not running the bundled copy)
	Client       *http.Client
}

func pythonZipURL() string {
	return fmt.Sprintf("https://www.python.org/ftp/python/%s/python-%s-embed-amd64.zip", PythonVersion, PythonVersion)
}

func TorchIndex(variant string) string { return "https://download.pytorch.org/whl/" + variant }

func validVariant(v string) bool {
	for _, x := range Variants {
		if x.ID == v {
			return true
		}
	}
	return false
}

// Setup installs (or repairs) the runtime. It is safe to re-run: finished steps are skipped
// where possible and pip reinstalls only what is missing or wrong.
func Setup(ctx context.Context, p Paths, opt Options, r Reporter) error {
	if runtime.GOOS != "windows" {
		return errors.New("one-click engine setup is only available on Windows; choose an existing Python instead")
	}
	if !validVariant(opt.Variant) {
		return fmt.Errorf("unknown engine variant %q", opt.Variant)
	}
	client := opt.Client
	if client == nil {
		client = &http.Client{Timeout: 0}
	}
	const total = 5
	// PyTorch ships files with deep paths; past Windows' 260-character limit pip fails part way.
	if len(p.Python) > 70 {
		r.Log(fmt.Sprintf("warning: the app folder path is long (%d characters). If PyTorch fails to install, "+
			"move AirLLM.exe to a short folder such as C:\\AirLLM or enable Windows long paths.", len(p.Python)))
	}
	if err := os.MkdirAll(p.Root, 0o755); err != nil {
		return err
	}
	_ = os.Remove(p.infoFile())

	// 1. Python
	r.Stage(1, total, "Downloading Python "+PythonVersion)
	if _, err := os.Stat(p.PythonExe()); err != nil {
		zipPath := filepath.Join(p.Root, "python-embed.zip")
		if err := Download(ctx, client, pythonZipURL(), zipPath, r.Bytes); err != nil {
			return fmt.Errorf("download python: %w", err)
		}
		_ = os.RemoveAll(p.Python)
		if err := Unzip(zipPath, p.Python); err != nil {
			return fmt.Errorf("extract python: %w", err)
		}
		os.Remove(zipPath)
	}
	if err := EnableSite(p.Python); err != nil {
		return err
	}

	// 2. pip
	r.Stage(2, total, "Installing pip")
	py := p.PythonExe()
	if err := run(ctx, py, []string{"-m", "pip", "--version"}, nil, nil); err != nil {
		gp := filepath.Join(p.Root, "get-pip.py")
		if err := Download(ctx, client, getPipURL, gp, r.Bytes); err != nil {
			return fmt.Errorf("download get-pip: %w", err)
		}
		if err := run(ctx, py, []string{gp, "--no-warn-script-location"}, Env(true, ""), r.Log); err != nil {
			return fmt.Errorf("install pip: %w", err)
		}
		os.Remove(gp)
	}

	pip := func(args ...string) error {
		base := []string{"-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check",
			"--progress-bar", "off", "--no-cache-dir"}
		return run(ctx, py, append(base, args...), Env(true, ""), r.Log)
	}

	// 3. torch (the big one, up to ~2.5 GB for CUDA builds)
	r.Stage(3, total, "Installing PyTorch ("+opt.Variant+"), this is the large download")
	if err := pip("torch", "--index-url", TorchIndex(opt.Variant)); err != nil {
		return fmt.Errorf("install torch: %w", err)
	}

	// 4. airllm dependencies
	r.Stage(4, total, "Installing transformers and AirLLM dependencies")
	pkgs := append([]string(nil), BasePackages...)
	// PyTorch's Windows DLLs need msvcp140.dll, which the embeddable Python lacks and fresh
	// Windows installs may too. This wheel puts the runtime next to python.exe, so nothing is
	// installed system-wide.
	pkgs = append(pkgs, "msvc-runtime")
	if opt.Bitsandbytes {
		pkgs = append(pkgs, "bitsandbytes")
	}
	if opt.PyPIAirLLM {
		pkgs = append(pkgs, "airllm")
	}
	if err := pip(pkgs...); err != nil {
		return fmt.Errorf("install packages: %w", err)
	}

	// 5. verify
	r.Stage(5, total, "Verifying")
	var out strings.Builder
	check := "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
	if err := run(ctx, py, []string{"-c", check}, Env(true, ""), func(l string) { out.WriteString(l); r.Log(l) }); err != nil {
		if strings.Contains(err.Error(), "WinError 126") || strings.Contains(err.Error(), "DLL load failed") {
			return fmt.Errorf("PyTorch could not load its DLLs (%w). Installing the Microsoft Visual C++ Redistributable (x64) usually fixes this", err)
		}
		return fmt.Errorf("verify: %w", err)
	}
	info := Info{Variant: opt.Variant, Python: PythonVersion, InstalledAt: time.Now(), Packages: append(pkgs, "torch")}
	b, _ := json.MarshalIndent(info, "", "  ")
	return os.WriteFile(p.infoFile(), b, 0o644)
}

// EnableSite turns on site-packages for the embeddable distribution, which ships with it off.
func EnableSite(pyDir string) error {
	pth := filepath.Join(pyDir, "python"+pythonTag+"._pth")
	b, err := os.ReadFile(pth)
	if err != nil {
		return fmt.Errorf("python ._pth file missing: %w", err)
	}
	s := strings.ReplaceAll(string(b), "\r\n", "\n")
	lines := strings.Split(s, "\n")
	hasSite, hasLib := false, false
	for i, l := range lines {
		t := strings.TrimSpace(l)
		if t == "#import site" {
			lines[i] = "import site"
		}
		if strings.TrimSpace(lines[i]) == "import site" {
			hasSite = true
		}
		if t == `Lib\site-packages` || t == "Lib/site-packages" {
			hasLib = true
		}
	}
	if !hasLib {
		lines = append([]string{`Lib\site-packages`}, lines...)
	}
	if !hasSite {
		lines = append(lines, "import site")
	}
	return os.WriteFile(pth, []byte(strings.Join(lines, "\r\n")), 0o644)
}

func run(ctx context.Context, exe string, args []string, env []string, logf func(string)) error {
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Env = append(os.Environ(), env...)
	worker.HideWindow(cmd)
	pr, pw := io.Pipe()
	cmd.Stdout, cmd.Stderr = pw, pw
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan struct{})
	var tail []string
	go func() {
		defer close(done)
		sc := bufio.NewScanner(pr)
		sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for sc.Scan() {
			l := strings.TrimRight(sc.Text(), "\r ")
			if l == "" {
				continue
			}
			tail = append(tail, l)
			if len(tail) > 8 {
				tail = tail[1:]
			}
			if logf != nil {
				logf(l)
			}
		}
	}()
	err := cmd.Wait()
	pw.Close()
	<-done
	if err != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if err != nil && len(tail) > 0 {
		return fmt.Errorf("%w: %s", err, tail[len(tail)-1])
	}
	return err
}

// Download fetches url into dst via a temp file, reporting progress.
func Download(ctx context.Context, c *http.Client, url, dst string, progress func(done, total int64)) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "AirLLM-Desktop")
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	var done int64
	buf := make([]byte, 256*1024)
	last := time.Time{}
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				os.Remove(tmp)
				return werr
			}
			done += int64(n)
			if progress != nil && time.Since(last) > 100*time.Millisecond {
				progress(done, resp.ContentLength)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			os.Remove(tmp)
			return rerr
		}
	}
	if progress != nil {
		progress(done, resp.ContentLength)
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

// Unzip extracts src into dir, refusing entries that would escape it.
func Unzip(src, dir string) error {
	zr, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer zr.Close()
	root, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	for _, f := range zr.File {
		dst := filepath.Join(root, filepath.FromSlash(f.Name))
		if dst != root && !strings.HasPrefix(dst, root+string(os.PathSeparator)) {
			return fmt.Errorf("zip entry escapes target: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(dst, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(dst)
		if err != nil {
			rc.Close()
			return err
		}
		_, err = io.Copy(out, rc)
		rc.Close()
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// ---- GPU detection ------------------------------------------------------------------------------

type GPU struct {
	Name     string `json:"name"`
	MemoryMB int    `json:"memoryMB"`
	Driver   string `json:"driver"`
}

// DetectGPUs asks nvidia-smi (installed with every NVIDIA driver) which cards are present.
func DetectGPUs(ctx context.Context) []GPU {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
		"--format=csv,noheader,nounits")
	worker.HideWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	return ParseNvidiaSMI(string(out))
}

func ParseNvidiaSMI(out string) []GPU {
	var gpus []GPU
	for _, line := range strings.Split(strings.ReplaceAll(out, "\r\n", "\n"), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}
		var mem int
		fmt.Sscanf(strings.TrimSpace(parts[1]), "%d", &mem)
		gpus = append(gpus, GPU{Name: strings.TrimSpace(parts[0]), MemoryMB: mem, Driver: strings.TrimSpace(parts[2])})
	}
	return gpus
}

// RecommendVariant picks the PyTorch build for the detected hardware.
func RecommendVariant(gpus []GPU) string {
	if len(gpus) == 0 {
		return "cpu"
	}
	for _, g := range gpus {
		n := strings.ToUpper(g.Name)
		if strings.Contains(n, "RTX 50") || strings.Contains(n, "RTX PRO") || strings.Contains(n, "B200") || strings.Contains(n, "B100") {
			return "cu128"
		}
	}
	var major int
	fmt.Sscanf(gpus[0].Driver, "%d", &major)
	if major > 0 && major < 525 {
		return "cu118"
	}
	return "cu126"
}
