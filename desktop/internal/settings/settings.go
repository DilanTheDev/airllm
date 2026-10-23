// Package settings persists the app's options as JSON next to the executable, so the app stays
// portable: copy the folder and everything (settings, engine, models if you choose) moves with it.
package settings

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
)

type Model struct {
	Model                 string `json:"model"`
	HFToken               string `json:"hfToken"`
	Device                string `json:"device"`
	Dtype                 string `json:"dtype"`
	MaxSeqLen             int    `json:"maxSeqLen"`
	Compression           string `json:"compression"`
	Prefetching           bool   `json:"prefetching"`
	DeleteOriginal        bool   `json:"deleteOriginal"`
	ProfilingMode         bool   `json:"profilingMode"`
	LayerShardsSavingPath string `json:"layerShardsSavingPath"`
}

type Generation struct {
	MaxNewTokens      int     `json:"maxNewTokens"`
	DoSample          bool    `json:"doSample"`
	Temperature       float64 `json:"temperature"`
	TopP              float64 `json:"topP"`
	TopK              int     `json:"topK"`
	RepetitionPenalty float64 `json:"repetitionPenalty"`
	UseCache          bool    `json:"useCache"`
	MaxLength         int     `json:"maxLength"`
	Truncation        bool    `json:"truncation"`
	Padding           bool    `json:"padding"`
	ChatTemplate      bool    `json:"chatTemplate"`
	SystemPrompt      string  `json:"systemPrompt"`
	Seed              string  `json:"seed"`
	Mode              string  `json:"mode"` // "chat" or "completion"
}

type Train struct {
	Trainer       string  `json:"trainer"`
	Model         string  `json:"model"`
	Data          string  `json:"data"`
	SeqLen        int     `json:"seqLen"`
	MaxSeqLen     int     `json:"maxSeqLen"`
	Epochs        int     `json:"epochs"`
	Steps         int     `json:"steps"`
	LoraR         int     `json:"loraR"`
	LoraAlpha     int     `json:"loraAlpha"`
	LoraDropout   float64 `json:"loraDropout"`
	LR            float64 `json:"lr"`
	WeightDecay   float64 `json:"weightDecay"`
	CEChunkSize   int     `json:"ceChunkSize"`
	PackedExperts bool    `json:"packedExperts"`
	SaveAdapter   string  `json:"saveAdapter"`
	SaveEvery     int     `json:"saveEvery"`
	ResumeAdapter string  `json:"resumeAdapter"`
	Device        string  `json:"device"`
	Verbose       bool    `json:"verbose"`
}

type Engine struct {
	Variant      string   `json:"variant"`      // cu126, cu128, cu118, cpu
	PythonPath   string   `json:"pythonPath"`   // optional external interpreter
	UseBundled   bool     `json:"useBundled"`   // use the airllm code shipped in the exe
	Bitsandbytes bool     `json:"bitsandbytes"` // install bitsandbytes for compression
	ModelCache   string   `json:"modelCache"`   // HF_HOME override
	ExtraCache   []string `json:"extraCache"`   // extra folders to scan for split models
}

type Settings struct {
	Model      Model      `json:"model"`
	Generation Generation `json:"generation"`
	Train      Train      `json:"train"`
	Engine     Engine     `json:"engine"`
}

func Defaults() Settings {
	return Settings{
		Model: Model{
			Model: "Qwen/Qwen3-8B", Device: "auto", Dtype: "auto", MaxSeqLen: 512,
			Prefetching: true,
		},
		Generation: Generation{
			MaxNewTokens: 128, Temperature: 0.7, TopP: 0.9, TopK: 50, RepetitionPenalty: 1.0,
			UseCache: true, MaxLength: 512, Truncation: true, ChatTemplate: true, Mode: "chat",
		},
		Train: Train{
			Trainer: "AirLLMLoRA", Model: "Qwen/Qwen3.8-27B", SeqLen: 512, Epochs: 1,
			LoraR: 16, LoraAlpha: 32, LR: 1e-4, CEChunkSize: 4096, Device: "cuda:0",
		},
		Engine: Engine{Variant: "", UseBundled: true},
	}
}

type Store struct {
	mu   sync.Mutex
	path string
	cur  Settings
}

// Open loads settings from path, filling any missing fields with defaults. A missing or
// corrupt file is not an error: the app starts with defaults and rewrites it on save.
func Open(path string) *Store {
	s := &Store{path: path, cur: Defaults()}
	if b, err := os.ReadFile(path); err == nil {
		merged := Defaults()
		if json.Unmarshal(b, &merged) == nil {
			s.cur = merged
		}
	}
	return s
}

func (s *Store) Get() Settings {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cur
}

func (s *Store) Path() string { return s.path }

func (s *Store) Save(v Settings) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, s.path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	s.cur = v
	return nil
}

// DataDir returns the folder the app keeps its state in: next to the executable when that
// folder is writable (portable mode), else the per-user config directory.
func DataDir(exeDir string) (string, error) {
	probe := filepath.Join(exeDir, ".airllm-write-test")
	if f, err := os.Create(probe); err == nil {
		f.Close()
		os.Remove(probe)
		return exeDir, nil
	}
	cfg, err := os.UserConfigDir()
	if err != nil {
		return "", errors.New("no writable folder for app data")
	}
	dir := filepath.Join(cfg, "AirLLM")
	return dir, os.MkdirAll(dir, 0o755)
}
