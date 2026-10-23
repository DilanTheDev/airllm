// Browser-only stand-in for the Go side, used by `npm run dev` and the UI tests.
// It is only bundled in --mode mock builds.

import type { Boot, CacheEntry, Settings } from "./bridge";

type Cb = (...a: any[]) => void;

export function install() {
  const listeners = new Map<string, Set<Cb>>();
  const emit = (name: string, data: any) => listeners.get(name)?.forEach((cb) => setTimeout(() => cb(data), 0));
  const params = new URLSearchParams(location.search);
  let installed = params.get("installed") !== "0";
  let running = false;
  let model: any = null;
  let cancel = false;
  let busy = false;
  let seq = 0;
  let settings: Settings = defaults();
  const calls: { method: string; args: any[] }[] = [];
  (window as any).__mockCalls = calls;

  const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const engineState = () => ({ installed, external: false, python: "C:\\AirLLM\\runtime\\python\\python.exe", running,
    settingUp: false, supported: true, runtimeDir: "C:\\AirLLM\\runtime",
    info: installed ? { variant: "cu126", python: "3.12.10", installedAt: new Date().toISOString(), packages: [] } : null });

  const job = async (id: string, cmd: string, args: any) => {
    const ev = (event: string, data: any) => emit("engine:msg", { id, type: "event", event, ...data });
    const done = (data: any) => { busy = false; emit("engine:msg", { id, type: "result", data }); };
    const fail = (error: string) => { busy = false; emit("engine:msg", { id, type: "error", error }); };
    busy = true;
    cancel = false;
    if (cmd === "load") {
      ev("stage", { stage: "Resolving model" });
      await delay(150);
      ev("stage", { stage: "Downloading and splitting layers", architecture: "Qwen3ForCausalLM" });
      for (let p = 0; p <= 100; p += 20) {
        emit("engine:progress", { label: "Splitting layers", pct: p, text: `${p / 4}/25` });
        emit("engine:log", [`saved model.layers.${p / 4}`]);
        await delay(60);
      }
      if (String(args.model).includes("fail")) return fail("OSError: model not found");
      model = { model: args.model, architecture: "Qwen3ForCausalLM", class: "AirLLMBaseModel", device: "cuda:0",
        dtype: "bfloat16", layers: 39, max_seq_len: args.max_seq_len, compression: args.compression,
        prefetching: args.prefetching, profiling_mode: args.profiling_mode, has_chat_template: true, load_seconds: 2.4 };
      return done(model);
    }
    if (cmd === "generate") {
      if (!model) return fail("RuntimeError: no model loaded");
      const words = "Washington, D.C. is the capital of the United States. It sits on the Potomac River between Maryland and Virginia.".split(" ");
      let text = "";
      let n = 0;
      for (const w of words.slice(0, args.max_new_tokens)) {
        if (cancel) break;
        await delay(40);
        const piece = (n ? " " : "") + w;
        text += piece;
        n++;
        ev("token", { text: piece, tokens: n });
      }
      return done({ text, prompt_tokens: 14, new_tokens: n, seconds: n * 0.04, tokens_per_second: 25, first_token_seconds: 0.05,
        cancelled: cancel, profile: model.profiling_mode ? { load_safe_tensor: { total: 1.82, count: 39 * n }, compression_time: { total: 0.12, count: 39 * n } } : undefined });
    }
    if (cmd === "train") {
      ev("stage", { stage: "Loading base model" });
      await delay(100);
      const total = 24;
      ev("stage", { stage: "Training", total });
      let i = 0;
      for (; i < total && !cancel; i++) {
        await delay(50);
        ev("train_step", { step: i, epoch: 0, loss: 2.6 * Math.exp(-i / 9) + 0.4 + Math.random() * 0.08, peak_vram_gb: 1.93, seq: 512, total, elapsed: i * 0.05 });
      }
      return done({ steps: i, final_loss: 0.5, mean_loss: 1.1, saved_adapter: args.save_adapter, cancelled: cancel, seconds: i * 0.05 });
    }
    return fail("unknown");
  };

  const api = {
    Bootstrap: async (): Promise<Boot> => ({ version: "1.0.0", platform: "windows", dataDir: "C:\\AirLLM", settingsPath: "C:\\AirLLM\\settings.json",
      settings, defaults: defaults(), engine: engineState(), hubDir: "C:\\Users\\you\\.cache\\huggingface\\hub",
      variants: [
        { id: "cu126", label: "NVIDIA GPU · CUDA 12.6", note: "GTX 10 series through RTX 40 series" },
        { id: "cu128", label: "NVIDIA GPU · CUDA 12.8", note: "Required for RTX 50 series; also runs older cards" },
        { id: "cu118", label: "NVIDIA GPU · CUDA 11.8", note: "Older drivers (below 525)" },
        { id: "cpu", label: "CPU only", note: "No GPU acceleration; slow, but works everywhere" }] }),
    EngineStatus: async () => engineState(),
    DetectHardware: async () => { await delay(80); return { gpus: [{ name: "NVIDIA GeForce RTX 3060", memoryMB: 12288, driver: "560.94" }], recommended: "cu126" }; },
    SaveSettings: async (s: Settings) => { settings = JSON.parse(JSON.stringify(s)); calls.push({ method: "SaveSettings", args: [settings] }); },
    SetupEngine: async (variant: string) => {
      calls.push({ method: "SetupEngine", args: [variant] });
      (async () => {
        const labels = ["Downloading Python 3.12.10", "Installing pip", "Installing PyTorch", "Installing transformers", "Verifying"];
        for (let i = 0; i < labels.length; i++) {
          emit("setup:stage", { index: i + 1, total: 5, label: labels[i] });
          for (let b = 0; b <= 4; b++) { emit("setup:bytes", { done: b * 2.5e6, total: 1e7 }); await delay(30); }
          emit("engine:log", [`[setup] ${labels[i]} ok`]);
        }
        installed = true;
        emit("setup:done", { error: "" });
      })();
    },
    CancelSetup: async () => undefined,
    StartEngine: async () => { if (!installed) throw "the engine is not set up yet; open Engine and run setup"; running = true; },
    StopEngine: async () => { running = false; model = null; emit("engine:exit", { error: "" }); },
    RestartEngine: async () => { running = true; model = null; },
    EngineCall: async (cmd: string, args: any) => {
      if (!installed) throw "the engine is not set up yet; open Engine and run setup";
      running = true;
      calls.push({ method: "EngineCall", args: [cmd, args] });
      if (cmd === "cancel") { cancel = true; return { cancelling: busy }; }
      if (cmd === "unload") { model = null; return { unloaded: true }; }
      if (cmd === "env") return { python: "3.12.10", cuda_available: true, cuda_version: "12.6", gpus: [{ index: 0, name: "NVIDIA GeForce RTX 3060", total_gb: 12 }],
        packages: { torch: "2.8.0+cu126", transformers: "4.56.2", accelerate: "1.10.1", safetensors: "0.6.2", huggingface_hub: "0.35.3", bitsandbytes: null, airllm: "installed" },
        airllm_path: "C:\\AirLLM\\runtime\\app\\airllm" };
      if (cmd === "dataset_preview") return { count: 2, masked: 1, sample: [{ text: "Your first training document.", masked: false }, { text: "What is AirLLM?\nA library.", masked: true }] };
      return {};
    },
    StartJob: async (cmd: string, args: any) => {
      if (!installed) throw "the engine is not set up yet; open Engine and run setup";
      if (busy) throw "busy";
      running = true;
      calls.push({ method: "StartJob", args: [cmd, args] });
      const id = "q" + ++seq;
      job(id, cmd, args);
      return id;
    },
    CancelJob: async () => { cancel = true; },
    ForceStop: async () => { cancel = true; running = false; model = null; busy = false; emit("engine:exit", { error: "" }); },
    ScanCache: async (): Promise<CacheEntry[]> => [
      { name: "Qwen/Qwen3-8B", path: "C:\\Users\\you\\.cache\\huggingface\\hub\\models--Qwen--Qwen3-8B", bytes: 32.8e9, hubRepo: true, snapshots: 1,
        splits: [{ path: "C:\\...\\splitted_model", compression: "", bytes: 16.4e9, layers: 39 }, { path: "C:\\...\\splitted_model.4bit", compression: "4bit", bytes: 4.6e9, layers: 39 }] },
      { name: "mistralai/Mistral-7B-Instruct-v0.3", path: "C:\\Users\\you\\.cache\\huggingface\\hub\\models--mistralai--Mistral-7B-Instruct-v0.3", bytes: 14.5e9, hubRepo: true, snapshots: 1, splits: null },
    ],
    DeleteCache: async (p: string) => { calls.push({ method: "DeleteCache", args: [p] }); },
    PickFile: async () => "C:\\data\\train.jsonl",
    PickFolder: async () => "D:\\models",
    SaveFile: async () => "C:\\adapters\\out.pt",
    OpenFolder: async () => undefined,
    OpenURL: async () => undefined,
  };

  const runtime = {
    EventsOn(name: string, cb: Cb) {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name)!.add(cb);
      return () => listeners.get(name)!.delete(cb);
    },
    WindowMinimise() {}, WindowToggleMaximise() {}, WindowIsMaximised: async () => false, Quit() {},
    ClipboardSetText: async () => true,
  };
  window.go = { main: { App: api as any } };
  window.runtime = runtime as any;
}

function defaults(): Settings {
  return {
    model: { model: "Qwen/Qwen3-8B", hfToken: "", device: "auto", dtype: "auto", maxSeqLen: 512, compression: "", prefetching: true, deleteOriginal: false, profilingMode: false, layerShardsSavingPath: "" },
    generation: { maxNewTokens: 128, doSample: false, temperature: 0.7, topP: 0.9, topK: 50, repetitionPenalty: 1, useCache: true, maxLength: 512, truncation: true, padding: false, chatTemplate: true, systemPrompt: "", seed: "", mode: "chat" },
    train: { trainer: "AirLLMLoRA", model: "Qwen/Qwen3.8-27B", data: "", seqLen: 512, maxSeqLen: 0, epochs: 1, steps: 0, loraR: 16, loraAlpha: 32, loraDropout: 0, lr: 0.0001, weightDecay: 0, ceChunkSize: 4096, packedExperts: false, saveAdapter: "", saveEvery: 0, resumeAdapter: "", device: "cuda:0", verbose: false },
    engine: { variant: "", pythonPath: "", useBundled: true, bitsandbytes: false, modelCache: "", extraCache: null },
  };
}
