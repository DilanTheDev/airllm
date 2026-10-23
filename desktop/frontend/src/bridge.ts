// Typed access to the Go side (window.go.main.App) and the Wails runtime (window.runtime).

export interface ModelSettings {
  model: string; hfToken: string; device: string; dtype: string; maxSeqLen: number;
  compression: string; prefetching: boolean; deleteOriginal: boolean; profilingMode: boolean;
  layerShardsSavingPath: string;
}
export interface GenerationSettings {
  maxNewTokens: number; doSample: boolean; temperature: number; topP: number; topK: number;
  repetitionPenalty: number; useCache: boolean; maxLength: number; truncation: boolean;
  padding: boolean; chatTemplate: boolean; systemPrompt: string; seed: string; mode: string;
}
export interface TrainSettings {
  trainer: string; model: string; data: string; seqLen: number; maxSeqLen: number; epochs: number;
  steps: number; loraR: number; loraAlpha: number; loraDropout: number; lr: number;
  weightDecay: number; ceChunkSize: number; packedExperts: boolean; saveAdapter: string;
  saveEvery: number; resumeAdapter: string; device: string; verbose: boolean;
}
export interface EngineSettings {
  variant: string; pythonPath: string; useBundled: boolean; bitsandbytes: boolean;
  modelCache: string; extraCache: string[] | null;
}
export interface Settings {
  model: ModelSettings; generation: GenerationSettings; train: TrainSettings; engine: EngineSettings;
}
export interface Variant { id: string; label: string; note: string }
export interface EngineState {
  installed: boolean; external: boolean; python: string; running: boolean; settingUp: boolean;
  supported: boolean; runtimeDir: string;
  info: { variant: string; python: string; installedAt: string; packages: string[] } | null;
}
export interface Boot {
  version: string; platform: string; dataDir: string; settingsPath: string; settings: Settings;
  defaults: Settings; engine: EngineState; variants: Variant[]; hubDir: string;
}
export interface GPU { name: string; memoryMB: number; driver: string }
export interface Hardware { gpus: GPU[] | null; recommended: string }
export interface CacheSplit { path: string; compression: string; bytes: number; layers: number }
export interface CacheEntry {
  name: string; path: string; bytes: number; splits: CacheSplit[] | null; hubRepo: boolean; snapshots: number;
}

export interface WorkerMsg {
  id: string | null; type: "event" | "result" | "error"; event?: string;
  data?: any; error?: string; [k: string]: any;
}

interface GoApp {
  Bootstrap(): Promise<Boot>;
  EngineStatus(): Promise<EngineState>;
  DetectHardware(): Promise<Hardware>;
  SaveSettings(s: Settings): Promise<void>;
  SetupEngine(variant: string, bnb: boolean): Promise<void>;
  CancelSetup(): Promise<void>;
  StartEngine(): Promise<void>;
  StopEngine(): Promise<void>;
  RestartEngine(): Promise<void>;
  EngineCall(cmd: string, args: Record<string, unknown> | null): Promise<any>;
  StartJob(cmd: string, args: Record<string, unknown>): Promise<string>;
  CancelJob(): Promise<void>;
  ForceStop(): Promise<void>;
  ScanCache(): Promise<CacheEntry[] | null>;
  DeleteCache(path: string): Promise<void>;
  PickFile(title: string, pattern: string): Promise<string>;
  PickFolder(title: string): Promise<string>;
  SaveFile(title: string, name: string, pattern: string): Promise<string>;
  OpenFolder(path: string): Promise<void>;
  OpenURL(url: string): Promise<void>;
}

interface WailsRuntime {
  EventsOn(name: string, cb: (...data: any[]) => void): () => void;
  WindowMinimise(): void;
  WindowToggleMaximise(): void;
  WindowIsMaximised(): Promise<boolean>;
  Quit(): void;
  ClipboardSetText(text: string): Promise<boolean>;
}

declare global {
  interface Window {
    go?: { main: { App: GoApp } };
    runtime?: WailsRuntime;
  }
}

export let api: GoApp;
export let rt: WailsRuntime;

export async function initBridge(): Promise<void> {
  if (import.meta.env.MODE === "mock" && !window.go) {
    const mock = await import("./mock");
    mock.install();
  }
  if (!window.go || !window.runtime) throw new Error("Wails bridge unavailable");
  api = window.go.main.App;
  rt = window.runtime;
  rt.EventsOn("engine:msg", onWorkerMessage);
  rt.EventsOn("engine:exit", () => failAllJobs("The engine stopped."));
}

// ---- jobs ---------------------------------------------------------------------------------------

export interface JobHandlers {
  onEvent?(m: WorkerMsg): void;
  onResult(data: any): void;
  onError(err: string): void;
}

const jobs = new Map<string, JobHandlers>();
// Messages can arrive before StartJob resolves with the id; hold them until it does.
const early = new Map<string, WorkerMsg[]>();

function onWorkerMessage(m: WorkerMsg) {
  if (!m || !m.id) return;
  const h = jobs.get(m.id);
  if (!h) {
    const list = early.get(m.id) ?? [];
    list.push(m);
    early.set(m.id, list);
    if (early.size > 32) early.delete(early.keys().next().value!);
    return;
  }
  dispatch(m.id, h, m);
}

function dispatch(id: string, h: JobHandlers, m: WorkerMsg) {
  if (m.type === "event") {
    h.onEvent?.(m);
  } else {
    jobs.delete(id);
    if (m.type === "result") h.onResult(m.data);
    else h.onError(m.error || "Unknown error");
  }
}

function failAllJobs(reason: string) {
  for (const [id, h] of jobs) {
    jobs.delete(id);
    h.onError(reason);
  }
}

export async function startJob(cmd: string, args: Record<string, unknown>, h: JobHandlers): Promise<string> {
  const id = await api.StartJob(cmd, args);
  jobs.set(id, h);
  const buffered = early.get(id);
  if (buffered) {
    early.delete(id);
    for (const m of buffered) {
      if (!jobs.has(id)) break;
      dispatch(id, h, m);
    }
  }
  return id;
}

export function errText(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
