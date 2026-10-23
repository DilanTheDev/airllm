import { api, type Boot, type EngineState, type Hardware, type Settings } from "./bridge";

export interface ModelInfo {
  model: string; architecture: string; class: string; device: string; dtype: string; layers: number;
  max_seq_len: number; compression: string | null; prefetching: boolean; profiling_mode: boolean;
  has_chat_template: boolean; load_seconds: number;
}

export interface ProfileRun {
  at: number; promptTokens: number; newTokens: number; seconds: number; tps: number;
  firstToken: number | null; profile?: Record<string, { total: number; count: number }>;
}

type Job = "load" | "generate" | "train" | null;

class Store {
  boot!: Boot;
  settings!: Settings;
  engine!: EngineState;
  hardware: Hardware | null = null;
  model: ModelInfo | null = null;
  job: Job = null;
  engineStarting = false;
  runs: ProfileRun[] = [];

  private subs = new Set<() => void>();
  private saveTimer = 0;

  init(boot: Boot) {
    this.boot = boot;
    this.settings = boot.settings;
    this.engine = boot.engine;
  }

  subscribe(fn: () => void): () => void {
    this.subs.add(fn);
    return () => this.subs.delete(fn);
  }

  notify() {
    for (const fn of this.subs) fn();
  }

  setJob(j: Job) {
    this.job = j;
    this.notify();
  }

  setModel(m: ModelInfo | null) {
    this.model = m;
    this.notify();
  }

  async refreshEngine() {
    this.engine = await api.EngineStatus();
    this.notify();
  }

  /** Read a dotted path such as "model.maxSeqLen". */
  get(path: string): any {
    return path.split(".").reduce((o: any, k) => (o == null ? o : o[k]), this.settings);
  }

  set(path: string, value: unknown) {
    const keys = path.split(".");
    let o: any = this.settings;
    for (const k of keys.slice(0, -1)) o = o[k];
    o[keys[keys.length - 1]] = value;
    this.scheduleSave();
  }

  scheduleSave() {
    clearTimeout(this.saveTimer);
    this.saveTimer = window.setTimeout(() => {
      api.SaveSettings(this.settings).catch((e) => console.error("save settings", e));
    }, 350);
  }

  flushSave(): Promise<void> {
    clearTimeout(this.saveTimer);
    return api.SaveSettings(this.settings);
  }

  addRun(r: ProfileRun) {
    this.runs.unshift(r);
    if (this.runs.length > 50) this.runs.length = 50;
  }
}

export const store = new Store();
