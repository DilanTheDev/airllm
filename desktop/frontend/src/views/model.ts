import { api, errText, startJob } from "../bridge";
import { shortName, type View } from "../main";
import { store, type ModelInfo } from "../store";
import { button, confirmDialog, field, fmtSecs, h, pageHeader, section, toast } from "../ui";

// Popular checkpoints and the VRAM the README measures for them.
const PRESETS: [string, string][] = [
  ["Qwen/Qwen3-8B", "Qwen3 8B · ~1–2 GB"],
  ["Qwen/Qwen3-32B", "Qwen3 32B · ~2 GB"],
  ["Qwen/Qwen3-30B-A3B", "Qwen3 30B MoE · ~1–3 GB"],
  ["Qwen/Qwen3.8-27B", "Qwen3.8 27B VL · 3.33 GB"],
  ["meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B · gated"],
  ["meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B · ~4 GB · gated"],
  ["mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B"],
  ["mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B MoE"],
  ["microsoft/phi-4", "Phi-4 14B"],
  ["google/gemma-2-9b-it", "Gemma 2 9B · gated"],
  ["Qwen/Qwen3-235B-A22B", "Qwen3 235B MoE · ~3 GB"],
  ["meta-llama/Llama-3.1-405B-Instruct", "Llama 3.1 405B · ~8 GB · gated"],
  ["deepseek-ai/DeepSeek-V3", "DeepSeek-V3 671B · ~12 GB"],
  ["Qwen/Qwen3.8-Flash-Next", "Qwen3.8 Flash-Next 125B · 5.95 GB"],
];

export function modelView(): View {
  const deviceOpts = (): [string, string][] => {
    const opts: [string, string][] = [["auto", "Automatic (best available)"]];
    const gpus = store.hardware?.gpus ?? [];
    gpus.forEach((g, i) => opts.push([`cuda:${i}`, `cuda:${i} · ${g.name} · ${(g.memoryMB / 1024).toFixed(0)} GB`]));
    if (!gpus.length) opts.push(["cuda:0", "cuda:0"]);
    opts.push(["cpu", "CPU"]);
    return opts;
  };

  const preset = h("select.preset", { "aria-label": "Popular models" }) as HTMLSelectElement;
  preset.append(h("option", { value: "" }, "Popular models…"));
  for (const [id, label] of PRESETS) preset.append(h("option", { value: id }, label));

  const modelField = field({ label: "Model", path: "model.model", placeholder: "org/name or a local folder", browse: "folder", wide: true,
    hint: "A Hugging Face repo id, or a folder that holds config.json and the weights." });
  const modelInput = modelField.querySelector("input") as HTMLInputElement;
  preset.addEventListener("change", () => {
    if (!preset.value) return;
    modelInput.value = preset.value;
    store.set("model.model", preset.value);
    preset.value = "";
  });
  modelField.querySelector(".field-label")!.after(preset);

  let deviceField = field({ label: "Device", path: "model.device", type: "select", options: deviceOpts() });

  const compressionWarn = h("div.note.note-warn", { hidden: true },
    "Compression needs bitsandbytes (Engine → Setup) and turns prefetching off.");
  const syncCompression = () => (compressionWarn.hidden = !store.settings.model.compression);
  const deleteWarnToggle = field({ label: "Delete original after split", path: "model.deleteOriginal", type: "toggle",
    hint: "Keeps only the layer shards. Halves disk use; the original must be downloaded again to re-split.",
    onchange: async (v: boolean) => {
      if (v && !(await confirmDialog("Delete originals after splitting?", "AirLLM will remove the downloaded checkpoint once its layers are split. Re-splitting later (for example with a different compression) needs a fresh download.", "Enable", true))) {
        store.set("model.deleteOriginal", false);
        (deleteWarnToggle.querySelector("input") as HTMLInputElement).checked = false;
      }
    } });

  const sections = h("div.stack", null,
    section("Source", "Where the weights come from.",
      modelField,
      field({ label: "Hugging Face token", path: "model.hfToken", type: "password", placeholder: "hf_…", hint: "Only needed for gated models such as Llama. Stored in settings.json beside the app." }),
    ),
    section("Runtime", "Where layers run and at what precision.",
      deviceField,
      field({ label: "Precision", path: "model.dtype", type: "select", options: [
        ["auto", "Model default (usually bfloat16)"], ["bfloat16", "bfloat16"], ["float16", "float16"], ["float32", "float32"]],
        hint: "float16 can overflow on very deep models; keep the default unless you know otherwise." }),
      field({ label: "Max sequence length", path: "model.maxSeqLen", type: "number", min: 16, max: 1048576 }),
    ),
    section("Speed", "Trade-offs between load time, memory and accuracy.",
      field({ label: "Compression", path: "model.compression", type: "select", options: [["", "None (full precision)"], ["4bit", "4-bit block-wise (up to 3× faster)"], ["8bit", "8-bit block-wise (up to 2× faster)"]], onchange: syncCompression }),
      field({ label: "Prefetching", path: "model.prefetching", type: "toggle", hint: "Load the next layer from disk while the current one computes." }),
      field({ label: "Profiling", path: "model.profilingMode", type: "toggle", hint: "Record per-stage timings. Results appear under Profiling." }),
      compressionWarn,
    ),
    section("Storage", "Splitting writes one file per layer. Make sure the drive has room.",
      field({ label: "Layer shard folder", path: "model.layerShardsSavingPath", browse: "folder", placeholder: "Default: next to the downloaded model", wide: true }),
      deleteWarnToggle,
    ),
  );

  // ---- load bar ----
  const stage = h("div.load-stage");
  const bar = h("div.progress", null, h("div.progress-fill"));
  const barText = h("div.load-sub");
  const loadBtn = button("Load model", { kind: "primary", onclick: () => load() });
  const unloadBtn = button("Unload", { onclick: () => unload() });
  const stopBtn = button("Force stop", { kind: "danger", title: "Kill the engine process. Loading can't be paused part way.", onclick: () => forceStop() });
  const info = h("div.info-grid");
  const loadCard = h("section.card.load-card", null,
    h("div.load-row", null, h("div.load-text", null, stage, barText), h("div.page-actions", null, stopBtn, unloadBtn, loadBtn)),
    bar, info);

  const el = h("div.page-inner", null,
    pageHeader("Model", "Choose a model and how AirLLM streams it. Settings apply the next time you load."),
    loadCard, sections);

  let progressOff: (() => void) | null = null;
  async function load() {
    if (store.job) return;
    const m = store.settings.model;
    if (!m.model) { toast("Enter a model id or folder", "error"); return; }
    await store.flushSave();
    store.setJob("load");
    setStage("Starting engine", "");
    setBar(null);
    progressOff = window.runtime!.EventsOn("engine:progress", (p: { label: string; pct: number; text: string }) => {
      setBar(p.pct);
      barText.textContent = [p.label, p.text].filter(Boolean).join(" · ");
    });
    const t0 = performance.now();
    const done = () => { progressOff?.(); progressOff = null; store.setJob(null); };
    try {
      await startJob("load", {
        model: m.model, hf_token: m.hfToken || null, device: m.device, dtype: m.dtype, max_seq_len: m.maxSeqLen,
        compression: m.compression || null, prefetching: m.prefetching, delete_original: m.deleteOriginal,
        profiling_mode: m.profilingMode, layer_shards_saving_path: m.layerShardsSavingPath || null,
      }, {
        onEvent(e) {
          if (e.event === "stage") { setStage(e.stage, e.architecture ? `Architecture: ${e.architecture}` : ""); setBar(null); }
        },
        onResult(r: ModelInfo) {
          done();
          store.setModel(r);
          toast(`Loaded ${shortName(r.model)} in ${fmtSecs((performance.now() - t0) / 1000)}`, "ok");
        },
        onError(err) {
          done();
          store.setModel(null);
          setStage("Load failed", err);
          toast(err, "error");
        },
      });
      setStage("Resolving model", "");
    } catch (e) {
      done();
      setStage("Engine unavailable", errText(e));
      toast(errText(e), "error");
    }
  }

  async function unload() {
    if (store.job) return;
    try {
      await api.EngineCall("unload", null);
      store.setModel(null);
    } catch (e) { toast(errText(e), "error"); }
  }

  async function forceStop() {
    if (!(await confirmDialog("Force stop the engine?", "This kills the engine process. Anything loaded is unloaded and its memory is freed. Split shards already written are kept.", "Force stop", true))) return;
    await api.ForceStop();
    store.setJob(null);
    store.setModel(null);
    setStage("Stopped", "");
  }

  function setStage(t: string, sub: string) { stage.textContent = t; barText.textContent = sub; }
  function setBar(pct: number | null) {
    bar.classList.toggle("indeterminate", pct == null);
    (bar.firstElementChild as HTMLElement).style.width = pct == null ? "" : `${pct}%`;
  }

  let lastDevices = "";
  const render = () => {
    const loading = store.job === "load";
    const mm = store.model;
    loadBtn.disabled = !!store.job;
    loadBtn.querySelector("span")!.textContent = mm ? "Reload" : "Load model";
    unloadBtn.hidden = !mm || loading;
    unloadBtn.disabled = !!store.job;
    stopBtn.hidden = !loading;
    bar.hidden = !loading;
    if (!loading) {
      if (mm) setStage(shortName(mm.model), `Loaded in ${fmtSecs(mm.load_seconds)}`);
      else if (!stage.textContent || stage.textContent === "Stopped") setStage("No model loaded", "Load downloads the model if needed, then splits it into per-layer shards (once).");
    }
    info.replaceChildren(...(mm ? [
      kv("Architecture", mm.architecture || "unknown"), kv("AirLLM class", mm.class), kv("Device", mm.device),
      kv("Precision", mm.dtype), kv("Layers streamed", String(mm.layers)), kv("Max sequence", String(mm.max_seq_len)),
      kv("Compression", mm.compression ?? "none"), kv("Prefetching", mm.prefetching ? "on" : "off"),
      kv("Chat template", mm.has_chat_template ? "yes" : "no"),
    ] : []));
    info.hidden = !mm;
    // Refresh the device list once GPU detection finishes.
    const sig = JSON.stringify(store.hardware?.gpus ?? []);
    if (sig !== lastDevices) {
      lastDevices = sig;
      const fresh = field({ label: "Device", path: "model.device", type: "select", options: deviceOpts() });
      deviceField.replaceWith(fresh);
      deviceField = fresh;
    }
  };
  store.subscribe(render);
  render();
  syncCompression();
  return { el };
}

function kv(k: string, v: string): HTMLElement {
  return h("div.kv", null, h("div.kv-k", null, k), h("div.kv-v", null, v));
}
