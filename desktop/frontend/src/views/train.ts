import { api, errText, startJob } from "../bridge";
import type { View } from "../main";
import { store } from "../store";
import { button, confirmDialog, field, fmtSecs, h, pageHeader, section, toast } from "../ui";

const TRAINERS: [string, string][] = [
  ["AirLLMLoRA", "AirLLMLoRA · Qwen3.8-27B dense (~2 GB at seq 512)"],
  ["AirLLMLoRAQwen4Exp", "AirLLMLoRAQwen4Exp · Qwen3.8-Flash-Next 125B (under 6 GB)"],
];
const DEFAULT_MODEL: Record<string, string> = {
  AirLLMLoRA: "Qwen/Qwen3.8-27B",
  AirLLMLoRAQwen4Exp: "Qwen/Qwen3.8-Flash-Next",
};

interface Point { step: number; loss: number }

export function trainView(): View {
  const points: Point[] = [];
  let total = 0;
  let startedAt = 0;

  // ---- dataset preview ----
  const preview = h("div.dataset-preview");
  const previewData = async () => {
    const path = store.settings.train.data;
    preview.replaceChildren();
    if (!path) {
      preview.append(h("div.note", null, "No dataset: the run overfits a built-in snippet, which is a quick way to check that training works on your GPU."));
      return;
    }
    preview.append(h("div.muted", null, "Reading dataset…"));
    try {
      const r = await api.EngineCall("dataset_preview", { path });
      preview.replaceChildren(
        h("div.dataset-stats", null, h("strong", null, `${r.count} examples`), r.masked ? h("span.muted", null, ` · ${r.masked} with the prompt masked from the loss`) : null),
        ...r.sample.map((s: { text: string; masked: boolean }) => h("div.sample", null, s.text)));
    } catch (e) {
      preview.replaceChildren(h("div.note.note-err", null, errText(e)));
    }
  };
  const previewBtn = button("Preview", { small: true, onclick: () => previewData() });

  const modelField = () => field({ label: "Base model", path: "train.model", browse: "folder", wide: true, hint: "Repo id or local folder. Must match the trainer's architecture." });
  let modelF = modelField();
  const packed = field({ label: "Packed expert LoRA", path: "train.packedExperts", type: "toggle", hint: "Flash-Next only: adapt the packed MoE expert weights too." });
  const syncTrainer = () => packed.classList.toggle("disabled", store.settings.train.trainer !== "AirLLMLoRAQwen4Exp");

  const trainerField = field({ label: "Trainer", path: "train.trainer", type: "select", options: TRAINERS, wide: true,
    onchange: (v: string) => {
      const cur = store.settings.train.model;
      if (!cur || Object.values(DEFAULT_MODEL).includes(cur)) {
        store.set("train.model", DEFAULT_MODEL[v]);
        const fresh = modelField();
        modelF.replaceWith(fresh);
        modelF = fresh;
      }
      syncTrainer();
    } });
  const dataField = field({ label: "Dataset", path: "train.data", browse: "file", pattern: "*.jsonl;*.json;*.txt", wide: true,
    hint: "JSONL/JSON with text, prompt/completion, instruction/output or messages; or TXT with blank-line separated examples.",
    onchange: () => preview.replaceChildren() });
  dataField.querySelector(".input-group")?.append(previewBtn);

  const form = h("div.stack", null,
    section("Model and data", "Frozen base weights stream from disk one layer at a time; only the adapters stay on the GPU.",
      trainerField, modelF, dataField, preview),
    section("Schedule", null,
      field({ label: "Sequence length", path: "train.seqLen", type: "number", min: 16, max: 131072, hint: "Start at 512 on a 4 GB card" }),
      field({ label: "Max sequence (shards)", path: "train.maxSeqLen", type: "number", min: 0, max: 131072, hint: "0 uses the sequence length" }),
      field({ label: "Epochs", path: "train.epochs", type: "number", min: 1, max: 1000 }),
      field({ label: "Step limit", path: "train.steps", type: "number", min: 0, max: 10000000, hint: "0 runs every example × epochs" }),
      field({ label: "Device", path: "train.device", placeholder: "cuda:0" }),
      field({ label: "Verbose layer logs", path: "train.verbose", type: "toggle", hint: "Per-layer VRAM in the log" }),
    ),
    section("LoRA", null,
      field({ label: "Rank (r)", path: "train.loraR", type: "number", min: 1, max: 1024 }),
      field({ label: "Alpha", path: "train.loraAlpha", type: "number", min: 1, max: 4096 }),
      field({ label: "Dropout", path: "train.loraDropout", type: "number", float: true, min: 0, max: 0.9, step: 0.05, hint: "Keep 0: the backward recompute assumes it" }),
      field({ label: "Learning rate", path: "train.lr", type: "number", float: true, min: 0, max: 1, step: 0.00001 }),
      field({ label: "Weight decay", path: "train.weightDecay", type: "number", float: true, min: 0, max: 1, step: 0.01 }),
      field({ label: "Cross-entropy chunk", path: "train.ceChunkSize", type: "number", min: 128, max: 262144, hint: "Smaller uses less VRAM for the vocab projection" }),
      packed,
    ),
    section("Adapter", "Uses the token and shard folder from the Model page.",
      field({ label: "Save adapter to", path: "train.saveAdapter", browse: "save", pattern: "*.pt", placeholder: "adapter.pt", wide: true }),
      field({ label: "Checkpoint every N steps", path: "train.saveEvery", type: "number", min: 0, max: 1000000, hint: "0 saves only at the end" }),
      field({ label: "Resume from adapter", path: "train.resumeAdapter", browse: "file", pattern: "*.pt", placeholder: "optional", wide: true }),
    ),
  );

  // ---- run panel ----
  const canvas = h("canvas.chart") as HTMLCanvasElement;
  const statEls = { step: h("div.stat-v", null, "–"), loss: h("div.stat-v", null, "–"), vram: h("div.stat-v", null, "–"), eta: h("div.stat-v", null, "–") };
  const stat = (k: string, v: HTMLElement) => h("div.stat", null, h("div.stat-k", null, k), v);
  const stageEl = h("div.run-stage", null, "Idle");
  const startBtn = button("Start training", { kind: "primary", onclick: () => start() });
  const stopBtn = button("Stop", { kind: "danger", onclick: () => stop() });
  const run = h("section.card.run-card", null,
    h("div.run-head", null, stageEl, h("div.page-actions", null, stopBtn, startBtn)),
    h("div.stats", null, stat("Step", statEls.step), stat("Loss", statEls.loss), stat("Peak VRAM", statEls.vram), stat("Remaining", statEls.eta)),
    h("div.chart-wrap", null, canvas));

  const el = h("div.page-inner", null,
    pageHeader("Train", "Fine-tune a LoRA adapter on huge models with a small GPU. Needs the CUDA engine."),
    run, form);

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, hgt = canvas.clientHeight;
    if (!w || !hgt) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hgt * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(hgt * dpr);
    }
    const c = canvas.getContext("2d")!;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, hgt);
    const css = getComputedStyle(document.documentElement);
    const grid = css.getPropertyValue("--border").trim(), muted = css.getPropertyValue("--text-3").trim();
    const accent = css.getPropertyValue("--accent").trim();
    const pad = { l: 44, r: 12, t: 10, b: 22 };
    const iw = w - pad.l - pad.r, ih = hgt - pad.t - pad.b;
    c.font = "11px " + css.getPropertyValue("--font-mono");
    c.fillStyle = muted;
    c.strokeStyle = grid;
    c.lineWidth = 1;
    if (!points.length) {
      c.textAlign = "center";
      c.fillText("Loss appears here once training starts", w / 2, hgt / 2);
      return;
    }
    let lo = Infinity, hi = -Infinity;
    for (const p of points) { lo = Math.min(lo, p.loss); hi = Math.max(hi, p.loss); }
    if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
    const span = hi - lo; lo -= span * 0.08; hi += span * 0.08;
    const maxStep = Math.max(total - 1, points[points.length - 1].step, 1);
    const x = (s: number) => pad.l + (s / maxStep) * iw;
    const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * ih;
    c.textAlign = "right";
    c.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4, yy = Math.round(y(v)) + 0.5;
      c.beginPath(); c.moveTo(pad.l, yy); c.lineTo(w - pad.r, yy); c.stroke();
      c.fillText(v.toFixed(v < 10 ? 3 : 1), pad.l - 6, yy);
    }
    c.textAlign = "center";
    c.textBaseline = "top";
    c.fillText("0", pad.l, hgt - pad.b + 6);
    c.fillText(String(maxStep), w - pad.r, hgt - pad.b + 6);
    c.strokeStyle = accent;
    c.lineWidth = 1.75;
    c.lineJoin = "round";
    c.beginPath();
    points.forEach((p, i) => (i ? c.lineTo(x(p.step), y(p.loss)) : c.moveTo(x(p.step), y(p.loss))));
    c.stroke();
    if (points.length === 1) {
      c.fillStyle = accent;
      c.beginPath(); c.arc(x(points[0].step), y(points[0].loss), 3, 0, Math.PI * 2); c.fill();
    }
  }
  let drawFrame = 0;
  const scheduleDraw = () => { if (!drawFrame) drawFrame = requestAnimationFrame(() => { drawFrame = 0; draw(); }); };
  new ResizeObserver(scheduleDraw).observe(canvas);

  async function start() {
    if (store.job) return;
    const t = store.settings.train, m = store.settings.model;
    if (store.model && !(await confirmDialog("Unload the current model?", "Training loads its own copy of the base model, so the model loaded for generation is unloaded first.", "Unload and train"))) return;
    await store.flushSave();
    points.length = 0;
    total = 0;
    scheduleDraw();
    statEls.step.textContent = statEls.loss.textContent = statEls.vram.textContent = statEls.eta.textContent = "–";
    stageEl.textContent = "Starting engine";
    store.setJob("train");
    try {
      await startJob("train", {
        trainer: t.trainer, model: t.model, data: t.data || null, seq_len: t.seqLen, max_seq_len: t.maxSeqLen || null,
        epochs: t.epochs, steps: t.steps || null, lora_r: t.loraR, lora_alpha: t.loraAlpha, lora_dropout: t.loraDropout,
        lr: t.lr, weight_decay: t.weightDecay, ce_chunk_size: t.ceChunkSize, packed_experts: t.packedExperts,
        save_adapter: t.saveAdapter || null, save_every: t.saveEvery, resume_adapter: t.resumeAdapter || null,
        device: t.device || null, verbose: t.verbose,
        hf_token: m.hfToken || null, layer_shards_saving_path: m.layerShardsSavingPath || null,
        prefetching: m.prefetching, delete_original: m.deleteOriginal, compression: m.compression || null,
      }, {
        onEvent(e) {
          if (e.event === "stage") {
            stageEl.textContent = e.stage;
            if (e.total) { total = e.total; startedAt = performance.now(); }
            store.setModel(null);
          } else if (e.event === "train_step") {
            points.push({ step: e.step, loss: e.loss });
            statEls.step.textContent = `${e.step + 1} / ${e.total}`;
            statEls.loss.textContent = e.loss.toFixed(4);
            statEls.vram.textContent = e.peak_vram_gb ? `${e.peak_vram_gb.toFixed(2)} GB` : "–";
            const per = (performance.now() - startedAt) / 1000 / (e.step + 1);
            statEls.eta.textContent = fmtSecs(Math.max(0, (e.total - e.step - 1) * per));
            stageEl.textContent = `Training · epoch ${e.epoch + 1}`;
            scheduleDraw();
          }
        },
        onResult(r) {
          store.setJob(null);
          stageEl.textContent = r.cancelled ? `Stopped after ${r.steps} steps` : `Finished ${r.steps} steps in ${fmtSecs(r.seconds)}`;
          statEls.eta.textContent = "–";
          toast(r.saved_adapter ? `Adapter saved to ${r.saved_adapter}` : "Training finished (no adapter path set, nothing saved)", r.saved_adapter ? "ok" : "info");
        },
        onError(err) {
          store.setJob(null);
          stageEl.textContent = "Training failed";
          toast(err, "error");
        },
      });
    } catch (e) {
      store.setJob(null);
      stageEl.textContent = "Engine unavailable";
      toast(errText(e), "error");
    }
  }

  async function stop() {
    try {
      await api.CancelJob();
      stageEl.textContent = "Stopping after the current step…";
    } catch (e) { toast(errText(e), "error"); }
  }

  const render = () => {
    const training = store.job === "train";
    startBtn.disabled = !!store.job;
    stopBtn.hidden = !training;
    previewBtn.disabled = !!store.job;
  };
  store.subscribe(render);
  render();
  syncTrainer();
  return { el, shown: scheduleDraw };
}
