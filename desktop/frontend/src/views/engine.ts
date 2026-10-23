import { api, errText, rt } from "../bridge";
import { consoleView } from "../logs";
import type { View } from "../main";
import { store } from "../store";
import { badge, button, confirmDialog, field, fmtBytes, h, pageHeader, section, toast } from "../ui";

export function engineView(): View {
  // ---- status ----
  const statusBody = h("div.info-grid");
  const statusBadge = h("span");
  const startBtn = button("Start", { onclick: () => act(() => api.StartEngine(), "Engine started") });
  const restartBtn = button("Restart", { onclick: () => act(() => api.RestartEngine(), "Engine restarted") });
  const stopBtn = button("Stop", { onclick: () => act(() => api.StopEngine(), "Engine stopped") });
  const statusCard = h("section.card", null,
    h("header.card-head.row", null, h("div", null, h("h3", null, "Status"), h("p", null, "The engine is a private Python process that runs AirLLM. It starts on demand.")),
      h("div.page-actions", null, statusBadge, startBtn, restartBtn, stopBtn)),
    h("div.card-body", null, statusBody));

  // ---- packages (from the running engine) ----
  const pkgBody = h("div.info-grid");
  const pkgCard = h("section.card", null,
    h("header.card-head.row", null, h("div", null, h("h3", null, "Environment"), h("p", null, "Versions and devices as PyTorch sees them.")),
      button("Check", { icon: "refresh", small: true, onclick: () => loadEnv() })),
    h("div.card-body", null, pkgBody));

  // ---- setup ----
  const variantList = h("div.variants", { role: "radiogroup" });
  let chosen = store.settings.engine.variant;
  const renderVariants = () => {
    const rec = store.hardware?.recommended;
    if (!chosen) chosen = rec ?? "cu126";
    variantList.replaceChildren(...store.boot.variants.map((v) => {
      const input = h("input", { type: "radio", name: "variant", value: v.id, checked: v.id === chosen }) as HTMLInputElement;
      input.addEventListener("change", () => { chosen = v.id; store.set("engine.variant", v.id); });
      return h("label.variant", null, input, h("div", null,
        h("div.variant-title", null, v.label, v.id === rec ? badge("Recommended", "ok") : null),
        h("div.variant-note", null, v.note)));
    }));
  };
  const gpuLine = h("div.muted");
  const stageEl = h("div.load-stage");
  const subEl = h("div.load-sub");
  const bar = h("div.progress", null, h("div.progress-fill"));
  const setupBtn = button("Set up engine", { kind: "primary", onclick: () => setup() });
  const cancelBtn = button("Cancel", { kind: "danger", onclick: () => api.CancelSetup() });
  const log = consoleView(400);
  log.el.classList.add("console-sm");
  const setupCard = h("section.card", null,
    h("header.card-head", null, h("h3", null, "Setup"),
      h("p", null, "Downloads Python and PyTorch into the runtime folder next to this app. Nothing is installed system-wide. CUDA builds are about 2.5 GB.")),
    h("div.card-body.stack-sm", null,
      gpuLine, variantList,
      field({ label: "Install bitsandbytes", path: "engine.bitsandbytes", type: "toggle", hint: "Needed for 4-bit / 8-bit compression. NVIDIA only." }),
      h("div.setup-progress", null, h("div.load-row", null, h("div.load-text", null, stageEl, subEl), h("div.page-actions", null, cancelBtn, setupBtn)), bar),
      log.el));

  const advanced = section("Advanced", null,
    field({ label: "Use an existing Python", path: "engine.pythonPath", browse: "file", pattern: "python.exe", wide: true,
      placeholder: "Leave empty to use the app's own engine",
      hint: "A python.exe that already has torch and transformers. Restart the engine after changing it.", onchange: () => store.refreshEngine() }),
    field({ label: "Use the AirLLM code built into this app", path: "engine.useBundled", type: "toggle",
      hint: "Off imports airllm from the Python environment instead (pip install airllm)." }),
  );

  const el = h("div.page-inner", null,
    pageHeader("Engine", "Python, PyTorch and AirLLM, managed for you."),
    statusCard, setupCard, pkgCard, advanced);

  async function act(fn: () => Promise<void>, ok: string) {
    if (store.job && !(await confirmDialog("Stop the running job?", "The engine is busy. Continuing interrupts it and unloads the model.", "Continue", true))) return;
    try {
      await store.flushSave();
      await fn();
      store.setModel(null);
      toast(ok, "ok");
    } catch (e) { toast(errText(e), "error"); }
    store.refreshEngine();
  }

  let envLoaded = false;
  async function loadEnv() {
    pkgBody.replaceChildren(h("div.muted", null, "Starting the engine…"));
    try {
      const r = await api.EngineCall("env", null);
      envLoaded = true;
      const gpus: { index: number; name: string; total_gb: number }[] = r.gpus ?? [];
      pkgBody.replaceChildren(
        kv("Python", r.python),
        kv("CUDA in PyTorch", r.cuda_available ? `yes · CUDA ${r.cuda_version}` : "no"),
        ...gpus.map((g) => kv(`cuda:${g.index}`, `${g.name} · ${g.total_gb} GB`)),
        ...Object.entries(r.packages as Record<string, string | null>).map(([k, v]) => kv(k, v ?? "not installed", !v)),
        ...(r.airllm_path ? [kv("airllm from", r.airllm_path, false, true)] : []),
        ...(r.airllm_error ? [kv("airllm error", r.airllm_error, true)] : []),
      );
      store.refreshEngine();
    } catch (e) {
      pkgBody.replaceChildren(h("div.note.note-err", null, errText(e)));
    }
  }

  async function setup() {
    const e = store.engine;
    if (e.external) { toast("Clear “Use an existing Python” first; setup only manages the built-in engine.", "info"); return; }
    if (e.installed && !(await confirmDialog("Repair or switch the engine?", "Setup re-runs pip against the selected PyTorch build. Already-downloaded parts are reused where possible.", "Run setup"))) return;
    try {
      await store.flushSave();
      store.setModel(null);
      await api.SetupEngine(chosen, store.settings.engine.bitsandbytes);
      store.engine.settingUp = true;
      stageEl.textContent = "Preparing";
      subEl.textContent = "";
      setBar(null);
      store.notify();
    } catch (err) { toast(errText(err), "error"); }
  }

  function setBar(p: number | null) {
    bar.classList.toggle("indeterminate", p == null);
    (bar.firstElementChild as HTMLElement).style.width = p == null ? "" : `${p}%`;
  }

  rt.EventsOn("setup:stage", (d: { index: number; total: number; label: string }) => {
    stageEl.textContent = `Step ${d.index} of ${d.total} · ${d.label}`;
    subEl.textContent = "";
    setBar(null);
  });
  rt.EventsOn("setup:bytes", (d: { done: number; total: number }) => {
    if (d.total > 0) {
      setBar((d.done / d.total) * 100);
      subEl.textContent = `${fmtBytes(d.done)} of ${fmtBytes(d.total)}`;
    } else subEl.textContent = fmtBytes(d.done);
  });
  rt.EventsOn("setup:done", async (d: { error: string }) => {
    await store.refreshEngine();
    setBar(100);
    if (d.error) {
      stageEl.textContent = d.error === "setup cancelled" ? "Setup cancelled" : "Setup failed";
      subEl.textContent = d.error === "setup cancelled" ? "Run setup again to continue where it stopped." : d.error;
      if (d.error !== "setup cancelled") toast(d.error, "error");
    } else {
      stageEl.textContent = "Engine ready";
      subEl.textContent = "Load a model on the Model page.";
      toast("Engine set up", "ok");
      loadEnv();
    }
  });

  const render = () => {
    const e = store.engine;
    statusBadge.replaceChildren(e.settingUp ? badge("Setting up", "warn") : e.running ? badge("Running", "ok") : e.installed ? badge("Stopped", "muted") : badge("Not set up", "err"));
    startBtn.hidden = e.running;
    startBtn.disabled = !e.installed || e.settingUp || store.engineStarting;
    restartBtn.hidden = stopBtn.hidden = !e.running;
    statusBody.replaceChildren(
      kv("Mode", e.external ? "Existing Python" : "Built-in engine"),
      kv("Python", e.python, false, true),
      kv("Build", e.info ? `${e.info.variant} · Python ${e.info.python}` : e.external ? "external" : "–"),
      kv("Installed", e.info ? new Date(e.info.installedAt).toLocaleString() : "–"),
      kv("Runtime folder", e.runtimeDir, false, true),
      kv("Settings file", store.boot.settingsPath, false, true),
    );
    const settingUp = e.settingUp;
    setupBtn.hidden = settingUp;
    cancelBtn.hidden = !settingUp;
    setupBtn.disabled = !e.supported || !!store.job;
    setupBtn.querySelector("span")!.textContent = e.installed && !e.external ? "Repair / switch" : "Set up engine";
    bar.hidden = !settingUp;
    if (!settingUp && !stageEl.textContent) {
      stageEl.textContent = e.installed ? "Engine installed" : e.supported ? "Not set up yet" : "Setup is Windows only";
      subEl.textContent = e.installed ? "Run setup again to repair it or switch PyTorch builds." : e.supported ? "Choose a build and click Set up engine." : "Use an existing Python under Advanced.";
    }
    const gpus = store.hardware?.gpus ?? [];
    gpuLine.textContent = store.hardware == null ? "Detecting GPUs…" : gpus.length
      ? `Detected: ${gpus.map((g) => `${g.name} (${(g.memoryMB / 1024).toFixed(0)} GB, driver ${g.driver})`).join(", ")}`
      : "No NVIDIA GPU detected. CPU only will work, slowly.";
    renderVariants();
  };
  store.subscribe(render);
  render();
  return { el, shown: () => { if (!envLoaded && store.engine.installed && !store.engine.settingUp) loadEnv(); } };
}

function kv(k: string, v: string, bad = false, mono = false): HTMLElement {
  return h("div.kv", null, h("div.kv-k", null, k), h(`div.kv-v${bad ? ".kv-bad" : ""}${mono ? ".mono.selectable" : ""}`, null, v));
}
