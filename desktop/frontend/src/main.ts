import "./styles.css";
import { api, errText, initBridge, rt } from "./bridge";
import { addLines } from "./logs";
import { store } from "./store";
import { button, clear, h, icon, toast } from "./ui";
import { chatView } from "./views/chat";
import { modelView } from "./views/model";
import { trainView } from "./views/train";
import { cacheView } from "./views/cache";
import { profileView } from "./views/profile";
import { engineView } from "./views/engine";
import { logsView } from "./views/logs";

export interface View { el: HTMLElement; dispose?(): void; shown?(): void }

const ROUTES: { id: string; label: string; icon: string; make: () => View }[] = [
  { id: "chat", label: "Generate", icon: "chat", make: chatView },
  { id: "model", label: "Model", icon: "model", make: modelView },
  { id: "train", label: "Train", icon: "train", make: trainView },
  { id: "cache", label: "Storage", icon: "cache", make: cacheView },
  { id: "profile", label: "Profiling", icon: "profile", make: profileView },
  { id: "engine", label: "Engine", icon: "engine", make: engineView },
  { id: "logs", label: "Logs", icon: "logs", make: logsView },
];

// Views are built once and kept, so switching tabs never loses a half-typed prompt or a
// running chart, and never re-renders a page from scratch.
const built = new Map<string, View>();
let current = "";
let main: HTMLElement;
const navButtons = new Map<string, HTMLElement>();

export function navigate(id: string) {
  if (id === current) return;
  const route = ROUTES.find((r) => r.id === id) ?? ROUTES[0];
  current = route.id;
  let v = built.get(route.id);
  if (!v) {
    v = route.make();
    v.el.classList.add("page");
    built.set(route.id, v);
    main.append(v.el);
  }
  for (const [rid, view] of built) view.el.hidden = rid !== route.id;
  for (const [rid, b] of navButtons) b.classList.toggle("active", rid === route.id);
  v.shown?.();
  try { localStorage.setItem("airllm.route", route.id); } catch { /* storage unavailable */ }
}

function titlebar(): HTMLElement {
  const maxBtn = button(null, { icon: "max", kind: "ghost", title: "Maximize", onclick: () => rt.WindowToggleMaximise() });
  return h("header.titlebar", null,
    h("div.brand", null, h("span.brand-mark"), h("span.brand-name", null, "AirLLM"), h("span.brand-ver", null, store.boot.version)),
    h("div.titlebar-fill"),
    h("div.winctl", null,
      button(null, { icon: "min", kind: "ghost", title: "Minimize", onclick: () => rt.WindowMinimise() }),
      maxBtn,
      Object.assign(button(null, { icon: "close", kind: "ghost", title: "Close", onclick: () => rt.Quit() }), { className: "btn btn-ghost btn-icon winctl-close" })));
}

function statusPanel(): HTMLElement {
  const dot = h("span.dot");
  const line1 = h("div.status-title");
  const line2 = h("div.status-sub");
  const el = h("div.sidebar-status", { role: "button", tabindex: 0, title: "Open Engine", onclick: () => navigate("engine") }, dot, h("div", null, line1, line2));
  const render = () => {
    const e = store.engine;
    let state = "off", t = "Engine not set up", s = "Open Engine to install";
    if (e.settingUp) { state = "busy"; t = "Setting up engine"; s = "Downloading packages"; }
    else if (store.engineStarting) { state = "busy"; t = "Starting engine"; s = "Importing PyTorch"; }
    else if (store.job) { state = "busy"; t = store.job === "load" ? "Loading model" : store.job === "train" ? "Training" : "Generating"; s = store.job === "train" ? store.settings.train.model : store.model?.model ?? store.settings.model.model; }
    else if (store.model) { state = "ok"; t = shortName(store.model.model); s = `${store.model.device} · ${store.model.dtype}`; }
    else if (e.running) { state = "ok"; t = "Engine ready"; s = "No model loaded"; }
    else if (e.installed) { state = "idle"; t = "Engine installed"; s = "Starts when needed"; }
    dot.className = "dot dot-" + state;
    line1.textContent = t;
    line2.textContent = s;
  };
  store.subscribe(render);
  render();
  return el;
}

export function shortName(m: string): string {
  const parts = m.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || m;
}

function sidebar(): HTMLElement {
  const nav = h("nav.nav", { "aria-label": "Sections" });
  for (const r of ROUTES) {
    const b = h("button.nav-item", { type: "button", onclick: () => navigate(r.id) }, icon(r.icon, 18), h("span", null, r.label));
    navButtons.set(r.id, b);
    nav.append(b);
  }
  return h("aside.sidebar", null, nav, h("div.sidebar-fill"), statusPanel());
}

function wireEngineEvents() {
  rt.EventsOn("engine:log", (lines: string[]) => addLines(lines));
  rt.EventsOn("engine:exit", (d: { error: string }) => {
    const hadModel = !!store.model;
    store.model = null;
    store.job = null;
    store.refreshEngine();
    if (hadModel || d?.error) addLines([`[app] engine stopped${d?.error ? ": " + d.error : ""}`]);
  });
  rt.EventsOn("engine:state", (d: { starting: boolean }) => {
    store.engineStarting = !!d.starting;
    store.refreshEngine();
  });
}

function shortcuts() {
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && !e.shiftKey && !e.altKey && /^[1-7]$/.test(e.key)) {
      e.preventDefault();
      navigate(ROUTES[Number(e.key) - 1].id);
    }
    if (e.key === "Escape" && store.job && !document.querySelector(".overlay")) {
      api.CancelJob().catch(() => undefined);
    }
  });
  // No browser context menu or zoom: this is an app, not a web page.
  document.addEventListener("contextmenu", (e) => {
    const t = e.target as HTMLElement;
    if (!t.closest("input, textarea, .console, .msg-body, .selectable")) e.preventDefault();
  });
}

async function boot() {
  const root = document.getElementById("app")!;
  try {
    await initBridge();
    store.init(await api.Bootstrap());
  } catch (e) {
    root.append(h("div.fatal", null, h("h1", null, "AirLLM could not start"), h("p", null, errText(e))));
    return;
  }
  wireEngineEvents();
  shortcuts();
  main = h("main.main");
  clear(root);
  root.append(titlebar(), h("div.body", null, sidebar(), main));
  let first = "chat";
  try { first = localStorage.getItem("airllm.route") || first; } catch { /* ignore */ }
  if (!store.engine.installed) first = "engine";
  navigate(first);
  api.DetectHardware().then((hw) => { store.hardware = hw; store.notify(); }).catch(() => undefined);
  window.addEventListener("error", (e) => toast(e.message, "error"));
  window.addEventListener("unhandledrejection", (e) => toast(errText(e.reason), "error"));
}

boot();
