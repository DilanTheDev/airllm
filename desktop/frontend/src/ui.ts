import { api, errText } from "./bridge";
import { store } from "./store";

type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, any>;

/** Minimal element factory: h("div.card#id", {onclick}, ...children). */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K | string, attrs?: Attrs | null, ...children: Child[]): HTMLElement {
  const m = /^([a-z0-9-]+)?((?:[.#][\w-]+)*)$/i.exec(tag as string);
  const el = document.createElement(m?.[1] || "div");
  for (const part of (m?.[2] || "").match(/[.#][\w-]+/g) ?? []) {
    if (part[0] === ".") el.classList.add(part.slice(1));
    else el.id = part.slice(1);
  }
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "class") el.className += (el.className ? " " : "") + v;
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "html") el.innerHTML = v;
      else if (k in el && typeof v !== "string") (el as any)[k] = v;
      else el.setAttribute(k, v === true ? "" : String(v));
    }
  }
  append(el, children);
  return el;
}

export function append(el: Element, children: Child[]) {
  for (const c of children) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

// ---- icons (24px grid, 1.6 stroke) --------------------------------------------------------------

const ICONS: Record<string, string> = {
  chat: '<path d="M4 5h16v11H8l-4 4z"/>',
  model: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 4v16M15 4v16M4 9h16M4 15h16"/>',
  train: '<path d="M4 19h16"/><path d="M4 15l5-5 4 4 7-7"/><path d="M15 7h5v5"/>',
  cache: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  profile: '<path d="M12 3a9 9 0 1 0 9 9"/><path d="M12 12l6-6"/><circle cx="12" cy="12" r="1.2"/>',
  engine: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  logs: '<path d="M5 4h14v16H5z"/><path d="M8 8h8M8 12h8M8 16h5"/>',
  folder: '<path d="M3 6h6l2 2h10v11H3z"/>',
  file: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
  send: '<path d="M4 12l16-8-6 16-2-7z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  check: '<path d="M5 12l5 5 9-10"/>',
  alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
  sliders: '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v6H4V6h6"/>',
  min: '<path d="M5 12h14"/>',
  max: '<rect x="5" y="5" width="14" height="14" rx="1"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  clear: '<path d="M4 20h16"/><path d="M7 16l9-12 4 3-9 12H7z"/>',
};

export function icon(name: string, size = 16): SVGElement {
  const wrap = document.createElement("span");
  wrap.innerHTML = `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] ?? ""}</svg>`;
  return wrap.firstElementChild as SVGElement;
}

export function button(label: string | null, opts: { icon?: string; kind?: "primary" | "ghost" | "danger" | "default"; onclick?: (e: MouseEvent) => void; title?: string; small?: boolean } = {}): HTMLButtonElement {
  const b = h("button.btn", { type: "button", title: opts.title, onclick: opts.onclick }) as HTMLButtonElement;
  b.classList.add(`btn-${opts.kind ?? "default"}`);
  if (opts.small) b.classList.add("btn-sm");
  if (!label) b.classList.add("btn-icon");
  if (opts.icon) b.append(icon(opts.icon, opts.small ? 14 : 16));
  if (label) b.append(h("span", null, label));
  return b;
}

// ---- form fields bound to settings paths --------------------------------------------------------

export interface FieldOpts {
  label: string;
  path: string;
  type?: "text" | "number" | "password" | "select" | "toggle" | "textarea" | "path";
  hint?: string;
  placeholder?: string;
  options?: [string, string][];
  min?: number; max?: number; step?: number;
  float?: boolean;
  browse?: "file" | "folder" | "save";
  pattern?: string;
  wide?: boolean;
  onchange?: (v: any) => void;
}

export function field(o: FieldOpts): HTMLElement {
  const id = "f-" + o.path.replace(/\./g, "-");
  const cur = store.get(o.path);
  let control: HTMLElement;
  const commit = (v: unknown) => {
    store.set(o.path, v);
    o.onchange?.(v);
  };

  switch (o.type ?? "text") {
    case "toggle": {
      const input = h("input", { type: "checkbox", id, checked: !!cur, role: "switch" }) as HTMLInputElement;
      input.addEventListener("change", () => commit(input.checked));
      const row = h("label.toggle-row", { for: id },
        h("span.toggle-text", null, h("span.field-label", null, o.label), o.hint ? h("span.field-hint", null, o.hint) : null),
        h("span.switch", null, input, h("span.switch-track")));
      return h("div.field.field-toggle" + (o.wide ? ".wide" : ""), null, row);
    }
    case "select": {
      const sel = h("select", { id }) as HTMLSelectElement;
      for (const [v, label] of o.options ?? []) sel.append(h("option", { value: v, selected: v === cur }, label));
      if (![...sel.options].some((x) => x.value === cur) && cur) sel.append(h("option", { value: cur, selected: true }, cur));
      sel.addEventListener("change", () => commit(sel.value));
      control = sel;
      break;
    }
    case "number": {
      const input = h("input", { id, type: "number", value: cur ?? "", min: o.min, max: o.max, step: o.step ?? (o.float ? "any" : 1), inputmode: o.float ? "decimal" : "numeric" }) as HTMLInputElement;
      input.addEventListener("change", () => {
        let v = o.float ? parseFloat(input.value) : parseInt(input.value, 10);
        if (Number.isNaN(v)) v = o.min ?? 0;
        if (o.min != null && v < o.min) v = o.min;
        if (o.max != null && v > o.max) v = o.max;
        input.value = String(v);
        commit(v);
      });
      control = input;
      break;
    }
    case "textarea": {
      const ta = h("textarea", { id, rows: 3, placeholder: o.placeholder ?? "" }) as HTMLTextAreaElement;
      ta.value = cur ?? "";
      ta.addEventListener("input", () => commit(ta.value));
      control = ta;
      break;
    }
    default: {
      const input = h("input", { id, type: o.type === "password" ? "password" : "text", value: cur ?? "", placeholder: o.placeholder ?? "", spellcheck: "false", autocomplete: "off" }) as HTMLInputElement;
      input.addEventListener("input", () => commit(input.value.trim()));
      control = input;
      if (o.type === "password") {
        const reveal = button(null, { icon: "eye", kind: "ghost", small: true, title: "Show or hide" });
        reveal.addEventListener("click", () => (input.type = input.type === "password" ? "text" : "password"));
        control = h("div.input-group", null, input, reveal);
      } else if (o.browse) {
        const pick = button(null, { icon: o.browse === "folder" ? "folder" : "file", small: true, title: "Browse" });
        pick.addEventListener("click", async () => {
          try {
            const p = o.browse === "folder" ? await api.PickFolder(o.label)
              : o.browse === "save" ? await api.SaveFile(o.label, input.value.split(/[\\/]/).pop() || "", o.pattern ?? "")
                : await api.PickFile(o.label, o.pattern ?? "");
            if (p) { input.value = p; commit(p); }
          } catch (e) { toast(errText(e), "error"); }
        });
        control = h("div.input-group", null, input, pick);
      }
    }
  }
  return h("div.field" + (o.wide ? ".wide" : ""), null,
    h("label.field-label", { for: id }, o.label),
    control,
    o.hint ? h("div.field-hint", null, o.hint) : null);
}

export function section(title: string, desc: string | null, ...children: Child[]): HTMLElement {
  return h("section.card", null,
    h("header.card-head", null, h("h3", null, title), desc ? h("p", null, desc) : null),
    h("div.card-body.grid", null, ...children));
}

export function pageHeader(title: string, desc: string, ...actions: Child[]): HTMLElement {
  return h("header.page-head", null,
    h("div", null, h("h1", null, title), h("p", null, desc)),
    h("div.page-actions", null, ...actions));
}

export function badge(text: string, kind: "ok" | "warn" | "err" | "info" | "muted" = "muted"): HTMLElement {
  return h(`span.badge.badge-${kind}`, null, text);
}

// ---- toast & modal -----------------------------------------------------------------------------

let toastHost: HTMLElement | null = null;
export function toast(msg: string, kind: "info" | "ok" | "error" = "info", ms = 4200) {
  toastHost ??= document.body.appendChild(h("div.toasts", { role: "status", "aria-live": "polite" }));
  const t = h(`div.toast.toast-${kind}`, null, icon(kind === "error" ? "alert" : kind === "ok" ? "check" : "info", 16), h("span", null, msg));
  toastHost.append(t);
  const kill = () => { t.classList.add("out"); setTimeout(() => t.remove(), 180); };
  t.addEventListener("click", kill);
  setTimeout(kill, kind === "error" ? ms * 2 : ms);
}

export function confirmDialog(title: string, body: string, okLabel = "Confirm", danger = false): Promise<boolean> {
  return new Promise((resolve) => {
    const prev = document.activeElement as HTMLElement | null;
    const close = (v: boolean) => { overlay.remove(); document.removeEventListener("keydown", onKey, true); prev?.focus(); resolve(v); };
    const ok = button(okLabel, { kind: danger ? "danger" : "primary", onclick: () => close(true) });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); close(false); }
      if (e.key === "Enter") { e.preventDefault(); close(true); }
    };
    const overlay = h("div.overlay", { onmousedown: (e: MouseEvent) => { if (e.target === overlay) close(false); } },
      h("div.dialog", { role: "dialog", "aria-modal": "true", "aria-labelledby": "dlg-title" },
        h("h3#dlg-title", null, title),
        h("p", null, body),
        h("div.dialog-actions", null, button("Cancel", { kind: "ghost", onclick: () => close(false) }), ok)));
    document.body.append(overlay);
    document.addEventListener("keydown", onKey, true);
    ok.focus();
  });
}

// ---- formatting ---------------------------------------------------------------------------------

export function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i >= 3 ? 2 : i ? 1 : 0)} ${u[i]}`;
}

export function fmtSecs(s: number): string {
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${Math.round(s % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function clear(el: Element) {
  el.replaceChildren();
}
