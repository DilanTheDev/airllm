import { api, errText, startJob } from "../bridge";
import { navigate, shortName, type View } from "../main";
import { store } from "../store";
import { button, field, fmtSecs, h, toast } from "../ui";

interface Msg { role: "user" | "assistant" | "system"; content: string; meta?: string; error?: boolean }

export function chatView(): View {
  const history: Msg[] = [];
  let streaming: { body: HTMLElement; pane: HTMLElement; msg: Msg; buf: string; frame: number } | null = null;

  // ---- header ----
  const modelChip = h("button.model-chip", { type: "button", onclick: () => navigate("model"), title: "Model settings" });
  const modeSeg = h("div.segmented", { role: "tablist" });
  const setMode = (m: string) => {
    store.set("generation.mode", m);
    for (const b of modeSeg.children) b.classList.toggle("active", (b as HTMLElement).dataset.mode === m);
    chatPane.hidden = m !== "chat";
    compPane.hidden = m !== "completion";
    composer.placeholder = m === "chat" ? "Message the model…" : "Write a prompt to continue…";
  };
  for (const [m, label] of [["chat", "Chat"], ["completion", "Completion"]]) {
    modeSeg.append(h("button", { type: "button", "data-mode": m, onclick: () => setMode(m) }, label));
  }
  const clearBtn = button("Clear", { icon: "clear", kind: "ghost", small: true, onclick: () => clearAll() });
  const paramsBtn = button(null, { icon: "sliders", kind: "ghost", small: true, title: "Generation settings", onclick: () => root.classList.toggle("params-hidden") });

  // ---- conversation ----
  const list = h("div.messages.selectable", { "aria-live": "polite" });
  const empty = h("div.empty", null,
    h("div.empty-title", null, "Start a conversation"),
    h("div.empty-sub", null, "Load a model on the Model page, then send a message. Ctrl+Enter sends, Esc stops."),
  );
  const chatPane = h("div.chat-scroll", null, empty, list);

  const compOut = h("div.completion-out.selectable");
  const compPane = h("div.chat-scroll.completion", null, compOut);

  // ---- composer ----
  const composer = h("textarea.composer-input", { rows: 1, placeholder: "Message the model…", spellcheck: "true" }) as HTMLTextAreaElement;
  const sendBtn = button("Send", { icon: "send", kind: "primary", onclick: () => send() });
  const stopBtn = button("Stop", { icon: "stop", kind: "danger", onclick: () => api.CancelJob().catch((e) => toast(errText(e), "error")) });
  const stats = h("div.composer-stats");
  const autosize = () => {
    composer.style.height = "auto";
    composer.style.height = Math.min(composer.scrollHeight, 220) + "px";
  };
  composer.addEventListener("input", autosize);
  composer.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
  });

  // ---- parameters ----
  const buildParams = (): HTMLElement => {
    const sampleFields = [
      field({ label: "Temperature", path: "generation.temperature", type: "number", float: true, min: 0.01, max: 5, step: 0.05 }),
      field({ label: "Top-p", path: "generation.topP", type: "number", float: true, min: 0, max: 1, step: 0.01 }),
      field({ label: "Top-k", path: "generation.topK", type: "number", min: 0, max: 1000, hint: "0 disables top-k" }),
      field({ label: "Seed", path: "generation.seed", placeholder: "random", hint: "Fixed seed for repeatable sampling" }),
    ];
    const syncSample = () => sampleFields.forEach((f) => f.classList.toggle("disabled", !store.settings.generation.doSample));
    syncSample();
    return h("aside.params", null,
      h("div.params-head", null, h("h3", null, "Generation"), button("Reset", { kind: "ghost", small: true, onclick: () => resetParams() })),
      h("div.params-body", null,
        h("div.params-group", null,
          field({ label: "Max new tokens", path: "generation.maxNewTokens", type: "number", min: 1, max: 32768 }),
          field({ label: "Sample", path: "generation.doSample", type: "toggle", hint: "Off uses greedy decoding", onchange: syncSample }),
          ...sampleFields,
          field({ label: "Repetition penalty", path: "generation.repetitionPenalty", type: "number", float: true, min: 0.5, max: 3, step: 0.05 }),
        ),
        h("div.params-group", null,
          h("div.params-sub", null, "Prompt"),
          field({ label: "System prompt", path: "generation.systemPrompt", type: "textarea", placeholder: "Optional. Chat mode only." }),
          field({ label: "Apply chat template", path: "generation.chatTemplate", type: "toggle", hint: "Uses the tokenizer's template when it has one" }),
          field({ label: "Max input tokens", path: "generation.maxLength", type: "number", min: 8, max: 1048576, hint: "Older chat turns are dropped past this" }),
          field({ label: "Truncate input", path: "generation.truncation", type: "toggle" }),
          field({ label: "Pad input", path: "generation.padding", type: "toggle" }),
          field({ label: "KV cache", path: "generation.useCache", type: "toggle", hint: "Faster decoding; uses more memory" }),
        )));
  };
  let params = buildParams();

  const root = h("div.chat", null,
    h("div.chat-main", null,
      h("header.chat-head", null, modelChip, modeSeg, h("div.titlebar-fill"), clearBtn, paramsBtn),
      chatPane, compPane,
      h("div.composer", null,
        h("div.composer-box", null, composer, h("div.composer-actions", null, stats, stopBtn, sendBtn)))),
    params);

  // ---- behaviour ----
  function resetParams() {
    const d = store.boot.defaults.generation;
    Object.assign(store.settings.generation, { ...d, mode: store.settings.generation.mode, systemPrompt: store.settings.generation.systemPrompt });
    store.scheduleSave();
    const fresh = buildParams();
    params.replaceWith(fresh);
    params = fresh;
    toast("Generation settings reset", "ok");
  }

  function clearAll() {
    if (store.job === "generate") return;
    history.length = 0;
    list.replaceChildren();
    compOut.textContent = "";
    empty.hidden = false;
    stats.textContent = "";
  }

  function addBubble(m: Msg): HTMLElement {
    empty.hidden = true;
    const body = h("div.msg-body", null, m.content);
    const copy = button(null, { icon: "copy", kind: "ghost", small: true, title: "Copy", onclick: () => { rt_copy(m.content); } });
    const el = h(`div.msg.msg-${m.role}`, null,
      h("div.msg-role", null, m.role === "user" ? "You" : m.role === "assistant" ? "Model" : "System"),
      body,
      h("div.msg-foot", null, h("span.msg-meta"), copy));
    if (m.error) el.classList.add("msg-error");
    list.append(el);
    chatPane.scrollTop = chatPane.scrollHeight;
    return el;
  }

  function rt_copy(text: string) {
    (window.runtime?.ClipboardSetText(text) ?? navigator.clipboard.writeText(text).then(() => true))
      .then(() => toast("Copied", "ok", 1500)).catch(() => toast("Copy failed", "error"));
  }

  function flushStream() {
    if (!streaming) return;
    streaming.frame = 0;
    if (!streaming.buf) return;
    const { pane } = streaming;
    const stick = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
    streaming.body.append(document.createTextNode(streaming.buf));
    streaming.msg.content += streaming.buf;
    streaming.buf = "";
    if (stick) pane.scrollTop = pane.scrollHeight;
  }

  async function send() {
    const text = composer.value.trim();
    if (!text || store.job) return;
    if (!store.model) {
      toast("Load a model first", "info");
      navigate("model");
      return;
    }
    const g = store.settings.generation;
    const completion = g.mode === "completion";
    const args: Record<string, unknown> = {
      max_new_tokens: g.maxNewTokens, do_sample: g.doSample, temperature: g.temperature, top_p: g.topP,
      top_k: g.topK, repetition_penalty: g.repetitionPenalty, use_cache: g.useCache, max_length: g.maxLength,
      truncation: g.truncation, padding: g.padding, chat_template: g.chatTemplate, seed: g.seed || null,
    };
    let assistant: Msg;
    let bubble: HTMLElement;
    if (completion) {
      args.prompt = text;
      compOut.replaceChildren(h("span.prompt-echo", null, text));
      assistant = { role: "assistant", content: "" };
      bubble = compOut;
    } else {
      const user: Msg = { role: "user", content: text };
      history.push(user);
      addBubble(user);
      const msgs = history.filter((m) => !m.error).map((m) => ({ role: m.role, content: m.content }));
      if (g.systemPrompt.trim()) msgs.unshift({ role: "system", content: g.systemPrompt.trim() });
      args.messages = msgs;
      assistant = { role: "assistant", content: "" };
      bubble = addBubble(assistant);
      bubble.classList.add("streaming");
    }
    composer.value = "";
    autosize();
    const body = completion ? compOut : (bubble.querySelector(".msg-body") as HTMLElement);
    streaming = { body, pane: completion ? compPane : chatPane, msg: assistant, buf: "", frame: 0 };
    const t0 = performance.now();
    let tokens = 0;
    store.setJob("generate");
    const tick = window.setInterval(() => {
      const s = (performance.now() - t0) / 1000;
      stats.textContent = tokens ? `${tokens} tokens · ${fmtSecs(s)}` : `Waiting for first token · ${fmtSecs(s)}`;
    }, 250);

    const finish = (meta: string, err?: string) => {
      clearInterval(tick);
      flushStream();
      if (streaming?.frame) cancelAnimationFrame(streaming.frame);
      streaming = null;
      store.setJob(null);
      if (!completion) {
        bubble.classList.remove("streaming");
        (bubble.querySelector(".msg-meta") as HTMLElement).textContent = meta;
        if (err) {
          assistant.error = true;
          bubble.classList.add("msg-error");
          if (!assistant.content) (bubble.querySelector(".msg-body") as HTMLElement).textContent = err;
        }
        if (!assistant.error || assistant.content) history.push(assistant);
      }
      stats.textContent = meta;
      composer.focus();
    };

    try {
      await startJob("generate", args, {
        onEvent(m) {
          if (m.event !== "token" || !streaming) return;
          tokens = m.tokens ?? tokens + 1;
          streaming.buf += m.text;
          if (!streaming.frame) streaming.frame = requestAnimationFrame(flushStream);
        },
        onResult(r) {
          // The final decode is authoritative: the stream can hold back a trailing partial word.
          flushStream();
          const shown = assistant.content;
          if (r.text && r.text !== shown) {
            if (r.text.startsWith(shown)) {
              if (streaming) streaming.buf = r.text.slice(shown.length);
            } else {
              body.textContent = completion ? "" : r.text;
              if (completion) body.append(h("span.prompt-echo", null, text), r.text);
              assistant.content = r.text;
            }
          }
          store.addRun({ at: Date.now(), promptTokens: r.prompt_tokens, newTokens: r.new_tokens, seconds: r.seconds,
            tps: r.tokens_per_second, firstToken: r.first_token_seconds, profile: r.profile });
          store.notify();
          const meta = `${r.new_tokens} tokens · ${r.tokens_per_second.toFixed(2)} tok/s · ${fmtSecs(r.seconds)}${r.cancelled ? " · stopped" : ""}`;
          finish(meta);
        },
        onError(err) {
          finish("Failed", err);
          toast(err, "error");
        },
      });
    } catch (e) {
      finish("Failed", errText(e));
      toast(errText(e), "error");
    }
  }

  const render = () => {
    const m = store.model;
    modelChip.replaceChildren(h(`span.dot.dot-${m ? "ok" : "off"}`), h("span", null, m ? shortName(m.model) : "No model loaded"),
      m ? h("span.chip-sub", null, `${m.device} · ${m.dtype}`) : h("span.chip-sub", null, "Open Model"));
    const busy = store.job === "generate";
    sendBtn.hidden = busy;
    stopBtn.hidden = !busy;
    sendBtn.disabled = !!store.job;
    clearBtn.toggleAttribute("disabled", busy);
    for (const b of modeSeg.children) (b as HTMLButtonElement).disabled = busy;
  };
  store.subscribe(render);
  render();
  setMode(store.settings.generation.mode || "chat");
  if (window.innerWidth < 1100) root.classList.add("params-hidden");

  return { el: root, shown: () => composer.focus() };
}
