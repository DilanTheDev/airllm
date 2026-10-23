// One ring buffer for everything the engine and setup print, shared by the Logs page and
// the inline consoles on other pages.

const MAX = 5000;
const lines: string[] = [];
const subs = new Set<(added: string[], reset: boolean) => void>();

export function addLines(added: string[]) {
  if (!added.length) return;
  lines.push(...added);
  let reset = false;
  if (lines.length > MAX) {
    lines.splice(0, lines.length - MAX);
    reset = true;
  }
  for (const fn of subs) fn(added, reset);
}

export function allLines(): readonly string[] {
  return lines;
}

export function clearLines() {
  lines.length = 0;
  for (const fn of subs) fn([], true);
}

export function onLines(fn: (added: string[], reset: boolean) => void): () => void {
  subs.add(fn);
  return () => subs.delete(fn);
}

/** A console element that follows the buffer, batching DOM writes per animation frame. */
export function consoleView(maxLines = MAX): { el: HTMLElement; dispose(): void } {
  const el = document.createElement("pre");
  el.className = "console";
  el.tabIndex = 0;
  let pending: string[] = [];
  let frame = 0;
  let full = true;

  const render = () => {
    frame = 0;
    const stick = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    if (full) {
      const src = lines.slice(-maxLines);
      el.textContent = src.length ? src.join("\n") + "\n" : "";
      full = false;
    } else if (pending.length) {
      el.append(document.createTextNode(pending.join("\n") + "\n"));
      // Trim from the top once the node count grows; cheap compared to a full re-render.
      if (el.childNodes.length > 400) full = true;
    }
    pending = [];
    if (full) { render(); return; }
    if (stick) el.scrollTop = el.scrollHeight;
  };

  const off = onLines((added, reset) => {
    if (reset) full = true;
    else pending.push(...added);
    if (!frame) frame = requestAnimationFrame(render);
  });
  render();
  requestAnimationFrame(() => (el.scrollTop = el.scrollHeight));
  return { el, dispose: () => { off(); cancelAnimationFrame(frame); } };
}
