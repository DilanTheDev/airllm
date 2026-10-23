import { allLines, clearLines, consoleView } from "../logs";
import type { View } from "../main";
import { button, h, pageHeader, toast } from "../ui";

export function logsView(): View {
  const con = consoleView();
  con.el.classList.add("console-full");
  const copy = button("Copy all", { icon: "copy", onclick: () => {
    const text = allLines().join("\n");
    (window.runtime?.ClipboardSetText(text) ?? Promise.resolve(false)).then(() => toast("Log copied", "ok", 1500));
  } });
  const clr = button("Clear", { icon: "clear", onclick: () => clearLines() });
  const el = h("div.page-inner.page-fill", null,
    pageHeader("Logs", "Everything the engine prints: downloads, layer splitting, warnings and errors.", copy, clr),
    con.el);
  return { el };
}
