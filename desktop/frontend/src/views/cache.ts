import { api, errText, type CacheEntry } from "../bridge";
import type { View } from "../main";
import { store } from "../store";
import { badge, button, confirmDialog, field, fmtBytes, h, pageHeader, section, toast } from "../ui";

export function cacheView(): View {
  const summary = h("div.muted");
  const tableBody = h("tbody");
  const table = h("table.table", null,
    h("thead", null, h("tr", null, h("th", null, "Model"), h("th", null, "Layer shards"), h("th.num", null, "On disk"), h("th"))),
    tableBody);
  const emptyRow = h("div.empty", null, h("div.empty-title", null, "Nothing cached yet"), h("div.empty-sub", null, "Models you load are downloaded here and split into per-layer shards."));
  const refreshBtn = button("Refresh", { icon: "refresh", onclick: () => refresh() });

  const extraList = h("div.chips");
  const renderExtra = () => {
    const ex = store.settings.engine.extraCache ?? [];
    extraList.replaceChildren(...ex.map((p) => h("span.chip", null, h("span", null, p),
      button(null, { icon: "x", kind: "ghost", small: true, title: "Remove", onclick: () => {
        store.set("engine.extraCache", ex.filter((x) => x !== p));
        renderExtra();
        refresh();
      } }))));
    if (!ex.length) extraList.append(h("span.muted", null, "None"));
  };
  const addExtra = button("Add folder", { icon: "plus", small: true, onclick: async () => {
    try {
      const p = await api.PickFolder("Folder with split models");
      if (!p) return;
      const ex = store.settings.engine.extraCache ?? [];
      if (!ex.includes(p)) store.set("engine.extraCache", [...ex, p]);
      renderExtra();
      refresh();
    } catch (e) { toast(errText(e), "error"); }
  } });

  const el = h("div.page-inner", null,
    pageHeader("Storage", "Downloaded checkpoints and the per-layer shards AirLLM builds from them.", refreshBtn),
    h("section.card", null, h("div.card-body", null, summary, h("div.table-wrap", null, table, emptyRow))),
    section("Locations", null,
      field({ label: "Model download folder", path: "engine.modelCache", browse: "folder", wide: true,
        placeholder: "Default Hugging Face cache", hint: "Sets HF_HOME for the engine. Restart the engine after changing it.", onchange: () => refresh() }),
      h("div.field.wide", null, h("div.field-label", null, "Extra folders to scan"), h("div.chips-row", null, extraList, addExtra))),
  );

  let scanning = false;
  async function refresh() {
    if (scanning) return;
    scanning = true;
    refreshBtn.disabled = true;
    try {
      await store.flushSave();
      render((await api.ScanCache()) ?? []);
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      scanning = false;
      refreshBtn.disabled = false;
    }
  }

  function render(entries: CacheEntry[]) {
    const totalBytes = entries.reduce((n, e) => n + e.bytes, 0);
    const hub = store.settings.engine.modelCache ? store.settings.engine.modelCache + " (hub)" : store.boot.hubDir;
    summary.replaceChildren(h("span", null, `${entries.length} model${entries.length === 1 ? "" : "s"} · ${fmtBytes(totalBytes)} · `), h("span.mono.selectable", null, hub));
    emptyRow.hidden = entries.length > 0;
    table.hidden = entries.length === 0;
    tableBody.replaceChildren(...entries.map((e) => {
      const splits = e.splits ?? [];
      const splitCell = h("td", null, splits.length ? h("div.split-list", null, ...splits.map((s) =>
        h("div.split", null, badge(s.compression || "full", s.compression ? "info" : "muted"),
          h("span.muted", null, `${s.layers} files · ${fmtBytes(s.bytes)}`),
          button(null, { icon: "folder", kind: "ghost", small: true, title: "Open folder", onclick: () => api.OpenFolder(s.path) }),
          button(null, { icon: "trash", kind: "ghost", small: true, title: "Delete these shards", onclick: () => del(s.path, `the ${s.compression || "full-precision"} shards of ${e.name}`) })))) : h("span.muted", null, "Not split"));
      return h("tr", null,
        h("td", null, h("div.cell-title", null, e.name), h("div.cell-sub.mono.selectable", null, e.path)),
        splitCell,
        h("td.num", null, fmtBytes(e.bytes)),
        h("td.row-actions", null,
          button(null, { icon: "folder", kind: "ghost", small: true, title: "Open folder", onclick: () => api.OpenFolder(e.path) }),
          e.hubRepo ? button(null, { icon: "trash", kind: "ghost", small: true, title: "Delete the whole model", onclick: () => del(e.path, `${e.name} (download and shards)`) }) : null));
    }));
  }

  async function del(path: string, what: string) {
    if (store.job) { toast("Wait for the running job to finish", "info"); return; }
    if (!(await confirmDialog("Delete from disk?", `This permanently deletes ${what}. You can download or split it again later.`, "Delete", true))) return;
    try {
      await api.DeleteCache(path);
      toast("Deleted", "ok");
    } catch (e) {
      toast(errText(e), "error");
    }
    refresh();
  }

  renderExtra();
  return { el, shown: () => refresh() };
}
