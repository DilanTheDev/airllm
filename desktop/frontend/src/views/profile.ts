import type { View } from "../main";
import { store } from "../store";
import { fmtSecs, h, pageHeader } from "../ui";

const LABELS: Record<string, string> = {
  load_safe_tensor: "Read layer from disk",
  compression_time: "Decompress layer",
  create_layer_from_state_dict: "Build layer",
  load_safe_tensor_cpu_wait: "Wait for prefetch",
};

export function profileView(): View {
  const hint = h("div.note");
  const latest = h("section.card");
  const runsBody = h("tbody");
  const runs = h("section.card", null,
    h("header.card-head", null, h("h3", null, "Recent generations"), h("p", null, "Newest first. Kept for this session.")),
    h("div.card-body", null, h("div.table-wrap", null, h("table.table", null,
      h("thead", null, h("tr", null, h("th", null, "Time"), h("th.num", null, "Prompt"), h("th.num", null, "New tokens"),
        h("th.num", null, "First token"), h("th.num", null, "Total"), h("th.num", null, "Tokens/s"))),
      runsBody))));

  const el = h("div.page-inner", null,
    pageHeader("Profiling", "Where generation time goes. AirLLM is usually bound by reading layers from disk."),
    hint, latest, runs);

  const render = () => {
    const prof = store.model?.profiling_mode;
    hint.replaceChildren(prof
      ? "Profiling is on for the loaded model. Each generation records per-stage timings."
      : "Turn on Profiling on the Model page and reload the model to record per-stage timings. Token speed is always recorded.");
    hint.className = "note" + (prof ? "" : " note-warn");

    const r = store.runs.find((x) => x.profile && Object.keys(x.profile).length);
    latest.hidden = !r;
    if (r && r.profile) {
      const items = Object.entries(r.profile).sort((a, b) => b[1].total - a[1].total);
      const max = Math.max(...items.map(([, v]) => v.total), 1e-9);
      const sum = items.reduce((n, [, v]) => n + v.total, 0);
      latest.replaceChildren(
        h("header.card-head", null, h("h3", null, "Latest profile"), h("p", null, `${r.newTokens} tokens in ${fmtSecs(r.seconds)} · stages below sum to ${fmtSecs(sum)}`)),
        h("div.card-body", null, h("div.bars", null, ...items.map(([k, v]) =>
          h("div.bar-row", null,
            h("div.bar-label", null, LABELS[k] ?? k, h("span.muted.mono", null, " " + k)),
            h("div.bar-track", null, h("div.bar-fill", { style: { width: `${(v.total / max) * 100}%` } })),
            h("div.bar-val.mono", null, `${fmtSecs(v.total)} · ${v.count}× · ${((v.total / v.count) * 1000).toFixed(1)} ms`))))));
    }
    runsBody.replaceChildren(...(store.runs.length ? store.runs.map((x) => h("tr", null,
      h("td", null, new Date(x.at).toLocaleTimeString()),
      h("td.num", null, String(x.promptTokens)),
      h("td.num", null, String(x.newTokens)),
      h("td.num", null, x.firstToken != null ? fmtSecs(x.firstToken) : "–"),
      h("td.num", null, fmtSecs(x.seconds)),
      h("td.num", null, x.tps.toFixed(2)))) : [h("tr", null, h("td.muted", { colSpan: 6 }, "No generations yet."))]));
  };
  store.subscribe(render);
  render();
  return { el };
}
