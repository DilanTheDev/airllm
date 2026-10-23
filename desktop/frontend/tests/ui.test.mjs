// UI smoke + interaction tests against the mock build (npm run build:mock).
// Usage: node tests/ui.test.mjs <url-of-dist-mock> <screenshot-dir>
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const base = process.argv[2] || "http://127.0.0.1:4317/";
const shots = process.argv[3] || "screens";
mkdirSync(shots, { recursive: true });

let failures = 0;
const check = (cond, msg) => {
  if (cond) console.log("  ok  " + msg);
  else { failures++; console.log("  FAIL " + msg); }
};

const browser = await chromium.launch();

async function open(query = "", size = { width: 1280, height: 820 }) {
  const page = await browser.newPage({ viewport: size });
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(base + query);
  await page.waitForSelector(".sidebar");
  return { page, errors };
}

const nav = (page, label) => page.locator(".nav-item", { hasText: label }).click();

async function noOverflow(page, label) {
  const r = await page.evaluate(() => {
    const pg = [...document.querySelectorAll(".page")].find((p) => !p.hidden);
    return { doc: document.documentElement.scrollWidth <= window.innerWidth, page: pg ? pg.scrollWidth <= pg.clientWidth + 1 : true };
  });
  check(r.doc && r.page, `${label}: no horizontal overflow`);
}

// ---- 1. every page renders, at two window sizes ----
for (const size of [{ width: 1280, height: 820 }, { width: 960, height: 620 }]) {
  console.log(`pages @ ${size.width}x${size.height}`);
  const { page, errors } = await open("", size);
  for (const label of ["Generate", "Model", "Train", "Storage", "Profiling", "Engine", "Logs"]) {
    await nav(page, label);
    await page.waitForTimeout(150);
    const visible = await page.locator(".page:not([hidden])").count();
    check(visible === 1, `${label}: exactly one page visible`);
    await noOverflow(page, label);
    await page.screenshot({ path: `${shots}/${label.toLowerCase()}-${size.width}.png` });
  }
  check(errors.length === 0, `no console errors (${errors.join(" | ")})`);
  await page.close();
}

// ---- 2. load a model, chat with streaming, stop mid-stream ----
console.log("model + generate flow");
{
  const { page, errors } = await open();
  await nav(page, "Generate");
  await page.fill(".composer-input", "What is the capital of the US?");
  await page.keyboard.press("Control+Enter");
  await page.waitForTimeout(200);
  check(await page.locator(".page:not([hidden]) h1", { hasText: "Model" }).count() === 1, "sending without a model routes to Model");

  await page.selectOption("select.preset", "mistralai/Mistral-7B-Instruct-v0.3");
  check(await page.inputValue("#f-model-model") === "mistralai/Mistral-7B-Instruct-v0.3", "preset fills the model field");
  await page.selectOption("#f-model-compression", "4bit");
  check(await page.locator(".note-warn", { hasText: "bitsandbytes" }).isVisible(), "compression shows the bitsandbytes note");
  await page.locator("label[for='f-model-profilingMode']").click();
  await page.getByRole("button", { name: "Load model" }).click();
  await page.waitForSelector(".progress:not([hidden])");
  check(true, "load shows progress");
  await page.waitForSelector(".info-grid:not([hidden]) .kv");
  const calls = await page.evaluate(() => window.__mockCalls);
  const load = calls.find((c) => c.method === "StartJob" && c.args[0] === "load");
  check(load && load.args[1].compression === "4bit" && load.args[1].profiling_mode === true && load.args[1].model.startsWith("mistralai/"), "load sends every model option");
  check(await page.locator(".sidebar-status .dot-ok").count() === 1, "sidebar shows model ready");

  await nav(page, "Generate");
  await page.fill(".composer-input", "What is the capital of the US?");
  await page.keyboard.press("Control+Enter");
  await page.waitForSelector(".msg-assistant.streaming");
  check(await page.getByRole("button", { name: "Stop" }).isVisible(), "stop visible while generating");
  await page.waitForSelector(".msg-assistant:not(.streaming)", { timeout: 10000 });
  const reply = await page.locator(".msg-assistant .msg-body").last().innerText();
  check(reply.startsWith("Washington") && reply.includes("Virginia."), "full streamed reply rendered");
  check((await page.locator(".msg-assistant .msg-meta").last().innerText()).includes("tok/s"), "reply shows speed");

  // second turn carries history; then stop it
  await page.fill(".composer-input", "Tell me more");
  await page.keyboard.press("Control+Enter");
  await page.waitForTimeout(150);
  await page.getByRole("button", { name: "Stop" }).click();
  await page.waitForSelector(".msg-assistant:not(.streaming) >> nth=1");
  const meta2 = await page.locator(".msg-assistant .msg-meta").last().innerText();
  check(meta2.includes("stopped"), "stop marks the reply as stopped");
  const gen = (await page.evaluate(() => window.__mockCalls)).filter((c) => c.method === "StartJob" && c.args[0] === "generate");
  check(gen.length === 2 && gen[1].args[1].messages.length === 3, "second turn sends the conversation");

  // completion mode
  await page.locator(".segmented button", { hasText: "Completion" }).click();
  await page.fill(".composer-input", "Once upon a time");
  await page.keyboard.press("Control+Enter");
  await page.waitForFunction(() => document.querySelector(".completion-out")?.textContent?.includes("Virginia."), null, { timeout: 10000 });
  check(true, "completion mode streams into the document");
  const g3 = (await page.evaluate(() => window.__mockCalls)).filter((c) => c.method === "StartJob" && c.args[0] === "generate")[2];
  check(g3.args[1].prompt === "Once upon a time" && !g3.args[1].messages, "completion sends a raw prompt");

  // sampling toggles enable their fields
  check(await page.locator(".params .field.disabled").count() >= 3, "sampling fields disabled when greedy");
  await page.locator("label[for='f-generation-doSample']").click();
  check(await page.locator(".params .field.disabled").count() === 0, "sampling fields enabled with sample on");
  await page.screenshot({ path: `${shots}/flow-chat.png` });

  await nav(page, "Profiling");
  await page.waitForTimeout(100);
  check(await page.locator(".bar-row").count() === 2, "profiling shows stage bars");
  check(await page.locator(".page:not([hidden]) tbody tr").count() === 3, "profiling lists each generation");
  await page.screenshot({ path: `${shots}/flow-profile.png` });
  check(errors.length === 0, `no console errors (${errors.join(" | ")})`);
  await page.close();
}

// ---- 3. training ----
console.log("train flow");
{
  const { page, errors } = await open();
  await nav(page, "Train");
  await page.selectOption("#f-train-trainer", "AirLLMLoRAQwen4Exp");
  check(await page.inputValue("#f-train-model") === "Qwen/Qwen3.8-Flash-Next", "trainer switch updates the default model");
  check(!(await page.locator(".field", { hasText: "Packed expert LoRA" }).getAttribute("class")).includes("disabled"), "packed experts enabled for Flash-Next");
  await page.locator(".field", { hasText: "Dataset" }).locator("button[title='Browse']").click();
  await page.getByRole("button", { name: "Preview" }).click();
  await page.waitForSelector(".dataset-stats");
  check((await page.locator(".dataset-stats").innerText()).includes("2 examples"), "dataset preview");
  await page.getByRole("button", { name: "Start training" }).click();
  await page.waitForFunction(() => /\d+ \/ 24/.test(document.querySelector(".stat-v")?.textContent || ""));
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${shots}/flow-train.png` });
  await page.locator(".run-card").getByRole("button", { name: "Stop" }).click();
  await page.waitForFunction(() => /Stopped after|Finished/.test(document.querySelector(".run-stage")?.textContent || ""), null, { timeout: 5000 });
  check(true, "training stops cleanly");
  const tr = (await page.evaluate(() => window.__mockCalls)).find((c) => c.method === "StartJob" && c.args[0] === "train");
  check(tr.args[1].data === "C:\\data\\train.jsonl" && tr.args[1].trainer === "AirLLMLoRAQwen4Exp" && tr.args[1].lora_r === 16, "train sends options");
  check(errors.length === 0, `no console errors (${errors.join(" | ")})`);
  await page.close();
}

// ---- 4. storage delete confirm ----
console.log("storage flow");
{
  const { page, errors } = await open();
  await nav(page, "Storage");
  await page.waitForSelector("tbody tr");
  check(await page.locator("tbody tr").count() === 2, "cache lists models");
  await page.locator("tbody tr").first().locator("button[title='Delete these shards']").first().click();
  await page.waitForSelector(".dialog");
  await page.getByRole("button", { name: "Cancel" }).click();
  let dels = (await page.evaluate(() => window.__mockCalls)).filter((c) => c.method === "DeleteCache");
  check(dels.length === 0, "cancel does not delete");
  await page.locator("tbody tr").first().locator("button[title='Delete these shards']").first().click();
  await page.locator(".dialog").getByRole("button", { name: "Delete" }).click();
  await page.waitForTimeout(100);
  dels = (await page.evaluate(() => window.__mockCalls)).filter((c) => c.method === "DeleteCache");
  check(dels.length === 1 && dels[0].args[0].endsWith("splitted_model"), "confirm deletes the chosen split");
  check(errors.length === 0, `no console errors (${errors.join(" | ")})`);
  await page.close();
}

// ---- 5. first run: engine not installed ----
console.log("engine setup flow");
{
  const { page, errors } = await open("?installed=0");
  await page.waitForTimeout(200);
  check(await page.locator(".page:not([hidden]) h1", { hasText: "Engine" }).count() === 1, "first run opens Engine");
  check(await page.locator(".variant", { hasText: "Recommended" }).locator("input").isChecked(), "recommended build preselected");
  await page.screenshot({ path: `${shots}/flow-engine-before.png` });
  await page.getByRole("button", { name: "Set up engine" }).click();
  await page.waitForFunction(() => document.querySelector(".load-stage")?.textContent?.includes("Engine ready"), null, { timeout: 10000 });
  check(true, "setup completes");
  await page.waitForSelector(".page:not([hidden]) .kv-v >> text=2.8.0+cu126");
  check(true, "environment shows package versions after setup");
  await page.screenshot({ path: `${shots}/flow-engine-after.png` });
  check(errors.length === 0, `no console errors (${errors.join(" | ")})`);
  await page.close();
}

// ---- 6. failure path ----
console.log("load failure");
{
  const { page, errors } = await open();
  await nav(page, "Model");
  await page.fill("#f-model-model", "nobody/fail-model");
  await page.getByRole("button", { name: "Load model" }).click();
  await page.waitForSelector(".load-stage >> text=Load failed");
  check(await page.locator(".toast-error").count() === 1, "load error shows a toast");
  check(await page.getByRole("button", { name: "Load model" }).isEnabled(), "can retry after failure");
  check(errors.length === 0, `no console errors (${errors.join(" | ")})`);
  await page.close();
}

await browser.close();
console.log(failures ? `\n${failures} check(s) failed` : "\nall UI checks passed");
process.exit(failures ? 1 : 0);
