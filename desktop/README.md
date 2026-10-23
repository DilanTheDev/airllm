# AirLLM Desktop

A portable Windows app for AirLLM. It runs and fine-tunes huge models on a small GPU, with every
AirLLM option in a dark, native-feeling UI. There is no installer: `AirLLM.exe` is the whole app.

## Using it

1. Put `AirLLM.exe` in a folder with plenty of free disk space and a short path, for example `C:\AirLLM\`.
2. Run it. The first launch opens **Engine**. Pick a PyTorch build (the app recommends one for
   your GPU) and click **Set up engine**. This downloads a private Python 3.12 and PyTorch into a
   `runtime` folder next to the exe: about 2.5 GB for CUDA builds and 200 MB for CPU only. Nothing
   is installed system-wide. To remove everything, delete the folder.
3. Open **Model**, enter a Hugging Face repo id (or pick one from *Popular models*), and click
   **Load model**. The first load downloads the checkpoint and splits it into per-layer shards.
4. Chat on **Generate**. Ctrl+Enter sends and Esc stops.

Windows 10 or 11 (x64) is required, with the WebView2 runtime, which Windows 11 includes and most
Windows 10 PCs already have. An NVIDIA GPU is optional.

## What's in it

| Page | AirLLM features |
|---|---|
| Generate | Chat (with the tokenizer's chat template) or raw completion. Streaming, stop, tokens/s. Options: `max_new_tokens`, sampling (`temperature`, `top_p`, `top_k`, seed), `repetition_penalty`, `use_cache`, input `max_length`, truncation, padding, system prompt |
| Model | `AutoModel.from_pretrained` with every option: repo id or local folder, `hf_token`, `device`, `dtype`, `max_seq_len`, `compression` (4bit/8bit), `prefetching`, `profiling_mode`, `layer_shards_saving_path`, `delete_original` |
| Train | Streamed LoRA (`AirLLMLoRA`, `AirLLMLoRAQwen4Exp`): dataset preview, seq len, epochs, steps, rank, alpha, dropout, lr, weight decay, CE chunk size, packed experts, checkpointing, resume. Live loss chart and peak VRAM |
| Storage | Downloaded models and their split shards, with size. Open or delete them, change the download folder (`HF_HOME`), scan extra folders |
| Profiling | Per-stage timings from `profiling_mode`, plus speed and first-token latency for every generation |
| Engine | One-click setup or repair (CUDA 12.6 / 12.8 / 11.8 / CPU, optional bitsandbytes), status, package versions, or use your own Python |
| Logs | Everything the engine prints |

Settings save automatically to `settings.json` next to the exe. Your Hugging Face token is stored
there too, in plain text.

## How it works

- The window is a [Wails](https://wails.io) app (Go + WebView2). The UI is plain TypeScript and
  CSS with no framework, so the JavaScript is under 50 KB.
- AirLLM runs in a separate Python process (`worker/airllm_worker.py`) that the app talks to over
  JSON lines on stdin and stdout. The UI never blocks on PyTorch, and a CUDA crash can't take the
  window down. **Force stop** kills that process, which always frees VRAM. A loaded model stays
  resident between messages.
- The exe embeds the worker and this repo's `air_llm/airllm` package, so the app runs the same
  AirLLM code as the checkout it was built from. Under *Engine → Advanced* you can switch to the
  `airllm` installed in the Python environment instead.

## Building

```bash
go install github.com/wailsapp/wails/v2/cmd/wails@v2.12.0
./desktop/build.sh          # -> desktop/dist/AirLLM.exe
```

The build works from Linux, macOS or Windows (Git Bash). Wails cross-compiles the Windows binary
without cgo. You need Go 1.22+ and Node 18+.

## Tests

```bash
# Go: protocol, process control, setup helpers, cache safety, settings
cd desktop && go test ./internal/...

# Go -> real Python worker -> AirLLM on a tiny Llama (needs torch + transformers)
AIRLLM_TEST_PYTHON=/path/to/python go test ./internal/worker -run TestRealWorker

# Python worker end to end: load, stream, cancel, profile, unload, datasets
python -m pytest desktop/worker/test_worker.py

# UI in Chromium against a mock backend (needs Playwright)
cd desktop/frontend && npm install && npm run build:mock
python3 -m http.server 4317 -d dist-mock &
node tests/ui.test.mjs http://127.0.0.1:4317/ tests/screens
```
