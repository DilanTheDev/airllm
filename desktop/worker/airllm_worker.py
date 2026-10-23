"""Long-lived AirLLM worker driven by the desktop app.

Protocol: one JSON object per line.
  stdin  -> {"id": str, "cmd": str, "args": {...}}
  stdout <- {"id": str, "type": "result" | "error" | "event", ...}

stdout is reserved for the protocol. Everything else (airllm prints, tqdm bars, warnings,
C-level writes) is redirected to stderr, which the app shows in its log console.

Only one long job (load / generate / train) runs at a time, on a background thread, so the
main thread stays free to receive ``cancel`` and ``status`` while it runs.
"""

import json
import os
import sys
import threading
import time
import traceback

# Reserve the real stdout for the protocol before anything else can print to it.
# newline="\n": text mode on Windows would otherwise write \r\n line endings.
_proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1, newline="\n")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("AIRLLM_WORKER_USE_BUNDLED", "1") == "1" and os.path.isdir(os.path.join(_HERE, "airllm")):
    sys.path.insert(0, _HERE)

_write_lock = threading.Lock()


def send(obj):
    line = json.dumps(obj, ensure_ascii=False, default=str)
    with _write_lock:
        _proto.write(line + "\n")
        _proto.flush()


def event(req_id, name, **data):
    send({"id": req_id, "type": "event", "event": name, **data})


def log(msg):
    print(msg, file=sys.stderr, flush=True)


DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class Worker:
    def __init__(self):
        self.model = None
        self.model_info = None
        self.trainer = None
        self.cancel_event = threading.Event()
        self.job = None
        self.job_name = None
        self._torch = None

    # ---- helpers ------------------------------------------------------------------------------

    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def busy(self):
        return self.job is not None and self.job.is_alive()

    def run_job(self, req_id, name, fn, args):
        if self.busy():
            raise RuntimeError(f"busy: '{self.job_name}' is still running")
        self.cancel_event.clear()

        def target():
            try:
                result = fn(req_id, args)
                send({"id": req_id, "type": "result", "data": result})
            except Exception as e:  # noqa: BLE001 - every failure goes back to the UI
                traceback.print_exc()
                send({"id": req_id, "type": "error", "error": f"{type(e).__name__}: {e}"})
            finally:
                self.job_name = None

        self.job_name = name
        self.job = threading.Thread(target=target, name=name, daemon=True)
        self.job.start()

    def _free(self):
        self.model = None
        self.model_info = None
        self.trainer = None
        import gc
        gc.collect()
        if self._torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    # ---- commands -----------------------------------------------------------------------------

    def cmd_ping(self, req_id, args):
        return {"pong": True, "pid": os.getpid()}

    def cmd_status(self, req_id, args):
        return {"busy": self.busy(), "job": self.job_name, "model": self.model_info,
                "trainer": self.trainer is not None}

    def cmd_env(self, req_id, args):
        info = {"python": sys.version.split()[0], "executable": sys.executable, "packages": {}}
        for pkg in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
                    "bitsandbytes", "airllm"):
            try:
                mod = __import__(pkg)
                info["packages"][pkg] = getattr(mod, "__version__", "installed")
                if pkg == "airllm":
                    info["airllm_path"] = os.path.dirname(mod.__file__)
            except Exception as e:  # noqa: BLE001
                info["packages"][pkg] = None
                if pkg == "airllm":
                    info["airllm_error"] = f"{type(e).__name__}: {e}"
        try:
            torch = self.torch
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = torch.version.cuda
            gpus = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    gpus.append({"index": i, "name": p.name,
                                 "total_gb": round(p.total_memory / 1024 ** 3, 2)})
            info["gpus"] = gpus
        except Exception as e:  # noqa: BLE001
            info["cuda_available"] = False
            info["gpus"] = []
            info["torch_error"] = str(e)
        return info

    def cmd_load(self, req_id, args):
        torch = self.torch
        from airllm import AutoModel

        self._free()
        repo = (args.get("model") or "").strip()
        if not repo:
            raise ValueError("no model given")

        kwargs = {
            "max_seq_len": int(args.get("max_seq_len") or 512),
            "profiling_mode": bool(args.get("profiling_mode")),
            "prefetching": bool(args.get("prefetching", True)),
            "delete_original": bool(args.get("delete_original")),
        }
        if args.get("compression") in ("4bit", "8bit"):
            kwargs["compression"] = args["compression"]
        if args.get("hf_token"):
            kwargs["hf_token"] = args["hf_token"]
        if args.get("layer_shards_saving_path"):
            kwargs["layer_shards_saving_path"] = args["layer_shards_saving_path"]
        device = args.get("device") or "auto"
        if device != "auto":
            kwargs["device"] = device
        dtype = args.get("dtype") or "auto"
        if dtype in DTYPES:
            kwargs["dtype"] = getattr(torch, DTYPES[dtype])

        event(req_id, "stage", stage="Resolving model")
        arch = ""
        try:
            arch = AutoModel.get_architecture(repo, **({"hf_token": kwargs["hf_token"]} if "hf_token" in kwargs else {}))
        except Exception as e:  # noqa: BLE001 - surfaced by from_pretrained below
            log(f"could not read config up front: {e}")
        event(req_id, "stage", stage="Downloading and splitting layers", architecture=arch)
        t0 = time.time()
        model = AutoModel.from_pretrained(repo, **kwargs)
        tok = model.tokenizer
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        self.model = model
        self.model_info = {
            "model": repo,
            "architecture": arch,
            "class": type(model).__name__,
            "device": str(getattr(model, "running_device", getattr(model, "device", ""))),
            "dtype": str(getattr(model, "running_dtype", "")).replace("torch.", ""),
            "layers": len(getattr(model, "layer_names", []) or []),
            "max_seq_len": kwargs["max_seq_len"],
            "compression": kwargs.get("compression"),
            "prefetching": bool(getattr(model, "prefetching", kwargs["prefetching"])),
            "profiling_mode": kwargs["profiling_mode"],
            "has_chat_template": bool(getattr(tok, "chat_template", None)),
            "load_seconds": round(time.time() - t0, 2),
        }
        return self.model_info

    def cmd_unload(self, req_id, args):
        self._free()
        return {"unloaded": True}

    def _build_inputs(self, args):
        tok = self.model.tokenizer
        max_length = int(args.get("max_length") or self.model_info["max_seq_len"])
        truncation = bool(args.get("truncation", True))
        messages = args.get("messages")
        use_chat = bool(args.get("chat_template", True)) and messages and getattr(tok, "chat_template", None)
        if use_chat:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            add_special = False
            # Keep the newest turns when the conversation outgrows the window.
            tok.truncation_side = "left"
        else:
            if messages:
                text = "\n\n".join(m["content"] for m in messages if m.get("content"))
            else:
                text = args.get("prompt") or ""
            add_special = True
            tok.truncation_side = "left" if messages else "right"
        if not text:
            raise ValueError("empty prompt")
        enc = tok([text], return_tensors="pt", truncation=truncation, max_length=max_length,
                  padding=bool(args.get("padding", False)), add_special_tokens=add_special)
        return enc

    def cmd_generate(self, req_id, args):
        if self.model is None:
            raise RuntimeError("no model loaded")
        torch = self.torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        model = self.model
        tok = model.tokenizer
        enc = self._build_inputs(args)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        cancel = self.cancel_event
        counter = {"n": 0, "first": None}
        t0 = time.time()

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return cancel.is_set()

        class Streamer(TextStreamer):
            def put(self, value):
                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return
                counter["n"] += value.numel()
                if counter["first"] is None:
                    counter["first"] = time.time()
                super().put(value)

            def on_finalized_text(self, text, stream_end=False):
                if text:
                    event(req_id, "token", text=text, tokens=counter["n"])

        do_sample = bool(args.get("do_sample", False))
        gen = {
            "max_new_tokens": int(args.get("max_new_tokens") or 64),
            "use_cache": bool(args.get("use_cache", True)),
            "do_sample": do_sample,
            "repetition_penalty": float(args.get("repetition_penalty") or 1.0),
            "streamer": Streamer(tok, skip_prompt=True, skip_special_tokens=True),
            "stopping_criteria": StoppingCriteriaList([Cancel()]),
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            "return_dict_in_generate": True,
        }
        if do_sample:
            gen["temperature"] = float(args.get("temperature") or 1.0)
            gen["top_p"] = float(args.get("top_p") or 1.0)
            gen["top_k"] = int(args.get("top_k") or 0)
        else:
            # Neutralize sampling defaults a model's generation_config may carry, which
            # would otherwise warn on every greedy call.
            gen.update(temperature=None, top_p=None, top_k=None)
        if args.get("seed") not in (None, ""):
            torch.manual_seed(int(args["seed"]))
        if attention_mask is not None:
            gen["attention_mask"] = attention_mask

        if getattr(model, "profiling_mode", False):
            model.profiler.clear_profiling_time()

        with torch.inference_mode():
            out = model.generate(input_ids, **gen)

        seqs = out.sequences if hasattr(out, "sequences") else out
        new_ids = seqs[0][input_ids.shape[1]:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        elapsed = time.time() - t0
        n_new = int(new_ids.numel())
        decode_time = (time.time() - counter["first"]) if counter["first"] else elapsed
        result = {
            "text": text,
            "prompt_tokens": int(input_ids.shape[1]),
            "new_tokens": n_new,
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(n_new / decode_time, 3) if decode_time > 0 and n_new else 0.0,
            "first_token_seconds": round(counter["first"] - t0, 3) if counter["first"] else None,
            "cancelled": cancel.is_set(),
        }
        if getattr(model, "profiling_mode", False):
            result["profile"] = {k: {"total": round(sum(v), 4), "count": len(v)}
                                 for k, v in model.profiler.profiling_time_dict.items()}
        return result

    def cmd_dataset_preview(self, req_id, args):
        from airllm.lora_data import load_records
        records = load_records(args["path"])
        sample = [{"text": r["text"][:400], "masked": bool(r.get("mask_prefix"))} for r in records[:5]]
        return {"count": len(records), "sample": sample,
                "masked": sum(1 for r in records if r.get("mask_prefix"))}

    def cmd_train(self, req_id, args):
        torch = self.torch
        from airllm.lora_data import encode_record, iter_train_items, load_records

        if not torch.cuda.is_available() and not args.get("allow_cpu"):
            raise RuntimeError("LoRA training needs an NVIDIA GPU (CUDA). Set up the CUDA engine.")

        kind = args.get("trainer") or "AirLLMLoRA"
        import airllm.airllm_lora as lora_mod
        cls = getattr(lora_mod, kind, None)
        if cls is None:
            raise ValueError(f"unknown trainer {kind}")

        if args.get("data"):
            records = load_records(args["data"])
            epochs, steps = int(args.get("epochs") or 1), (int(args["steps"]) if args.get("steps") else None)
        else:
            records = [{"text": "AirLLM trains LoRA with one decoder layer on the GPU at a time. " * 256,
                        "mask_prefix": None}]
            epochs, steps = int(args.get("steps") or 1), None
        total = len(records) * epochs if steps is None else min(steps, len(records) * epochs)

        seq_len = int(args.get("seq_len") or 512)
        self._free()
        event(req_id, "stage", stage="Loading base model")
        kwargs = dict(
            device=args.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"),
            max_seq_len=int(args.get("max_seq_len") or seq_len),
            lora_r=int(args.get("lora_r") or 16),
            lora_alpha=int(args.get("lora_alpha") or 32),
            lora_dropout=float(args.get("lora_dropout") or 0.0),
            lr=float(args.get("lr") or 1e-4),
            weight_decay=float(args.get("weight_decay") or 0.0),
            ce_chunk_size=int(args.get("ce_chunk_size") or 4096),
            prefetching=bool(args.get("prefetching", True)),
            delete_original=bool(args.get("delete_original")),
            profiling_mode=bool(args.get("profiling_mode")),
        )
        if args.get("compression") in ("4bit", "8bit"):
            kwargs["compression"] = args["compression"]
        if args.get("hf_token"):
            kwargs["hf_token"] = args["hf_token"]
        if args.get("layer_shards_saving_path"):
            kwargs["layer_shards_saving_path"] = args["layer_shards_saving_path"]
        if kind == "AirLLMLoRAQwen4Exp":
            kwargs["packed_experts"] = bool(args.get("packed_experts"))
        trainer = cls(args.get("model") or "Qwen/Qwen3.8-27B", **kwargs)
        self.trainer = trainer
        if args.get("resume_adapter"):
            trainer.load_adapter(args["resume_adapter"])
            log(f"loaded adapter {args['resume_adapter']}")

        tok = trainer.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats(trainer.device)

        event(req_id, "stage", stage="Training", total=total)
        losses, n, t0 = [], 0, time.time()
        save_every = int(args.get("save_every") or 0)
        save_path = args.get("save_adapter") or ""
        for step, epoch, rec in iter_train_items(records, epochs, steps):
            if self.cancel_event.is_set():
                break
            input_ids, attention_mask, labels = encode_record(tok, rec, seq_len, trainer.device)
            loss = trainer.train_step(input_ids, labels=labels, attention_mask=attention_mask,
                                      verbose=bool(args.get("verbose")))
            peak = torch.cuda.max_memory_allocated(trainer.device) / 1024 ** 3 if cuda else 0.0
            losses.append(loss)
            n += 1
            event(req_id, "train_step", step=step, epoch=epoch, loss=loss, peak_vram_gb=round(peak, 3),
                  seq=int(input_ids.shape[1]), total=total, elapsed=round(time.time() - t0, 2))
            if save_every and save_path and n % save_every == 0:
                trainer.save_adapter(save_path)
                log(f"checkpoint: saved adapter to {save_path} at step {step}")

        saved = None
        if save_path and n:
            trainer.save_adapter(save_path)
            saved = save_path
        result = {"steps": n, "final_loss": losses[-1] if losses else None,
                  "mean_loss": (sum(losses) / len(losses)) if losses else None,
                  "saved_adapter": saved, "cancelled": self.cancel_event.is_set(),
                  "seconds": round(time.time() - t0, 2)}
        self.trainer = None
        self._free()
        return result

    def cmd_cancel(self, req_id, args):
        self.cancel_event.set()
        return {"cancelling": self.busy()}


SYNC = {"ping", "status", "cancel", "env", "unload", "dataset_preview"}
ASYNC = {"load", "generate", "train"}


def main():
    w = Worker()
    send({"id": None, "type": "event", "event": "ready", "pid": os.getpid()})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            send({"id": None, "type": "error", "error": f"bad json: {e}"})
            continue
        req_id, cmd, args = msg.get("id"), msg.get("cmd"), msg.get("args") or {}
        if cmd == "shutdown":
            send({"id": req_id, "type": "result", "data": {"bye": True}})
            break
        fn = getattr(w, f"cmd_{cmd}", None)
        if fn is None or cmd not in SYNC | ASYNC:
            send({"id": req_id, "type": "error", "error": f"unknown command {cmd}"})
            continue
        try:
            if cmd in ASYNC:
                w.run_job(req_id, cmd, fn, args)
            else:
                if cmd == "unload" and w.busy():
                    raise RuntimeError("stop the running job before unloading")
                send({"id": req_id, "type": "result", "data": fn(req_id, args)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            send({"id": req_id, "type": "error", "error": f"{type(e).__name__}: {e}"})
    w.cancel_event.set()
    os._exit(0)


if __name__ == "__main__":
    main()
