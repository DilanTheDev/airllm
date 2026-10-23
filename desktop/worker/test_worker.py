"""End-to-end tests for airllm_worker.py over its real stdin/stdout protocol.

Runs the worker as a subprocess, exactly like the desktop app does, against a tiny random Llama
from the Hugging Face hub (CPU is fine). Run with: python -m pytest desktop/worker/test_worker.py
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TINY = os.environ.get("AIRLLM_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")


class Client:
    def __init__(self, tmp):
        env = dict(os.environ)
        # Use this repo's airllm (the build copies it next to the worker; tests point at the source).
        env["PYTHONPATH"] = os.path.join(REPO, "air_llm") + os.pathsep + env.get("PYTHONPATH", "")
        env["AIRLLM_WORKER_USE_BUNDLED"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        self.tmp = tmp
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "airllm_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=env, bufsize=1)
        self.msgs = queue.Queue()
        self.stderr = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self.stderr.extend(self.proc.stderr), daemon=True).start()
        ready = self.next(timeout=120)
        assert ready["event"] == "ready"
        self.n = 0

    def _read(self):
        for line in self.proc.stdout:
            self.msgs.put(json.loads(line))  # every stdout line must be valid protocol JSON

    def next(self, timeout=300):
        return self.msgs.get(timeout=timeout)

    def send(self, cmd, **args):
        self.n += 1
        rid = f"r{self.n}"
        self.proc.stdin.write(json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n")
        self.proc.stdin.flush()
        return rid

    def call(self, cmd, timeout=600, **args):
        rid = self.send(cmd, **args)
        events = []
        deadline = time.time() + timeout
        while True:
            m = self.next(timeout=max(1, deadline - time.time()))
            if m.get("id") != rid:
                continue
            if m["type"] == "event":
                events.append(m)
                continue
            return m, events

    def close(self):
        try:
            self.call("shutdown", timeout=30)
        finally:
            self.proc.wait(timeout=30)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    c = Client(tmp_path_factory.mktemp("w"))
    yield c
    c.close()


def test_ping_and_env(client):
    m, _ = client.call("ping")
    assert m["type"] == "result" and m["data"]["pong"]
    m, _ = client.call("env")
    data = m["data"]
    assert data["packages"]["torch"]
    assert data["packages"]["airllm"], data.get("airllm_error")
    assert isinstance(data["gpus"], list)


def test_generate_without_model_errors(client):
    m, _ = client.call("generate", prompt="hi")
    assert m["type"] == "error" and "no model" in m["error"]


def test_unknown_command(client):
    m, _ = client.call("rm_rf")
    assert m["type"] == "error"


def test_load_generate_profile_unload(client, tmp_path):
    m, events = client.call("load", model=TINY, device="cpu", dtype="float32", max_seq_len=128,
                            profiling_mode=True, prefetching=True,
                            layer_shards_saving_path=str(tmp_path / "shards"))
    assert m["type"] == "result", m
    info = m["data"]
    assert info["architecture"] == "LlamaForCausalLM"
    assert info["device"] == "cpu" and info["dtype"] == "float32"
    assert any(e["event"] == "stage" for e in events)

    m, events = client.call("generate", prompt="Hello world", max_new_tokens=12, do_sample=False)
    assert m["type"] == "result", m
    r = m["data"]
    assert r["new_tokens"] > 0
    streamed = "".join(e["text"] for e in events if e["event"] == "token")
    assert streamed.strip() == r["text"].strip()
    assert "profile" in r and r["profile"]

    # sampling path + chat messages without a template falls back to plain text
    m, _ = client.call("generate", messages=[{"role": "user", "content": "hi"}],
                       chat_template=True, max_new_tokens=6, do_sample=True, temperature=0.8,
                       top_p=0.9, top_k=20, repetition_penalty=1.1, seed=1)
    assert m["type"] == "result", m

    # busy guard: a second long job while one runs is refused
    rid = client.send("generate", prompt="Long story", max_new_tokens=400, do_sample=False)
    m2, _ = client.call("generate", prompt="x", max_new_tokens=2)
    assert m2["type"] == "error" and "busy" in m2["error"]
    # cancel mid-stream
    time.sleep(0.5)
    c, _ = client.call("cancel")
    assert c["type"] == "result"
    while True:
        m = client.next()
        if m.get("id") == rid and m["type"] != "event":
            break
    assert m["type"] == "result", m
    assert m["data"]["cancelled"] is True
    assert m["data"]["new_tokens"] < 400

    m, _ = client.call("status")
    assert m["data"]["busy"] is False and m["data"]["model"]["model"] == TINY

    m, _ = client.call("unload")
    assert m["type"] == "result"
    m, _ = client.call("status")
    assert m["data"]["model"] is None


def test_dataset_preview(client, tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "a b c"}\n{"prompt": "q", "completion": "a"}\n', encoding="utf-8")
    m, _ = client.call("dataset_preview", path=str(p))
    assert m["type"] == "result", m
    assert m["data"]["count"] == 2 and m["data"]["masked"] == 1


def test_train_requires_cuda(client):
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    m, _ = client.call("train", model=TINY)
    assert m["type"] == "error" and "CUDA" in m["error"]


def test_stdout_stays_clean(client):
    # airllm prints a lot; none of it may leak onto the protocol stream (the reader would raise).
    assert client.proc.poll() is None
    assert any("generic AirLLM" in line or "splitting" in line.lower() or line.strip()
               for line in client.stderr)
