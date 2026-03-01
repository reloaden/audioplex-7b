#!/usr/bin/env python3
"""
PersonaPlex-7B  ·  RunPod Serverless Handler

Architecture
────────────
  RunPod handler (this file)
      │  receives job ──► starts moshi subprocess on 127.0.0.1:8998
      │                 ──► starts a WebSocket *proxy* on 0.0.0.0:8888
      │                 ──► reports {public_ip, tcp_port} via progress_update
      │
      ▼
  WebSocket proxy (aiohttp, port 8888 — exposed via RunPod TCP)
      │  external client connects  ──► proxy opens WS to moshi /api/chat
      │  binary frames forwarded bidirectionally
      │  text frames (kind 0x02) accumulated & checked against regex
      │  on match  ──► proxy shuts down, moshi killed, handler returns
      ▼
  Result returned to RunPod caller

RunPod setup
────────────
  • Docker Configuration  → Expose TCP Ports → 8888
  • The env var RUNPOD_TCP_PORT_8888 will contain the mapped port
  • RUNPOD_PUBLIC_IP has the worker's public address
  • Attach a Network Volume for HF model caching (mounted at /runpod-volume)

Input schema  (job["input"])
─────────────
  system_prompt   str   Text/role prompt.        Default: friendly-teacher prompt
  voice_prompt    str   Voice preset filename.   Default: "NATF2.pt"
  regex_pattern   str   Regex to watch for in model text output.   Optional
  seed            int   RNG seed.                Optional (-1 = random)
  timeout         int   Session timeout (sec).   Default: 300
  startup_timeout int   Model-load timeout (sec).Default: 300
  cpu_offload     bool  Offload LM layers to RAM.Default: false
  device          str   Torch device.            Default: "cuda"
  hf_repo         str   HuggingFace model repo.  Default: nvidia/personaplex-7b-v1

Output schema
─────────────
  success         bool
  matched_text    str   (on success)  accumulated text up to & including match
  match_time      float (on success)  seconds from session start to match
  connection_info dict  {public_ip, tcp_port, websocket_url}
  collected_text  str   (on failure)  all text received before timeout/disconnect
  reason / error  str   (on failure)
"""

import asyncio
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Dict, Optional
from urllib.parse import urlencode

import aiohttp
from aiohttp import web
import runpod

# ── Constants ─────────────────────────────────────────────────────────

MOSHI_INTERNAL_PORT = 8998
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8888"))

DEFAULT_REPO = "nvidia/personaplex-7b-v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a wise and friendly teacher. "
    "Answer questions or provide advice in a clear and engaging way."
)

HF_CACHE_SEARCH_BASES = ["/runpod-volume", "/workspace", os.path.expanduser("~/.cache")]
HF_DIR_NAMES = ["huggingface", "huggingface-cache", ".cache/huggingface"]

MODEL_DIR_CANDIDATES = [
    "/runpod-volume/models",
    "/runpod-volume/personaplex",
    "/workspace/models",
    "/models",
]

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("personaplex")

# ── Model / cache discovery ──────────────────────────────────────────


def discover_hf_cache(hf_repo: str = DEFAULT_REPO) -> Optional[str]:
    """
    Return an HF_HOME path.  Priority:
      1. HF_HOME env var (if the dir exists)
      2. A directory that already contains the cached model
      3. A writable persistent directory (so future runs benefit)
    """
    env_home = os.environ.get("HF_HOME")
    if env_home and os.path.isdir(env_home):
        logger.info(f"HF_HOME from env: {env_home}")
        return env_home

    model_subpath = os.path.join("hub", f"models--{hf_repo.replace('/', '--')}")

    for base in HF_CACHE_SEARCH_BASES:
        if not os.path.isdir(base):
            continue
        for hf_dir in HF_DIR_NAMES:
            candidate = os.path.join(base, hf_dir)
            if os.path.isdir(os.path.join(candidate, model_subpath)):
                logger.info(f"Cached model found under {candidate}")
                return candidate

    for base in HF_CACHE_SEARCH_BASES:
        if os.path.isdir(base) and os.access(base, os.W_OK):
            fallback = os.path.join(base, "huggingface")
            os.makedirs(fallback, exist_ok=True)
            logger.info(f"Using {fallback} as HF_HOME (model will be downloaded)")
            return fallback

    return None


def is_model_cached(hf_home: str, hf_repo: str = DEFAULT_REPO) -> bool:
    """Check whether the model snapshots directory is populated."""
    snaps = os.path.join(
        hf_home, "hub", f"models--{hf_repo.replace('/', '--')}", "snapshots"
    )
    return os.path.isdir(snaps) and bool(os.listdir(snaps))


def discover_model_files() -> Dict[str, str]:
    """Look for explicit weight / tokenizer / voice-prompt paths via env or known dirs."""
    found: Dict[str, str] = {}
    env_map = {
        "MOSHI_WEIGHT": "moshi_weight",
        "MIMI_WEIGHT": "mimi_weight",
        "TOKENIZER_PATH": "tokenizer",
        "VOICE_PROMPT_DIR": "voice_prompt_dir",
    }
    for env_key, field in env_map.items():
        val = os.environ.get(env_key)
        if val and os.path.exists(val):
            found[field] = val

    if found:
        logger.info(f"Model files from env: {list(found.keys())}")
        return found

    for base in MODEL_DIR_CANDIDATES:
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            low = entry.lower()
            if "moshi" in low and low.endswith((".safetensors", ".bin", ".pt")):
                found["moshi_weight"] = full
            elif "mimi" in low and low.endswith((".safetensors", ".bin", ".pt")):
                found["mimi_weight"] = full
            elif low.endswith(".model") and "token" in low:
                found["tokenizer"] = full
            elif entry == "voices" and os.path.isdir(full):
                found["voice_prompt_dir"] = full
        if found:
            logger.info(f"Discovered model files in {base}: {list(found.keys())}")
            break

    return found


# ── Subprocess log pipe ──────────────────────────────────────────────


def _pipe_output(proc: subprocess.Popen):
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            logger.info("[moshi] %s", line.rstrip())
    except (ValueError, OSError):
        pass


# ── Session ──────────────────────────────────────────────────────────


class PersonaplexSession:
    """One RunPod job  ↔  one moshi server  +  WebSocket proxy."""

    def __init__(self, cfg: dict):
        self.system_prompt: str = (
            cfg.get("system_prompt")
            or cfg.get("text_prompt")
            or DEFAULT_SYSTEM_PROMPT
        )
        self.voice_prompt: str = cfg.get("voice_prompt", "NATF2.pt")
        self.seed: Optional[int] = cfg.get("seed")
        self.timeout: int = cfg.get("timeout", 300)
        self.startup_timeout: int = cfg.get("startup_timeout", 300)
        self.cpu_offload: bool = cfg.get("cpu_offload", False)
        self.device: str = cfg.get("device", "cuda")
        self.hf_repo: str = cfg.get("hf_repo", DEFAULT_REPO)

        regex_src = cfg.get("regex_pattern") or cfg.get("string")
        self.regex: Optional[re.Pattern] = re.compile(regex_src) if regex_src else None

        self.moshi_proc: Optional[subprocess.Popen] = None
        self.collected_text: str = ""
        self.match_found: bool = False
        self.match_result: Optional[dict] = None
        self.client_connected: bool = False
        self.start_time: float = time.time()
        self.shutdown_event: asyncio.Event = asyncio.Event()

    # ── moshi server lifecycle ────────────────────────────────────

    def start_moshi_server(self):
        env = os.environ.copy()

        hf_cache = discover_hf_cache(self.hf_repo)
        if hf_cache:
            env["HF_HOME"] = hf_cache
            env["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")
            if is_model_cached(hf_cache, self.hf_repo):
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
                logger.info("HF offline mode ON (model already cached)")

        model_files = discover_model_files()

        cmd = [
            sys.executable, "-m", "moshi.server",
            "--host", "127.0.0.1",
            "--port", str(MOSHI_INTERNAL_PORT),
            "--static", "none",
            "--device", self.device,
        ]
        if self.hf_repo != DEFAULT_REPO:
            cmd.extend(["--hf-repo", self.hf_repo])
        if self.cpu_offload:
            cmd.append("--cpu-offload")
        for flag, key in [
            ("--moshi-weight", "moshi_weight"),
            ("--mimi-weight", "mimi_weight"),
            ("--tokenizer", "tokenizer"),
            ("--voice-prompt-dir", "voice_prompt_dir"),
        ]:
            if key in model_files:
                cmd.extend([flag, model_files[key]])

        logger.info("Launching: %s", " ".join(cmd))
        self.moshi_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=_pipe_output, args=(self.moshi_proc,), daemon=True).start()
        self._wait_for_ready()
        logger.info("Moshi server ready on :%d", MOSHI_INTERNAL_PORT)

    def _wait_for_ready(self):
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.moshi_proc and self.moshi_proc.poll() is not None:
                raise RuntimeError(
                    f"Moshi server exited during startup (code {self.moshi_proc.returncode})"
                )
            try:
                s = socket.create_connection(("127.0.0.1", MOSHI_INTERNAL_PORT), timeout=2)
                s.close()
                return
            except (ConnectionRefusedError, OSError, socket.timeout):
                time.sleep(3)
        raise TimeoutError(f"Moshi server not ready after {self.startup_timeout}s")

    def stop_moshi_server(self):
        if self.moshi_proc and self.moshi_proc.poll() is None:
            logger.info("Terminating moshi server …")
            self.moshi_proc.terminate()
            try:
                self.moshi_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.moshi_proc.kill()
                self.moshi_proc.wait()
            logger.info("Moshi server stopped")
        self.moshi_proc = None

    # ── WebSocket proxy ───────────────────────────────────────────

    async def _proxy_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws_ext = web.WebSocketResponse()
        await ws_ext.prepare(request)
        self.client_connected = True
        logger.info("Client connected from %s", request.remote)

        params: Dict[str, str] = {
            "voice_prompt": request.query.get("voice_prompt", self.voice_prompt),
            "text_prompt": request.query.get("text_prompt", self.system_prompt),
        }
        seed = request.query.get("seed") or (str(self.seed) if self.seed is not None else None)
        if seed is not None:
            params["seed"] = seed

        moshi_url = f"ws://127.0.0.1:{MOSHI_INTERNAL_PORT}/api/chat?{urlencode(params)}"
        logger.info("Connecting proxy → moshi at %s", moshi_url)

        session = aiohttp.ClientSession()
        try:
            ws_moshi = await session.ws_connect(moshi_url)
            logger.info("Proxy ↔ moshi WebSocket established")

            async def ext_to_moshi():
                try:
                    async for msg in ws_ext:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            await ws_moshi.send_bytes(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                except Exception as exc:
                    logger.debug("ext→moshi ended: %s", exc)
                finally:
                    self.shutdown_event.set()

            async def moshi_to_ext():
                try:
                    async for msg in ws_moshi:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            data: bytes = msg.data
                            # Text token: kind byte 0x02 + UTF-8 payload
                            if len(data) > 1 and data[0] == 0x02:
                                chunk = data[1:].decode("utf-8", errors="replace")
                                self.collected_text += chunk
                                if self.regex and self.regex.search(self.collected_text):
                                    logger.info(
                                        "Regex matched after %.1fs",
                                        time.time() - self.start_time,
                                    )
                                    self.match_found = True
                                    self.match_result = {
                                        "matched_text": self.collected_text,
                                        "match_time": time.time() - self.start_time,
                                    }
                                    try:
                                        await ws_ext.send_bytes(data)
                                    except Exception:
                                        pass
                                    self.shutdown_event.set()
                                    return
                            try:
                                await ws_ext.send_bytes(data)
                            except Exception:
                                break
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                except Exception as exc:
                    logger.debug("moshi→ext ended: %s", exc)
                finally:
                    self.shutdown_event.set()

            async def shutdown_watcher():
                await self.shutdown_event.wait()
                await asyncio.sleep(0.2)
                for ws in (ws_moshi, ws_ext):
                    if not ws.closed:
                        try:
                            await ws.close()
                        except Exception:
                            pass

            tasks = [
                asyncio.create_task(ext_to_moshi()),
                asyncio.create_task(moshi_to_ext()),
                asyncio.create_task(shutdown_watcher()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        except Exception as exc:
            logger.error("Proxy error: %s", exc, exc_info=True)
            self.shutdown_event.set()
        finally:
            await session.close()

        return ws_ext

    async def _health(self, _request: web.Request) -> web.Response:
        alive = self.moshi_proc is not None and self.moshi_proc.poll() is None
        return web.json_response({
            "moshi_alive": alive,
            "client_connected": self.client_connected,
            "collected_text_len": len(self.collected_text),
            "match_found": self.match_found,
        })

    # ── Main run loop ─────────────────────────────────────────────

    async def run(self):
        app = web.Application()
        app.router.add_get("/ws", self._proxy_ws)
        app.router.add_get("/api/chat", self._proxy_ws)
        app.router.add_get("/health", self._health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", PROXY_PORT)
        await site.start()
        logger.info("WebSocket proxy listening on 0.0.0.0:%d", PROXY_PORT)

        async def moshi_health_monitor():
            while not self.shutdown_event.is_set():
                if self.moshi_proc and self.moshi_proc.poll() is not None:
                    logger.error(
                        "Moshi server died (code %s)", self.moshi_proc.returncode
                    )
                    self.shutdown_event.set()
                    return
                await asyncio.sleep(5)

        monitor = asyncio.create_task(moshi_health_monitor())
        try:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.info("Session timed out after %ds", self.timeout)

        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass

        await runner.cleanup()


# ── RunPod handler ────────────────────────────────────────────────────


def handler(job):
    input_data = job.get("input", {})
    logger.info("Job received — keys: %s", list(input_data.keys()))

    session = PersonaplexSession(input_data)

    try:
        session.start_moshi_server()

        public_ip = os.environ.get("RUNPOD_PUBLIC_IP", "localhost")
        tcp_port = int(os.environ.get(f"RUNPOD_TCP_PORT_{PROXY_PORT}", str(PROXY_PORT)))

        conn_info = {
            "status": "ready",
            "public_ip": public_ip,
            "tcp_port": tcp_port,
            "websocket_url": f"ws://{public_ip}:{tcp_port}/ws",
        }
        logger.info("Connection info: %s", conn_info)

        try:
            runpod.serverless.progress_update(job, conn_info)
        except Exception:
            logger.debug("progress_update unavailable (running locally?)")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(session.run())
        finally:
            loop.close()

        if session.match_found:
            return {
                "success": True,
                "matched_text": session.match_result["matched_text"],
                "match_time": session.match_result["match_time"],
                "connection_info": conn_info,
            }

        return {
            "success": False,
            "reason": (
                "timeout"
                if not session.client_connected
                else "client_disconnected"
            ),
            "collected_text": session.collected_text,
            "connection_info": conn_info,
        }

    except Exception as exc:
        logger.error("Handler error: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}

    finally:
        session.stop_moshi_server()


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
