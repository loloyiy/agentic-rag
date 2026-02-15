"""
llama.cpp Server Manager API
Manage llama-server lifecycle: start, stop, status, logs, model downloads.
"""

import asyncio
import logging
import os
import signal
import shutil
import time
import threading
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Deque

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from core.store import settings_store

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLAMA_SERVER_BINARY = shutil.which("llama-server") or "/opt/homebrew/bin/llama-server"
MODELS_DIR = os.path.expanduser("~/.cache/llama.cpp")

# Additional directories to scan for GGUF models
MODELS_DIRS = [
    MODELS_DIR,
    os.path.expanduser("~/Models"),
    os.path.expanduser("~/.cache/lm-studio/models"),
]

# Ensure primary models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ServerStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class LlamaServerConfig(BaseModel):
    """Configuration for launching llama-server."""
    # Basic
    model_path: str = Field(..., description="Path to GGUF model file")
    port: int = Field(default=8080, ge=1024, le=65535)
    host: str = Field(default="127.0.0.1")
    alias: Optional[str] = Field(default=None, description="Model alias for API")

    # Performance
    ctx_size: int = Field(default=0, ge=0, description="Context size (0 = model default)")
    n_gpu_layers: str = Field(default="auto", description="GPU layers: auto, all, or number")
    threads: int = Field(default=1, ge=1, le=256)
    batch_size: int = Field(default=2048, ge=1)
    ubatch_size: int = Field(default=512, ge=1)
    flash_attn: str = Field(default="auto", description="Flash attention: on, off, auto")
    parallel: int = Field(default=1, ge=1, le=64, description="Number of parallel slots")

    # Features
    embedding: bool = Field(default=False, description="Enable embedding endpoint")
    cont_batching: bool = Field(default=True, description="Enable continuous batching")
    cache_prompt: bool = Field(default=True, description="Enable prompt caching")
    mlock: bool = Field(default=False, description="Lock model in RAM")
    metrics: bool = Field(default=False, description="Enable Prometheus metrics")
    jinja: bool = Field(default=True, description="Enable Jinja template engine")

    # KV Cache
    cache_type_k: str = Field(default="f16", description="KV cache type for K")
    cache_type_v: str = Field(default="f16", description="KV cache type for V")

    # Advanced
    chat_template: Optional[str] = Field(default=None, description="Override chat template")
    reasoning_format: str = Field(default="none", description="Reasoning format: none, deepseek, auto")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    log_file: Optional[str] = Field(default=None, description="Log file path")


class ServerStatusResponse(BaseModel):
    status: ServerStatus
    pid: Optional[int] = None
    uptime_seconds: Optional[float] = None
    uptime_pretty: Optional[str] = None
    loaded_model: Optional[str] = None
    port: Optional[int] = None
    host: Optional[str] = None
    config: Optional[LlamaServerConfig] = None
    error_message: Optional[str] = None
    binary_path: Optional[str] = None
    binary_found: bool = True


class GGUFModelInfo(BaseModel):
    filename: str
    filepath: str
    size_bytes: int
    size_pretty: str
    modified_at: str


class ModelDownloadRequest(BaseModel):
    repo_id: str = Field(..., description="HuggingFace repo, e.g. bartowski/Meta-Llama-3.1-70B-Instruct-GGUF")
    filename: str = Field(..., description="GGUF filename, e.g. Meta-Llama-3.1-70B-Instruct-Q5_K_M.gguf")


class DownloadProgress(BaseModel):
    status: str = "idle"  # idle, downloading, completed, failed
    filename: Optional[str] = None
    total_bytes: Optional[int] = None
    downloaded_bytes: int = 0
    percentage: Optional[float] = None
    speed_mbps: Optional[float] = None
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None


class ServerPreset(BaseModel):
    name: str
    description: str
    config: LlamaServerConfig
    builtin: bool = False


class SavePresetRequest(BaseModel):
    name: str
    description: str = ""
    config: LlamaServerConfig


class RecommendedModel(BaseModel):
    name: str
    description: str
    repo_id: str
    filename: str
    size_gb: float
    category: str  # llm or embedding


# ---------------------------------------------------------------------------
# Built-in Presets
# ---------------------------------------------------------------------------

# We need a placeholder model_path for presets; the UI will replace it
_PLACEHOLDER_MODEL = ""

BUILTIN_PRESETS: List[ServerPreset] = [
    ServerPreset(
        name="Default",
        description="Sensible defaults for most use cases",
        config=LlamaServerConfig(
            model_path=_PLACEHOLDER_MODEL,
            port=8080,
            host="127.0.0.1",
            ctx_size=0,
            n_gpu_layers="auto",
            threads=1,
            batch_size=2048,
            ubatch_size=512,
            flash_attn="auto",
        ),
        builtin=True,
    ),
    ServerPreset(
        name="M1 Ultra Optimized",
        description="Optimized for Apple M1 Ultra 64GB — safe for 70B models, quantized KV cache to save memory",
        config=LlamaServerConfig(
            model_path=_PLACEHOLDER_MODEL,
            port=8080,
            host="127.0.0.1",
            ctx_size=8192,
            n_gpu_layers="all",
            threads=1,
            batch_size=2048,
            ubatch_size=2048,
            flash_attn="on",
            mlock=False,
            cache_prompt=True,
            jinja=True,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
        builtin=True,
    ),
    ServerPreset(
        name="Embedding Mode",
        description="Optimized for embedding generation — embedding enabled, multiple parallel slots",
        config=LlamaServerConfig(
            model_path=_PLACEHOLDER_MODEL,
            port=8081,
            host="127.0.0.1",
            embedding=True,
            n_gpu_layers="all",
            threads=1,
            batch_size=2048,
            ubatch_size=2048,
            flash_attn="on",
            cont_batching=True,
            parallel=4,
        ),
        builtin=True,
    ),
    ServerPreset(
        name="Low Memory",
        description="Conservative settings — smaller context, quantized KV cache, reduced batches",
        config=LlamaServerConfig(
            model_path=_PLACEHOLDER_MODEL,
            port=8080,
            host="127.0.0.1",
            ctx_size=2048,
            n_gpu_layers="auto",
            threads=4,
            batch_size=512,
            ubatch_size=256,
            flash_attn="on",
            cache_type_k="q4_0",
            cache_type_v="q4_0",
        ),
        builtin=True,
    ),
]

RECOMMENDED_MODELS: List[RecommendedModel] = [
    RecommendedModel(
        name="Qwen 2.5 32B Instruct (Q4_K_M) ★ Recommended",
        description="Best balance of quality and memory — fits comfortably on 64GB with room for RAG services",
        repo_id="bartowski/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        size_gb=20.0,
        category="llm",
    ),
    RecommendedModel(
        name="Qwen 2.5 32B Instruct (Q6_K)",
        description="Higher quality 32B — uses ~24GB, best quality-per-GB for 64GB systems",
        repo_id="bartowski/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q6_K.gguf",
        size_gb=24.0,
        category="llm",
    ),
    RecommendedModel(
        name="Llama 3.1 70B Instruct (Q4_K_M)",
        description="Best 70B model — WARNING: requires ~50GB RAM, may cause instability on 64GB systems with other services",
        repo_id="bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        filename="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        size_gb=40.5,
        category="llm",
    ),
    RecommendedModel(
        name="Qwen 2.5 72B Instruct (Q4_K_M)",
        description="High-quality multilingual 72B — WARNING: requires ~48GB RAM, may cause instability on 64GB systems",
        repo_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
        filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        size_gb=42.5,
        category="llm",
    ),
    RecommendedModel(
        name="Llama 3.2 3B Instruct (Q4_K_M)",
        description="Small and fast model for quick inference and testing",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_gb=1.9,
        category="llm",
    ),
    RecommendedModel(
        name="Nomic Embed Text v1.5 (Q8_0)",
        description="High-quality text embedding model — 768 dimensions, great for RAG",
        repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
        filename="nomic-embed-text-v1.5.Q8_0.gguf",
        size_gb=0.14,
        category="embedding",
    ),
    RecommendedModel(
        name="All-MiniLM-L6-v2 (Q8_0)",
        description="Lightweight embedding model — 384 dimensions, very fast",
        repo_id="leliuga/all-MiniLM-L6-v2-GGUF",
        filename="all-MiniLM-L6-v2.Q8_0.gguf",
        size_gb=0.024,
        category="embedding",
    ),
]


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _format_uptime(seconds: float) -> str:
    """Format seconds to human-readable uptime."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def _validate_model_path(model_path: str) -> str:
    """Validate and resolve model path. Returns resolved path or raises."""
    resolved = os.path.realpath(os.path.expanduser(model_path))

    # Check file exists
    if not os.path.isfile(resolved):
        raise ValueError(f"Model file not found: {resolved}")

    # Check it's a .gguf file
    if not resolved.lower().endswith(".gguf"):
        raise ValueError(f"Not a GGUF file: {resolved}")

    # Security: must be within the user's home directory
    home = os.path.expanduser("~")
    if not resolved.startswith(home):
        raise ValueError(f"Model path must be within home directory: {resolved}")

    return resolved


# ---------------------------------------------------------------------------
# LlamaServerManager — Singleton for subprocess lifecycle
# ---------------------------------------------------------------------------

class LlamaServerManager:
    """Manages the llama-server subprocess lifecycle."""

    LOG_BUFFER_SIZE = 2000  # Max log lines to keep

    def __init__(self):
        self._lock = threading.RLock()  # RLock: reentrant to avoid deadlocks when get_status() is called inside locked sections
        self._process: Optional[asyncio.subprocess.Process] = None
        self._status: ServerStatus = ServerStatus.STOPPED
        self._config: Optional[LlamaServerConfig] = None
        self._pid: Optional[int] = None
        self._start_time: Optional[float] = None
        self._error_message: Optional[str] = None
        self._log_buffer: Deque[str] = deque(maxlen=self.LOG_BUFFER_SIZE)
        self._log_task_stdout: Optional[asyncio.Task] = None
        self._log_task_stderr: Optional[asyncio.Task] = None

    # -- Build CLI args from config -------------------------------------------

    def _build_args(self, config: LlamaServerConfig) -> List[str]:
        """Convert LlamaServerConfig to CLI argument list."""
        args = [LLAMA_SERVER_BINARY]
        args.extend(["-m", config.model_path])
        args.extend(["--port", str(config.port)])
        args.extend(["--host", config.host])

        if config.ctx_size > 0:
            args.extend(["-c", str(config.ctx_size)])

        # GPU layers
        if config.n_gpu_layers == "all":
            args.extend(["-ngl", "99999"])
        elif config.n_gpu_layers != "auto":
            try:
                int(config.n_gpu_layers)
                args.extend(["-ngl", config.n_gpu_layers])
            except ValueError:
                pass  # Skip invalid values, use auto

        args.extend(["-t", str(config.threads)])
        args.extend(["-b", str(config.batch_size)])
        args.extend(["-ub", str(config.ubatch_size)])

        if config.flash_attn != "auto":
            args.extend(["-fa", config.flash_attn])

        if config.parallel > 1:
            args.extend(["-np", str(config.parallel)])

        if config.embedding:
            args.append("--embedding")

        if config.cont_batching:
            args.append("--cont-batching")
        else:
            args.append("--no-cont-batching")

        if config.cache_prompt:
            args.append("--cache-prompt")
        else:
            args.append("--no-cache-prompt")

        if config.mlock:
            args.append("--mlock")

        if config.metrics:
            args.append("--metrics")

        if config.jinja:
            args.append("--jinja")
        else:
            args.append("--no-jinja")

        args.extend(["-ctk", config.cache_type_k])
        args.extend(["-ctv", config.cache_type_v])

        if config.chat_template:
            args.extend(["--chat-template", config.chat_template])

        if config.reasoning_format and config.reasoning_format != "none":
            args.extend(["--reasoning-format", config.reasoning_format])

        if config.api_key:
            args.extend(["--api-key", config.api_key])

        if config.alias:
            args.extend(["-a", config.alias])

        if config.log_file:
            args.extend(["--log-file", config.log_file])

        return args

    # -- Log reader -----------------------------------------------------------

    async def _read_stream(self, stream: asyncio.StreamReader, prefix: str = ""):
        """Read lines from an async stream and append to log buffer."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self._log_buffer.append(f"[{timestamp}] {decoded}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._log_buffer.append(f"[LOG ERROR] {prefix}: {e}")

    # -- Start ----------------------------------------------------------------

    async def start(self, config: LlamaServerConfig) -> ServerStatusResponse:
        """Start llama-server with the given configuration."""
        with self._lock:
            if self._status in (ServerStatus.RUNNING, ServerStatus.STARTING):
                raise HTTPException(
                    status_code=409,
                    detail=f"Server is already {self._status.value}. Stop it first."
                )

        # Validate binary
        if not os.path.isfile(LLAMA_SERVER_BINARY):
            raise HTTPException(
                status_code=500,
                detail=f"llama-server binary not found at {LLAMA_SERVER_BINARY}. "
                       f"Install with: brew install llama.cpp"
            )

        # Validate model path
        try:
            config.model_path = _validate_model_path(config.model_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        with self._lock:
            self._status = ServerStatus.STARTING
            self._error_message = None
            self._log_buffer.clear()
            self._config = config

        args = self._build_args(config)

        # Log the command (mask API key)
        safe_args = [a if i == 0 or args[i - 1] != "--api-key" else "***" for i, a in enumerate(args)]
        logger.info(f"Starting llama-server: {' '.join(safe_args)}")
        self._log_buffer.append(f"[SYSTEM] Starting: {' '.join(safe_args)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            with self._lock:
                self._process = process
                self._pid = process.pid
                self._start_time = time.time()

            # Start background log readers
            self._log_task_stdout = asyncio.create_task(
                self._read_stream(process.stdout, "stdout")
            )
            self._log_task_stderr = asyncio.create_task(
                self._read_stream(process.stderr, "stderr")
            )

            # Wait a moment to check for immediate crashes
            await asyncio.sleep(3)

            if process.returncode is not None:
                # Process exited immediately — error
                with self._lock:
                    self._status = ServerStatus.ERROR
                    self._error_message = f"Server exited immediately with code {process.returncode}"
                    self._log_buffer.append(f"[SYSTEM] ERROR: {self._error_message}")
                return self.get_status()

            with self._lock:
                self._status = ServerStatus.RUNNING
                self._log_buffer.append(f"[SYSTEM] Server started on {config.host}:{config.port} (PID {process.pid})")

            # Update the llamacpp_base_url in settings so RAG uses this server
            base_url = f"http://{config.host}:{config.port}"
            settings_store.update({"llamacpp_base_url": base_url})
            logger.info(f"Updated llamacpp_base_url to {base_url}")

            # Monitor process in background
            asyncio.create_task(self._monitor_process(process))

            return self.get_status()

        except FileNotFoundError:
            with self._lock:
                self._status = ServerStatus.ERROR
                self._error_message = f"Binary not found: {LLAMA_SERVER_BINARY}"
            raise HTTPException(status_code=500, detail=self._error_message)
        except Exception as e:
            with self._lock:
                self._status = ServerStatus.ERROR
                self._error_message = str(e)
            raise HTTPException(status_code=500, detail=f"Failed to start server: {e}")

    async def _monitor_process(self, process: asyncio.subprocess.Process):
        """Monitor the process and update status when it exits."""
        try:
            returncode = await process.wait()
            with self._lock:
                if self._status != ServerStatus.STOPPING:
                    self._status = ServerStatus.ERROR if returncode != 0 else ServerStatus.STOPPED
                    self._error_message = f"Server exited with code {returncode}" if returncode != 0 else None
                else:
                    self._status = ServerStatus.STOPPED
                    self._error_message = None
                self._process = None
                self._pid = None
                self._log_buffer.append(f"[SYSTEM] Server stopped (exit code: {returncode})")
        except asyncio.CancelledError:
            pass

    # -- Reset (clear error state) -------------------------------------------

    def reset(self) -> ServerStatusResponse:
        """Reset from ERROR state back to STOPPED. Also works from any non-running state."""
        with self._lock:
            if self._status == ServerStatus.RUNNING:
                raise HTTPException(
                    status_code=409,
                    detail="Server is running. Use stop instead."
                )
            if self._status == ServerStatus.STARTING:
                raise HTTPException(
                    status_code=409,
                    detail="Server is starting. Wait or use stop."
                )
            # Kill any zombie process just in case
            process = self._process
            if process is not None:
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass
            self._status = ServerStatus.STOPPED
            self._process = None
            self._pid = None
            self._error_message = None
            self._log_buffer.append("[SYSTEM] State reset to STOPPED")
        return self.get_status()

    # -- Stop -----------------------------------------------------------------

    async def stop(self) -> ServerStatusResponse:
        """Stop the running llama-server."""
        with self._lock:
            if self._status not in (ServerStatus.RUNNING, ServerStatus.STARTING, ServerStatus.ERROR):
                return self.get_status()

            # If in ERROR state with no process, just reset to STOPPED
            if self._status == ServerStatus.ERROR and self._process is None:
                self._status = ServerStatus.STOPPED
                self._error_message = None
                self._log_buffer.append("[SYSTEM] Cleared error state")
                return self.get_status()

            self._status = ServerStatus.STOPPING
            process = self._process

        if process is None:
            with self._lock:
                self._status = ServerStatus.STOPPED
                self._error_message = None
            return self.get_status()

        self._log_buffer.append("[SYSTEM] Stopping server...")
        logger.info(f"Stopping llama-server (PID {process.pid})")

        try:
            process.terminate()  # SIGTERM
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                self._log_buffer.append("[SYSTEM] Server terminated gracefully")
            except asyncio.TimeoutError:
                self._log_buffer.append("[SYSTEM] SIGTERM timeout, sending SIGKILL...")
                process.kill()  # SIGKILL
                await process.wait()
                self._log_buffer.append("[SYSTEM] Server killed")
        except ProcessLookupError:
            self._log_buffer.append("[SYSTEM] Process already gone")
        except Exception as e:
            self._log_buffer.append(f"[SYSTEM] Error stopping: {e}")
            logger.error(f"Error stopping llama-server: {e}")

        # Cancel log reader tasks
        for task in [self._log_task_stdout, self._log_task_stderr]:
            if task and not task.done():
                task.cancel()

        with self._lock:
            self._status = ServerStatus.STOPPED
            self._process = None
            self._pid = None
            self._error_message = None

        return self.get_status()

    # -- Status ---------------------------------------------------------------

    def get_status(self) -> ServerStatusResponse:
        """Get current server status."""
        with self._lock:
            uptime = None
            uptime_pretty = None
            if self._start_time and self._status == ServerStatus.RUNNING:
                uptime = time.time() - self._start_time
                uptime_pretty = _format_uptime(uptime)

            model_name = None
            if self._config:
                model_name = os.path.basename(self._config.model_path)

            return ServerStatusResponse(
                status=self._status,
                pid=self._pid,
                uptime_seconds=uptime,
                uptime_pretty=uptime_pretty,
                loaded_model=model_name,
                port=self._config.port if self._config else None,
                host=self._config.host if self._config else None,
                config=self._config if self._status != ServerStatus.STOPPED else None,
                error_message=self._error_message,
                binary_path=LLAMA_SERVER_BINARY,
                binary_found=os.path.isfile(LLAMA_SERVER_BINARY),
            )

    # -- Logs -----------------------------------------------------------------

    def get_logs(self, lines: int = 200) -> List[str]:
        """Get the last N log lines."""
        with self._lock:
            buf = list(self._log_buffer)
        return buf[-lines:] if len(buf) > lines else buf


# ---------------------------------------------------------------------------
# ModelDownloadManager — Singleton for HuggingFace downloads
# ---------------------------------------------------------------------------

class ModelDownloadManager:
    """Manages model downloads from HuggingFace."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = "idle"
        self._filename: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._downloaded_bytes: int = 0
        self._start_time: Optional[float] = None
        self._error_message: Optional[str] = None

    async def download(self, repo_id: str, filename: str) -> None:
        """Download a model file from HuggingFace using streaming."""
        with self._lock:
            if self._status == "downloading":
                raise HTTPException(status_code=409, detail="A download is already in progress")
            self._status = "downloading"
            self._filename = filename
            self._total_bytes = None
            self._downloaded_bytes = 0
            self._start_time = time.time()
            self._error_message = None

        dest_path = os.path.join(MODELS_DIR, filename)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

        logger.info(f"Downloading model: {url} -> {dest_path}")

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0, read=None)) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise Exception(f"HTTP {response.status_code}: {response.reason_phrase}")

                    total = response.headers.get("content-length")
                    if total:
                        with self._lock:
                            self._total_bytes = int(total)

                    with open(dest_path + ".part", "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                            f.write(chunk)
                            with self._lock:
                                self._downloaded_bytes += len(chunk)

            # Rename .part to final name
            os.rename(dest_path + ".part", dest_path)

            with self._lock:
                self._status = "completed"
            logger.info(f"Download completed: {filename}")

        except Exception as e:
            # Clean up partial file
            part_file = dest_path + ".part"
            if os.path.exists(part_file):
                try:
                    os.remove(part_file)
                except OSError:
                    pass

            with self._lock:
                self._status = "failed"
                self._error_message = str(e)
            logger.error(f"Download failed: {e}")

    def get_progress(self) -> DownloadProgress:
        """Get current download progress."""
        with self._lock:
            speed = None
            eta = None
            percentage = None

            if self._status == "downloading" and self._start_time:
                elapsed = time.time() - self._start_time
                if elapsed > 0 and self._downloaded_bytes > 0:
                    speed = (self._downloaded_bytes / elapsed) / (1024 * 1024)  # MB/s
                    if self._total_bytes and self._total_bytes > 0:
                        percentage = (self._downloaded_bytes / self._total_bytes) * 100
                        remaining = self._total_bytes - self._downloaded_bytes
                        bytes_per_sec = self._downloaded_bytes / elapsed
                        if bytes_per_sec > 0:
                            eta = remaining / bytes_per_sec

            return DownloadProgress(
                status=self._status,
                filename=self._filename,
                total_bytes=self._total_bytes,
                downloaded_bytes=self._downloaded_bytes,
                percentage=round(percentage, 1) if percentage else None,
                speed_mbps=round(speed, 1) if speed else None,
                eta_seconds=round(eta) if eta else None,
                error_message=self._error_message,
            )

    def reset(self):
        """Reset download state to idle."""
        with self._lock:
            self._status = "idle"
            self._filename = None
            self._total_bytes = None
            self._downloaded_bytes = 0
            self._start_time = None
            self._error_message = None


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

server_manager = LlamaServerManager()
download_manager = ModelDownloadManager()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.post("/start", response_model=ServerStatusResponse)
async def start_server(config: LlamaServerConfig):
    """Start llama-server with the given configuration."""
    return await server_manager.start(config)


@router.post("/stop", response_model=ServerStatusResponse)
async def stop_server():
    """Stop the running llama-server."""
    return await server_manager.stop()


@router.post("/reset", response_model=ServerStatusResponse)
async def reset_server():
    """Reset server from error state back to stopped."""
    return server_manager.reset()


@router.get("/status", response_model=ServerStatusResponse)
async def get_server_status():
    """Get current llama-server status."""
    return server_manager.get_status()


@router.get("/logs")
async def get_server_logs(lines: int = 200):
    """Get recent llama-server log lines."""
    return {"logs": server_manager.get_logs(lines)}


@router.get("/models")
async def list_local_models():
    """List available GGUF models in all known model directories."""
    models: List[GGUFModelInfo] = []
    seen_paths: set = set()
    scanned_dirs: List[str] = []

    for models_dir in MODELS_DIRS:
        if not os.path.isdir(models_dir):
            continue
        scanned_dirs.append(models_dir)
        for entry in os.scandir(models_dir):
            if entry.is_file() and entry.name.lower().endswith(".gguf"):
                real_path = os.path.realpath(entry.path)
                if real_path in seen_paths:
                    continue
                seen_paths.add(real_path)
                stat = entry.stat()
                models.append(GGUFModelInfo(
                    filename=entry.name,
                    filepath=entry.path,
                    size_bytes=stat.st_size,
                    size_pretty=_format_size(stat.st_size),
                    modified_at=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                ))

    # Sort by modification date, most recent first
    models.sort(key=lambda m: m.modified_at, reverse=True)

    return {"models": models, "models_dir": MODELS_DIR, "scanned_dirs": scanned_dirs}


@router.post("/download")
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading a model from HuggingFace."""
    # Check if already downloaded
    dest = os.path.join(MODELS_DIR, request.filename)
    if os.path.exists(dest):
        return {"message": f"Model already exists: {request.filename}", "already_exists": True}

    # Reset if previous download completed/failed
    progress = download_manager.get_progress()
    if progress.status in ("completed", "failed", "idle"):
        download_manager.reset()

    background_tasks.add_task(download_manager.download, request.repo_id, request.filename)
    return {"message": f"Download started: {request.filename}", "already_exists": False}


@router.get("/download/progress", response_model=DownloadProgress)
async def get_download_progress():
    """Get current model download progress."""
    return download_manager.get_progress()


@router.get("/presets")
async def get_presets():
    """Get available configuration presets (built-in + user-saved)."""
    # Load user presets from settings
    user_presets_raw = settings_store.get("llamacpp_presets", [])
    user_presets = []
    if isinstance(user_presets_raw, list):
        for p in user_presets_raw:
            try:
                user_presets.append(ServerPreset(**p))
            except Exception:
                pass

    all_presets = BUILTIN_PRESETS + user_presets

    return {
        "presets": [p.model_dump() for p in all_presets],
        "recommended_models": [m.model_dump() for m in RECOMMENDED_MODELS],
    }


@router.post("/config/save")
async def save_preset(request: SavePresetRequest):
    """Save current configuration as a named preset."""
    # Load existing user presets
    user_presets_raw = settings_store.get("llamacpp_presets", [])
    if not isinstance(user_presets_raw, list):
        user_presets_raw = []

    # Check name doesn't conflict with built-in
    builtin_names = {p.name.lower() for p in BUILTIN_PRESETS}
    if request.name.lower() in builtin_names:
        raise HTTPException(status_code=400, detail=f"Cannot overwrite built-in preset: {request.name}")

    # Remove existing preset with same name (update)
    user_presets_raw = [p for p in user_presets_raw if p.get("name", "").lower() != request.name.lower()]

    # Add new preset
    new_preset = ServerPreset(
        name=request.name,
        description=request.description,
        config=request.config,
        builtin=False,
    )
    user_presets_raw.append(new_preset.model_dump())

    # Save to settings
    settings_store.set("llamacpp_presets", user_presets_raw)

    return {"success": True, "message": f"Preset '{request.name}' saved"}


@router.delete("/config/{name}")
async def delete_preset(name: str):
    """Delete a user-saved preset."""
    # Can't delete built-in
    builtin_names = {p.name.lower() for p in BUILTIN_PRESETS}
    if name.lower() in builtin_names:
        raise HTTPException(status_code=400, detail=f"Cannot delete built-in preset: {name}")

    user_presets_raw = settings_store.get("llamacpp_presets", [])
    if not isinstance(user_presets_raw, list):
        return {"success": False, "message": "No user presets found"}

    original_count = len(user_presets_raw)
    user_presets_raw = [p for p in user_presets_raw if p.get("name", "").lower() != name.lower()]

    if len(user_presets_raw) == original_count:
        raise HTTPException(status_code=404, detail=f"Preset not found: {name}")

    settings_store.set("llamacpp_presets", user_presets_raw)
    return {"success": True, "message": f"Preset '{name}' deleted"}
