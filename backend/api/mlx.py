"""
MLX Server Manager API
Manage mlx_lm.server lifecycle: start, stop, status, logs, model downloads.
Uses Apple's MLX framework for fast LLM inference on Apple Silicon.
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

# mlx_lm is a Python module — check if it's importable
try:
    import importlib.util
    MLX_AVAILABLE = importlib.util.find_spec("mlx_lm") is not None
except Exception:
    MLX_AVAILABLE = False

# HuggingFace cache where MLX models are stored
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

# Default MLX server port (8081 to avoid conflict with llama.cpp on 8080)
DEFAULT_MLX_PORT = 8081
DEFAULT_MLX_BASE_URL = os.getenv("MLX_BASE_URL", f"http://localhost:{DEFAULT_MLX_PORT}")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ServerStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class MLXServerConfig(BaseModel):
    """Configuration for launching mlx_lm.server."""
    # Basic
    model: str = Field(..., description="HuggingFace model repo (e.g. mlx-community/Qwen2.5-32B-Instruct-4bit)")
    port: int = Field(default=DEFAULT_MLX_PORT, ge=1024, le=65535)
    host: str = Field(default="127.0.0.1")

    # Generation
    max_tokens: int = Field(default=8192, ge=128, le=131072, description="Max generation tokens (mlx_lm default is 512)")
    max_kv_size: Optional[int] = Field(default=None, ge=512, description="Max KV cache size (None = unlimited)")


class ServerStatusResponse(BaseModel):
    status: ServerStatus
    pid: Optional[int] = None
    uptime_seconds: Optional[float] = None
    uptime_pretty: Optional[str] = None
    loaded_model: Optional[str] = None
    port: Optional[int] = None
    host: Optional[str] = None
    config: Optional[MLXServerConfig] = None
    error_message: Optional[str] = None
    mlx_available: bool = True


class MLXModelInfo(BaseModel):
    """Represents a locally cached MLX model from HuggingFace."""
    name: str  # e.g. "mlx-community/Qwen2.5-32B-Instruct-4bit"
    path: str  # Full path to model directory
    size_bytes: int
    size_pretty: str
    modified_at: str


class ModelDownloadRequest(BaseModel):
    repo_id: str = Field(..., description="HuggingFace repo, e.g. mlx-community/Qwen2.5-32B-Instruct-4bit")


class DownloadProgress(BaseModel):
    status: str = "idle"  # idle, downloading, completed, failed
    repo_id: Optional[str] = None
    percentage: Optional[float] = None
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    downloaded_pretty: Optional[str] = None
    total_pretty: Optional[str] = None
    current_file: Optional[str] = None
    files_done: int = 0
    files_total: int = 0
    error_message: Optional[str] = None


class ServerPreset(BaseModel):
    name: str
    description: str
    config: MLXServerConfig
    builtin: bool = False


class SavePresetRequest(BaseModel):
    name: str
    description: str = ""
    config: MLXServerConfig


class RecommendedModel(BaseModel):
    name: str
    description: str
    repo_id: str
    size_gb: float
    category: str  # llm or embedding


# ---------------------------------------------------------------------------
# Built-in Presets
# ---------------------------------------------------------------------------

_PLACEHOLDER_MODEL = ""

BUILTIN_PRESETS: List[ServerPreset] = [
    ServerPreset(
        name="Default",
        description="Sensible defaults for Apple Silicon — 8K context, generous token limit",
        config=MLXServerConfig(
            model=_PLACEHOLDER_MODEL,
            port=DEFAULT_MLX_PORT,
            host="127.0.0.1",
            max_tokens=8192,
        ),
        builtin=True,
    ),
    ServerPreset(
        name="High Context",
        description="Large context window for long documents — uses more memory",
        config=MLXServerConfig(
            model=_PLACEHOLDER_MODEL,
            port=DEFAULT_MLX_PORT,
            host="127.0.0.1",
            max_tokens=32768,
        ),
        builtin=True,
    ),
    ServerPreset(
        name="Low Memory",
        description="Conservative settings — limited KV cache to save memory",
        config=MLXServerConfig(
            model=_PLACEHOLDER_MODEL,
            port=DEFAULT_MLX_PORT,
            host="127.0.0.1",
            max_tokens=4096,
            max_kv_size=4096,
        ),
        builtin=True,
    ),
]

RECOMMENDED_MODELS: List[RecommendedModel] = [
    RecommendedModel(
        name="Qwen 2.5 32B Instruct 4-bit ★ Recommended",
        description="Best balance of quality and memory — fits comfortably on 64GB with room for RAG services",
        repo_id="mlx-community/Qwen2.5-32B-Instruct-4bit",
        size_gb=19.0,
        category="llm",
    ),
    RecommendedModel(
        name="Qwen 2.5 32B Instruct 8-bit",
        description="Higher quality 32B — uses ~35GB, excellent quality on 64GB systems",
        repo_id="mlx-community/Qwen2.5-32B-Instruct-8bit",
        size_gb=35.0,
        category="llm",
    ),
    RecommendedModel(
        name="Qwen 2.5 14B Instruct 4-bit",
        description="Great quality at low memory — ideal sweet spot for 64GB systems with heavy multitasking",
        repo_id="mlx-community/Qwen2.5-14B-Instruct-4bit",
        size_gb=8.0,
        category="llm",
    ),
    RecommendedModel(
        name="Qwen 2.5 7B Instruct 4-bit",
        description="Fast and capable — good quality with minimal memory footprint",
        repo_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        size_gb=4.5,
        category="llm",
    ),
    RecommendedModel(
        name="Llama 3.1 8B Instruct 4-bit",
        description="Fast and lightweight — great for testing and quick inference",
        repo_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        size_gb=4.5,
        category="llm",
    ),
    RecommendedModel(
        name="Mistral 7B Instruct v0.3 4-bit",
        description="Efficient multilingual model — very fast on Apple Silicon",
        repo_id="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        size_gb=4.0,
        category="llm",
    ),
    RecommendedModel(
        name="Qwen 2.5 72B Instruct 4-bit",
        description="Top quality 72B — WARNING: requires ~42GB, tight fit on 64GB systems",
        repo_id="mlx-community/Qwen2.5-72B-Instruct-4bit",
        size_gb=42.0,
        category="llm",
    ),
    RecommendedModel(
        name="Llama 3.2 3B Instruct 4-bit",
        description="Tiny and fast — ideal for testing and development",
        repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        size_gb=1.8,
        category="llm",
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


def _get_dir_size(path: str) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except OSError:
        pass
    return total


def _find_local_mlx_models() -> List[MLXModelInfo]:
    """Scan HuggingFace cache for MLX models (directories containing model.safetensors or config.json)."""
    models: List[MLXModelInfo] = []

    if not os.path.isdir(HF_CACHE_DIR):
        return models

    for entry in os.scandir(HF_CACHE_DIR):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue

        # Convert dir name to repo format: models--org--name -> org/name
        parts = entry.name.split("--", 2)
        if len(parts) < 3:
            continue
        repo_name = f"{parts[1]}/{parts[2]}"

        # Check if it looks like an MLX model (has snapshots with safetensors)
        snapshots_dir = os.path.join(entry.path, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        # Find the most recent snapshot
        latest_snapshot = None
        latest_mtime = 0
        for snap in os.scandir(snapshots_dir):
            if snap.is_dir():
                mtime = snap.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_snapshot = snap.path

        if not latest_snapshot:
            continue

        # Check for MLX model indicators
        has_safetensors = any(
            f.endswith(".safetensors") for f in os.listdir(latest_snapshot) if os.path.isfile(os.path.join(latest_snapshot, f))
        )
        has_config = os.path.isfile(os.path.join(latest_snapshot, "config.json"))

        if not (has_safetensors and has_config):
            continue

        # Only include models from mlx-community or with mlx in the name
        if "mlx" not in repo_name.lower():
            continue

        size = _get_dir_size(entry.path)
        stat = entry.stat()

        models.append(MLXModelInfo(
            name=repo_name,
            path=entry.path,
            size_bytes=size,
            size_pretty=_format_size(size),
            modified_at=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        ))

    # Sort by modification date, most recent first
    models.sort(key=lambda m: m.modified_at, reverse=True)
    return models


# ---------------------------------------------------------------------------
# MLXServerManager — Singleton for subprocess lifecycle
# ---------------------------------------------------------------------------

class MLXServerManager:
    """Manages the mlx_lm.server subprocess lifecycle."""

    LOG_BUFFER_SIZE = 2000

    def __init__(self):
        self._lock = threading.RLock()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._status: ServerStatus = ServerStatus.STOPPED
        self._config: Optional[MLXServerConfig] = None
        self._pid: Optional[int] = None
        self._start_time: Optional[float] = None
        self._error_message: Optional[str] = None
        self._log_buffer: Deque[str] = deque(maxlen=self.LOG_BUFFER_SIZE)
        self._log_task_stdout: Optional[asyncio.Task] = None
        self._log_task_stderr: Optional[asyncio.Task] = None

    def _build_args(self, config: MLXServerConfig) -> List[str]:
        """Convert MLXServerConfig to CLI argument list."""
        args = ["python", "-m", "mlx_lm.server"]
        args.extend(["--model", config.model])
        args.extend(["--port", str(config.port)])
        args.extend(["--host", config.host])
        args.extend(["--max-tokens", str(config.max_tokens)])

        if config.max_kv_size is not None:
            args.extend(["--max-kv-size", str(config.max_kv_size)])

        return args

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

    async def start(self, config: MLXServerConfig) -> ServerStatusResponse:
        """Start mlx_lm.server with the given configuration."""
        with self._lock:
            if self._status in (ServerStatus.RUNNING, ServerStatus.STARTING):
                raise HTTPException(
                    status_code=409,
                    detail=f"Server is already {self._status.value}. Stop it first."
                )

        with self._lock:
            self._status = ServerStatus.STARTING
            self._error_message = None
            self._log_buffer.clear()
            self._config = config

        args = self._build_args(config)

        logger.info(f"Starting MLX server: {' '.join(args)}")
        self._log_buffer.append(f"[SYSTEM] Starting: {' '.join(args)}")

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

            # Wait a moment for model loading (MLX is fast but still needs a few seconds)
            await asyncio.sleep(5)

            if process.returncode is not None:
                with self._lock:
                    self._status = ServerStatus.ERROR
                    self._error_message = f"Server exited immediately with code {process.returncode}"
                    self._log_buffer.append(f"[SYSTEM] ERROR: {self._error_message}")
                return self.get_status()

            with self._lock:
                self._status = ServerStatus.RUNNING
                self._log_buffer.append(f"[SYSTEM] Server started on {config.host}:{config.port} (PID {process.pid})")

            # Update the mlx_base_url in settings so RAG uses this server
            base_url = f"http://{config.host}:{config.port}"
            settings_store.update({"mlx_base_url": base_url})
            logger.info(f"Updated mlx_base_url to {base_url}")

            # Monitor process in background
            asyncio.create_task(self._monitor_process(process))

            return self.get_status()

        except FileNotFoundError:
            with self._lock:
                self._status = ServerStatus.ERROR
                self._error_message = "mlx_lm not found. Install with: pip install mlx-lm"
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

    def reset(self) -> ServerStatusResponse:
        """Reset from ERROR state back to STOPPED."""
        with self._lock:
            if self._status == ServerStatus.RUNNING:
                raise HTTPException(status_code=409, detail="Server is running. Use stop instead.")
            if self._status == ServerStatus.STARTING:
                raise HTTPException(status_code=409, detail="Server is starting. Wait or use stop.")
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

    async def stop(self) -> ServerStatusResponse:
        """Stop the running MLX server."""
        with self._lock:
            if self._status not in (ServerStatus.RUNNING, ServerStatus.STARTING, ServerStatus.ERROR):
                return self.get_status()

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
        logger.info(f"Stopping MLX server (PID {process.pid})")

        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                self._log_buffer.append("[SYSTEM] Server terminated gracefully")
            except asyncio.TimeoutError:
                self._log_buffer.append("[SYSTEM] SIGTERM timeout, sending SIGKILL...")
                process.kill()
                await process.wait()
                self._log_buffer.append("[SYSTEM] Server killed")
        except ProcessLookupError:
            self._log_buffer.append("[SYSTEM] Process already gone")
        except Exception as e:
            self._log_buffer.append(f"[SYSTEM] Error stopping: {e}")
            logger.error(f"Error stopping MLX server: {e}")

        for task in [self._log_task_stdout, self._log_task_stderr]:
            if task and not task.done():
                task.cancel()

        with self._lock:
            self._status = ServerStatus.STOPPED
            self._process = None
            self._pid = None
            self._error_message = None

        return self.get_status()

    def get_status(self) -> ServerStatusResponse:
        """Get current server status."""
        with self._lock:
            uptime = None
            uptime_pretty = None
            if self._start_time and self._status == ServerStatus.RUNNING:
                uptime = time.time() - self._start_time
                uptime_pretty = _format_uptime(uptime)

            model_name = self._config.model if self._config else None

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
                mlx_available=MLX_AVAILABLE,
            )

    def get_logs(self, lines: int = 200) -> List[str]:
        """Get the last N log lines."""
        with self._lock:
            buf = list(self._log_buffer)
        return buf[-lines:] if len(buf) > lines else buf


# ---------------------------------------------------------------------------
# ModelDownloadManager — Downloads from HuggingFace via Python API
# ---------------------------------------------------------------------------

class ModelDownloadManager:
    """Manages model downloads from HuggingFace using huggingface_hub Python API with progress tracking."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = "idle"
        self._repo_id: Optional[str] = None
        self._error_message: Optional[str] = None
        self._downloaded_bytes: int = 0
        self._total_bytes: int = 0
        self._current_file: Optional[str] = None
        self._files_done: int = 0
        self._files_total: int = 0

    def _do_download(self, repo_id: str) -> None:
        """Blocking download with progress tracking via tqdm_class parameter."""
        from huggingface_hub import snapshot_download, HfApi
        from tqdm import tqdm as base_tqdm

        manager = self

        # First, get the file list to know total count and total size
        try:
            api = HfApi()
            repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
            siblings = repo_info.siblings or []
            total_size = sum(getattr(s, "size", 0) or 0 for s in siblings)
            total_files = len(siblings)
            with manager._lock:
                manager._total_bytes = total_size
                manager._files_total = total_files
            logger.info(f"Model {repo_id}: {total_files} files, {_format_size(total_size)}")
        except Exception as e:
            logger.warning(f"Could not pre-fetch repo info: {e}")

        class TrackingTqdm(base_tqdm):
            """Custom tqdm passed via tqdm_class to track per-file download progress."""
            def __init__(self, *args, **kwargs):
                # Remove unknown kwargs that huggingface_hub may pass
                # but tqdm doesn't accept (e.g. 'name' in newer versions)
                known_tqdm_kwargs = {
                    'iterable', 'desc', 'total', 'leave', 'file', 'ncols',
                    'mininterval', 'maxinterval', 'miniters', 'ascii', 'disable',
                    'unit', 'unit_scale', 'dynamic_ncols', 'smoothing',
                    'bar_format', 'initial', 'position', 'postfix', 'unit_divisor',
                    'write_bytes', 'lock_args', 'nrows', 'colour', 'delay', 'gui',
                }
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_tqdm_kwargs}
                # Extract file name from 'name' kwarg if present
                file_name = kwargs.get('name', None)
                # Disable console output — we only track internally
                filtered_kwargs["disable"] = False
                super().__init__(*args, **filtered_kwargs)
                desc = file_name or self.desc or ""
                if desc:
                    with manager._lock:
                        manager._current_file = str(desc)

            def update(self, n=1):
                super().update(n)
                with manager._lock:
                    manager._downloaded_bytes += int(n)

            def close(self):
                with manager._lock:
                    manager._files_done += 1
                super().close()

        snapshot_download(repo_id=repo_id, tqdm_class=TrackingTqdm)

    async def download(self, repo_id: str) -> None:
        """Download a model from HuggingFace with real-time progress tracking."""
        with self._lock:
            if self._status == "downloading":
                raise HTTPException(status_code=409, detail="A download is already in progress")
            self._status = "downloading"
            self._repo_id = repo_id
            self._error_message = None
            self._downloaded_bytes = 0
            self._total_bytes = 0
            self._current_file = None
            self._files_done = 0
            self._files_total = 0

        logger.info(f"Downloading MLX model: {repo_id}")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._do_download, repo_id)

            with self._lock:
                self._status = "completed"
            logger.info(f"Download completed: {repo_id}")

        except ImportError:
            with self._lock:
                self._status = "failed"
                self._error_message = "huggingface_hub not installed. Run: pip install huggingface_hub"
            logger.error("huggingface_hub not installed")
        except Exception as e:
            with self._lock:
                self._status = "failed"
                self._error_message = str(e)
            logger.error(f"Download failed: {e}")

    def get_progress(self) -> DownloadProgress:
        """Get current download progress with byte-level detail."""
        with self._lock:
            pct = None
            if self._total_bytes > 0:
                pct = round((self._downloaded_bytes / self._total_bytes) * 100, 1)
            return DownloadProgress(
                status=self._status,
                repo_id=self._repo_id,
                percentage=pct,
                downloaded_bytes=self._downloaded_bytes,
                total_bytes=self._total_bytes,
                downloaded_pretty=_format_size(self._downloaded_bytes) if self._downloaded_bytes else None,
                total_pretty=_format_size(self._total_bytes) if self._total_bytes else None,
                current_file=self._current_file,
                files_done=self._files_done,
                files_total=self._files_total,
                error_message=self._error_message,
            )

    def reset(self):
        """Reset download state to idle."""
        with self._lock:
            self._status = "idle"
            self._repo_id = None
            self._error_message = None
            self._downloaded_bytes = 0
            self._total_bytes = 0
            self._current_file = None
            self._files_done = 0
            self._files_total = 0


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

server_manager = MLXServerManager()
download_manager = ModelDownloadManager()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.post("/start", response_model=ServerStatusResponse)
async def start_server(config: MLXServerConfig):
    """Start MLX server with the given configuration."""
    return await server_manager.start(config)


@router.post("/stop", response_model=ServerStatusResponse)
async def stop_server():
    """Stop the running MLX server."""
    return await server_manager.stop()


@router.post("/reset", response_model=ServerStatusResponse)
async def reset_server():
    """Reset server from error state back to stopped."""
    return server_manager.reset()


@router.get("/status", response_model=ServerStatusResponse)
async def get_server_status():
    """Get current MLX server status."""
    return server_manager.get_status()


@router.get("/logs")
async def get_server_logs(lines: int = 200):
    """Get recent MLX server log lines."""
    return {"logs": server_manager.get_logs(lines)}


@router.get("/models")
async def list_local_models():
    """List available MLX models in the HuggingFace cache."""
    models = _find_local_mlx_models()
    return {"models": models, "cache_dir": HF_CACHE_DIR}


@router.post("/download")
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading a model from HuggingFace."""
    # Check if already downloaded
    existing = _find_local_mlx_models()
    if any(m.name == request.repo_id for m in existing):
        return {"message": f"Model already exists: {request.repo_id}", "already_exists": True}

    # Reset if previous download completed/failed
    progress = download_manager.get_progress()
    if progress.status in ("completed", "failed", "idle"):
        download_manager.reset()

    background_tasks.add_task(download_manager.download, request.repo_id)
    return {"message": f"Download started: {request.repo_id}", "already_exists": False}


@router.get("/download/progress", response_model=DownloadProgress)
async def get_download_progress():
    """Get current model download progress."""
    return download_manager.get_progress()


@router.get("/presets")
async def get_presets():
    """Get available configuration presets (built-in + user-saved)."""
    user_presets_raw = settings_store.get("mlx_presets", [])
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
    user_presets_raw = settings_store.get("mlx_presets", [])
    if not isinstance(user_presets_raw, list):
        user_presets_raw = []

    builtin_names = {p.name.lower() for p in BUILTIN_PRESETS}
    if request.name.lower() in builtin_names:
        raise HTTPException(status_code=400, detail=f"Cannot overwrite built-in preset: {request.name}")

    user_presets_raw = [p for p in user_presets_raw if p.get("name", "").lower() != request.name.lower()]

    new_preset = ServerPreset(
        name=request.name,
        description=request.description,
        config=request.config,
        builtin=False,
    )
    user_presets_raw.append(new_preset.model_dump())

    settings_store.set("mlx_presets", user_presets_raw)

    return {"success": True, "message": f"Preset '{request.name}' saved"}


@router.delete("/config/{name}")
async def delete_preset(name: str):
    """Delete a user-saved preset."""
    builtin_names = {p.name.lower() for p in BUILTIN_PRESETS}
    if name.lower() in builtin_names:
        raise HTTPException(status_code=400, detail=f"Cannot delete built-in preset: {name}")

    user_presets_raw = settings_store.get("mlx_presets", [])
    if not isinstance(user_presets_raw, list):
        return {"success": False, "message": "No user presets found"}

    original_count = len(user_presets_raw)
    user_presets_raw = [p for p in user_presets_raw if p.get("name", "").lower() != name.lower()]

    if len(user_presets_raw) == original_count:
        raise HTTPException(status_code=404, detail=f"Preset not found: {name}")

    settings_store.set("mlx_presets", user_presets_raw)
    return {"success": True, "message": f"Preset '{name}' deleted"}
