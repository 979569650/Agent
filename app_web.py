from pathlib import Path
import os
import threading
import time
import re
import sys
import io
import logging
import uuid
from collections import deque
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field

from config.settings import Settings
from core.observability.metrics import runtime_metrics
from core.security.frequency_guard import SlidingWindowFrequencyGuard
from infra.llm.openai_compatible import OpenAICompatibleClient
from rag_engine import RAGEngine
from workflow.graph import build_app
from workflow.nodes.agent_node import create_agent_node


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)
    ticket_id: str | None = Field(default=None, max_length=64)
    access_code: str | None = Field(default=None, max_length=256)
    abort_ticket: bool = False


class ChatResponse(BaseModel):
    reply: str
    status: str = "ok"
    message: str | None = None
    ticket_id: str | None = None


class NoteWriteRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=240)
    content: str = Field(default="", max_length=200000)


class NoteRenameRequest(BaseModel):
    old_path: str = Field(..., min_length=1, max_length=240)
    new_path: str = Field(..., min_length=1, max_length=240)


class ConfigUpdateRequest(BaseModel):
    values: dict[str, str | int | bool]


class ConsoleBuffer:
    def __init__(self, max_lines: int = 400):
        self._lines = deque(maxlen=max_lines)
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        if not text:
            return
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = normalized.split("\n")
        with self._lock:
            for part in parts:
                if part.strip():
                    self._lines.append(part)

    def tail(self, limit: int = 200) -> str:
        with self._lock:
            return "\n".join(list(self._lines)[-limit:])


class TeeTextIO(io.TextIOBase):
    def __init__(self, stream, buffer: ConsoleBuffer):
        self._stream = stream
        self._buffer = buffer

    def write(self, s):
        text = str(s)
        self._buffer.append(text)
        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


class ConsoleBufferLogHandler(logging.Handler):
    def __init__(self, buffer: ConsoleBuffer):
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append(self.format(record))
        except Exception:
            pass


CONSOLE_BUFFER = ConsoleBuffer()


def _install_console_capture() -> None:
    if not isinstance(sys.stdout, TeeTextIO):
        sys.stdout = TeeTextIO(sys.stdout, CONSOLE_BUFFER)
    if not isinstance(sys.stderr, TeeTextIO):
        sys.stderr = TeeTextIO(sys.stderr, CONSOLE_BUFFER)

    formatter = logging.Formatter("%(levelname)s: %(message)s")
    for logger_name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(logger_name)
        if any(isinstance(h, ConsoleBufferLogHandler) for h in logger.handlers):
            continue
        handler = ConsoleBufferLogHandler(CONSOLE_BUFFER)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


_install_console_capture()


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

load_dotenv(BASE_DIR / ".env")
settings = Settings.from_env()
llm_client = OpenAICompatibleClient(settings)
restricted_query_limiter = SlidingWindowFrequencyGuard(
    limit=settings.restricted_query_limit_per_minute,
    window_seconds=60,
)
rag = RAGEngine(
    data_dir=str(BASE_DIR / "data"),
    db_path=str(BASE_DIR / "faiss_index"),
)
agent_node = create_agent_node(settings, llm_client, rag, restricted_query_limiter)
graph_app = build_app(agent_node)
startup_at = int(time.time())
runtime_lock = threading.Lock()

ENV_FILE = BASE_DIR / ".env"
ENV_EXAMPLE_FILE = BASE_DIR / ".env.example"
LOG_FILE_PATH = Path(settings.log_file).expanduser()
if not LOG_FILE_PATH.is_absolute():
    LOG_FILE_PATH = (BASE_DIR / LOG_FILE_PATH).resolve()
ALLOWED_CONFIG_KEYS = {
    "API_KEY",
    "BASE_URL",
    "MODEL",
    "API_MODE",
    "REQUEST_TIMEOUT",
    "REQUEST_MAX_RETRIES",
    "REQUEST_RETRY_BACKOFF_MS",
    "RERANKER_ENABLED",
    "RERANKER_MODEL",
    "ENABLE_IMAGE_OCR",
    "ENABLE_IMAGE_VLM",
    "VISION_MODEL",
    "ALLOWED_TOOLS",
    "OTEL_ENABLED",
    "LOG_FILE",
    "RESTRICTED_QUERY_LIMIT_PER_MINUTE",
}


class IndexTaskManager:
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self._lock = threading.Lock()
        self._status = {
            "state": "idle",  # idle / running / success / error
            "progress": 0,
            "message": "ready",
            "started_at": None,
            "finished_at": None,
            "last_error": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._status)

    def start_rebuild(self) -> bool:
        with self._lock:
            if self._status["state"] == "running":
                return False
            self._status.update(
                {
                    "state": "running",
                    "progress": 5,
                    "message": "index rebuild started",
                    "started_at": int(time.time()),
                    "finished_at": None,
                    "last_error": None,
                }
            )

        t = threading.Thread(target=self._run_rebuild, daemon=True)
        t.start()
        return True

    def _run_rebuild(self) -> None:
        try:
            with self._lock:
                self._status.update({"progress": 20, "message": "building index"})
            self.rag.build_index()
            with self._lock:
                self._status.update(
                    {
                        "state": "success",
                        "progress": 100,
                        "message": "index rebuild finished",
                        "finished_at": int(time.time()),
                    }
                )
        except Exception as e:
            with self._lock:
                self._status.update(
                    {
                        "state": "error",
                        "progress": 100,
                        "message": "index rebuild failed",
                        "finished_at": int(time.time()),
                        "last_error": f"{type(e).__name__}: {e}",
                    }
                )


index_task_manager = IndexTaskManager(rag)
security_ticket_store: dict[str, dict] = {}
security_ticket_lock = threading.Lock()

app = FastAPI(title="AgentLearn Web API", version="0.1.0")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


def _system_prompt() -> str:
    return (
        "你是一个智能个人助理。\n"
        "1. 你的首要任务是帮助用户管理和检索【本地笔记】。\n"
        "2. 你会收到系统注入的本地检索上下文，请优先基于该上下文回答。\n"
        "3. 若本地检索无结果，请直接说明“未找到相关本地笔记”。\n"
        "4. 回答要简洁明了，引用笔记内容时请说明。"
    )


def _bool_str(v: str | bool | int) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    s = str(v).strip()
    if s.lower() in {"true", "false"}:
        return s.lower()
    return s


def _mask_key(key: str, value: str) -> str:
    if key == "API_KEY" and value:
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}***{value[-4:]}"
    return value


def _load_env_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if not m:
            continue
        env[m.group(1)] = m.group(2)
    return env


def _ticket_ttl_seconds() -> int:
    return 300


def _cleanup_security_tickets() -> None:
    now = time.time()
    with security_ticket_lock:
        expired_ids = [
            cid for cid, item in security_ticket_store.items()
            if now - float(item.get("created_at", 0)) > _ticket_ttl_seconds()
        ]
        for cid in expired_ids:
            security_ticket_store.pop(cid, None)


def _create_security_ticket(message: str, history: list[ChatTurn]) -> str:
    _cleanup_security_tickets()
    ticket_id = uuid.uuid4().hex[:12]
    with security_ticket_lock:
        security_ticket_store[ticket_id] = {
            "message": message,
            "history": [turn.model_dump() for turn in history[-20:]],
            "created_at": time.time(),
        }
    return ticket_id


def _get_security_ticket(ticket_id: str | None) -> dict | None:
    if not ticket_id:
        return None
    _cleanup_security_tickets()
    with security_ticket_lock:
        item = security_ticket_store.get(ticket_id)
        return dict(item) if item else None


def _delete_security_ticket(ticket_id: str | None) -> None:
    if not ticket_id:
        return
    with security_ticket_lock:
        security_ticket_store.pop(ticket_id, None)


def _write_env_map(path: Path, updates: dict[str, str]) -> None:
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    used: set[str] = set()
    out_lines: list[str] = []
    pattern = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$")

    for line in existing_lines:
        m = pattern.match(line)
        if not m:
            out_lines.append(line)
            continue
        key = m.group(2)
        if key in updates:
            out_lines.append(f"{key}={updates[key]}")
            used.add(key)
        else:
            out_lines.append(line)

    for key, value in updates.items():
        if key in used:
            continue
        out_lines.append(f"{key}={value}")

    path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def _refresh_runtime_from_env() -> None:
    global settings, llm_client, restricted_query_limiter, agent_node, graph_app, LOG_FILE_PATH
    with runtime_lock:
        load_dotenv(ENV_FILE, override=True)
        load_dotenv(BASE_DIR / ".env", override=False)
        settings = Settings.from_env()
        LOG_FILE_PATH = Path(settings.log_file).expanduser()
        if not LOG_FILE_PATH.is_absolute():
            LOG_FILE_PATH = (BASE_DIR / LOG_FILE_PATH).resolve()
        llm_client = OpenAICompatibleClient(settings)
        restricted_query_limiter = SlidingWindowFrequencyGuard(
            limit=settings.restricted_query_limit_per_minute,
            window_seconds=60,
        )
        agent_node = create_agent_node(settings, llm_client, rag, restricted_query_limiter)
        graph_app = build_app(agent_node)


def _notes_root() -> Path:
    root = Path(rag.data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_note_path(rel_path: str) -> Path:
    rel = (rel_path or "").strip().replace("\\", "/")
    if not rel:
        raise HTTPException(status_code=400, detail="empty path")
    if rel.startswith("/") or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="invalid path")

    p = (_notes_root() / rel).resolve()
    if _notes_root() not in p.parents and p != _notes_root():
        raise HTTPException(status_code=400, detail="path out of data dir")
    return p


def _ensure_text_ext(path_obj: Path) -> None:
    if path_obj.suffix.lower() not in {".md", ".txt"}:
        raise HTTPException(status_code=400, detail="only .md/.txt supported")


@app.on_event("startup")
def _startup() -> None:
    if not ENV_FILE.exists() and ENV_EXAMPLE_FILE.exists():
        ENV_FILE.write_text(ENV_EXAMPLE_FILE.read_text(encoding="utf-8"), encoding="utf-8")

    try:
        if not rag.load_index():
            rag.build_index()
    except Exception as e:
        print(f"⚠️ 启动时索引初始化失败，已降级继续运行: {type(e).__name__}: {e}")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": settings.model_name,
        "api_mode": settings.api_mode,
        "config_ready": bool(settings.base_url and settings.api_key),
    }


@app.get("/api/config")
def get_config() -> dict:
    env = _load_env_map(ENV_FILE)
    defaults = _load_env_map(ENV_EXAMPLE_FILE)
    keys = sorted(ALLOWED_CONFIG_KEYS)
    values = {k: env.get(k, defaults.get(k, "")) for k in keys}
    masked = {k: _mask_key(k, str(v)) for k, v in values.items()}
    return {"ok": True, "values": masked, "defaults": {k: defaults.get(k, "") for k in keys}}


@app.put("/api/config")
def update_config(req: ConfigUpdateRequest) -> dict:
    updates: dict[str, str] = {}
    for k, v in (req.values or {}).items():
        if k not in ALLOWED_CONFIG_KEYS:
            raise HTTPException(status_code=400, detail=f"unsupported key: {k}")
        updates[k] = _bool_str(v)

    if updates:
        _write_env_map(ENV_FILE, updates)
        for k, v in updates.items():
            import os
            os.environ[k] = v
        _refresh_runtime_from_env()
    return {"ok": True, "updated": sorted(list(updates.keys()))}


@app.post("/api/config/reset")
def reset_config_to_defaults() -> dict:
    defaults = _load_env_map(ENV_EXAMPLE_FILE)
    updates = {k: defaults.get(k, "") for k in ALLOWED_CONFIG_KEYS if k in defaults}
    _write_env_map(ENV_FILE, updates)
    for k, v in updates.items():
        import os
        os.environ[k] = v
    _refresh_runtime_from_env()
    return {"ok": True, "reset": sorted(list(updates.keys()))}


@app.post("/api/config/test")
def test_config_connection() -> dict:
    try:
        preview = llm_client.call_model([HumanMessage(content="你好")])
        return {"ok": True, "message": "connection success", "preview": (preview or "")[:120]}
    except Exception as e:
        return {"ok": False, "message": f"connection failed: {type(e).__name__}: {e}"}


@app.get("/api/system/status")
def system_status() -> dict:
    files = rag.list_note_files()
    config_ready = bool(settings.base_url and settings.api_key)
    return {
        "ok": True,
        "uptime_seconds": max(0, int(time.time()) - startup_at),
        "model": settings.model_name,
        "api_mode": settings.api_mode,
        "config_ready": config_ready,
        "status_hint": "ready" if config_ready else "config_incomplete",
        "index_task": index_task_manager.snapshot(),
        "notes_count": len(files),
        "metrics": runtime_metrics.snapshot(),
        "features": {
            "reranker_enabled": settings.reranker_enabled,
            "image_ocr_enabled": rag.enable_image_ocr,
            "image_vlm_enabled": rag.enable_image_vlm,
        },
    }


@app.get("/api/admin/diagnostics")
def admin_diagnostics() -> dict:
    def tail_text(path: Path, limit: int = 12000) -> str:
        try:
            if not path.exists() or not path.is_file():
                return ""
            text = path.read_text(encoding="utf-8", errors="ignore")
            return text[-limit:]
        except Exception as e:
            return f"<读取失败: {type(e).__name__}: {e}>"

    env_map = _load_env_map(ENV_FILE)
    masked_env = {k: _mask_key(k, v) for k, v in env_map.items()}

    return {
        "ok": True,
        "paths": {
            "base_dir": str(BASE_DIR),
            "env_file": str(ENV_FILE),
            "env_example_file": str(ENV_EXAMPLE_FILE),
            "data_dir": str(_notes_root()),
            "faiss_index_dir": str(Path(rag.db_path).resolve()),
            "log_file": str(LOG_FILE_PATH),
        },
        "config": {
            "config_ready": bool(settings.base_url and settings.api_key),
            "model": settings.model_name,
            "api_mode": settings.api_mode,
            "request_timeout": settings.request_timeout,
            "request_max_retries": settings.request_max_retries,
            "request_retry_backoff_ms": settings.request_retry_backoff_ms,
            "env_values": masked_env,
        },
        "runtime": {
            "startup_at": startup_at,
            "uptime_seconds": max(0, int(time.time()) - startup_at),
            "index_task": index_task_manager.snapshot(),
            "metrics": runtime_metrics.snapshot(),
            "rag": {
                "rag_ready": rag.rag_ready,
                "image_ready": rag.image_ready,
                "image_ocr_enabled": rag.enable_image_ocr,
                "image_vlm_enabled": rag.enable_image_vlm,
                "notes_count": len(rag.list_note_files()),
            },
        },
        "recent_logs": {
            "console_tail": CONSOLE_BUFFER.tail(),
            "events_jsonl_tail": tail_text(LOG_FILE_PATH),
        },
    }


@app.get("/api/admin/console")
def admin_console_tail(limit: int = Query(200, ge=20, le=1000)) -> dict:
    return {
        "ok": True,
        "console_tail": CONSOLE_BUFFER.tail(limit=limit),
        "index_task": index_task_manager.snapshot(),
        "uptime_seconds": max(0, int(time.time()) - startup_at),
    }


@app.get("/api/notes")
def list_notes() -> dict:
    return {"files": rag.list_note_files()}


@app.get("/api/notes/content")
def get_note_content(path: str = Query(..., min_length=1, max_length=240)) -> dict:
    p = _safe_note_path(path)
    _ensure_text_ext(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="note not found")
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = p.read_text(encoding="gbk")
    return {"path": path, "content": content}


@app.post("/api/notes")
def create_note(req: NoteWriteRequest) -> dict:
    p = _safe_note_path(req.path)
    _ensure_text_ext(p)
    if p.exists():
        raise HTTPException(status_code=409, detail="note already exists")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(req.content or "", encoding="utf-8")
    return {"ok": True, "path": req.path}


@app.put("/api/notes")
def update_note(req: NoteWriteRequest) -> dict:
    p = _safe_note_path(req.path)
    _ensure_text_ext(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="note not found")
    p.write_text(req.content or "", encoding="utf-8")
    return {"ok": True, "path": req.path}


@app.post("/api/notes/rename")
def rename_note(req: NoteRenameRequest) -> dict:
    old_p = _safe_note_path(req.old_path)
    new_p = _safe_note_path(req.new_path)
    _ensure_text_ext(old_p)
    _ensure_text_ext(new_p)
    if not old_p.exists() or not old_p.is_file():
        raise HTTPException(status_code=404, detail="source note not found")
    if new_p.exists():
        raise HTTPException(status_code=409, detail="target note already exists")
    new_p.parent.mkdir(parents=True, exist_ok=True)
    old_p.rename(new_p)
    return {"ok": True, "old_path": req.old_path, "new_path": req.new_path}


@app.delete("/api/notes")
def delete_note(path: str = Query(..., min_length=1, max_length=240)) -> dict:
    p = _safe_note_path(path)
    _ensure_text_ext(p)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="note not found")
    p.unlink()
    return {"ok": True, "path": path}


@app.post("/api/notes/upload")
async def upload_note(file: UploadFile = File(...)) -> dict:
    rel = (file.filename or "").strip().replace("\\", "/")
    if not rel:
        raise HTTPException(status_code=400, detail="empty filename")
    p = _safe_note_path(rel)
    ext = p.suffix.lower()
    if ext not in {".md", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="unsupported extension")
    data = await file.read()
    p.parent.mkdir(parents=True, exist_ok=True)
    if ext in {".md", ".txt"}:
        text = data.decode("utf-8", errors="ignore")
        p.write_text(text, encoding="utf-8")
    else:
        p.write_bytes(data)
    return {"ok": True, "path": rel, "size": len(data)}


@app.post("/api/index/rebuild")
def rebuild_index() -> dict:
    started = index_task_manager.start_rebuild()
    if not started:
        return {"ok": True, "message": "already running", "status": index_task_manager.snapshot()}
    return {"ok": True, "message": "rebuild started", "status": index_task_manager.snapshot()}


@app.get("/api/index/status")
def rebuild_status() -> dict:
    return {"ok": True, "status": index_task_manager.snapshot()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        active_message = req.message
        active_history = list(req.history[-20:])

        if req.ticket_id:
            ticket = _get_security_ticket(req.ticket_id)
            if ticket is None:
                raise HTTPException(status_code=400, detail="security ticket expired or invalid")
            active_message = str(ticket.get("message") or req.message)
            active_history = [ChatTurn(**turn) for turn in ticket.get("history", [])]

        messages = [SystemMessage(content=_system_prompt())]
        for turn in active_history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))
        messages.append(HumanMessage(content=active_message))

        state = {
            "messages": messages,
            "access_code": req.access_code,
            "ticket_id": req.ticket_id,
            "abort_ticket": req.abort_ticket,
        }

        final_event = None
        final_response = None
        for event in graph_app.stream(state, stream_mode="values"):
            final_event = event
            final_response = event["messages"][-1]

        if final_response is None or final_event is None:
            raise RuntimeError("empty response")

        ticket_required = bool(final_event.get("ticket_required"))
        response_ticket_id = final_event.get("ticket_id")
        security_status = str(final_event.get("security_status") or "")
        if ticket_required and response_ticket_id:
            ticket_id = _create_security_ticket(active_message, active_history)
            return ChatResponse(
                reply="",
                status="awaiting_access_code",
                message=str(final_response.content),
                ticket_id=ticket_id,
            )

        if req.ticket_id and security_status != "verification_failed":
            _delete_security_ticket(req.ticket_id)

        final_text = str(final_response.content)
        if security_status == "aborted":
            return ChatResponse(reply=final_text, status="aborted", message=final_text)
        if security_status == "verification_failed":
            return ChatResponse(
                reply=final_text,
                status="verification_failed",
                message=final_text,
                ticket_id=req.ticket_id,
            )
        if req.access_code is not None and req.ticket_id:
            return ChatResponse(reply=final_text, status="verified", message="访问口令核验通过")
        return ChatResponse(reply=final_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat failed: {type(e).__name__}: {e}")


