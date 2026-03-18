import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_name: str = "gpt-5.3-codex"
    api_mode: str = "responses"
    request_timeout: int = 45
    request_max_retries: int = 2
    request_retry_backoff_ms: int = 500
    base_url: str = ""
    api_key: str = ""
    access_code_hash: str = ""
    restricted_query_limit_per_minute: int = 3
    allowed_tools: tuple[str, ...] = ("search_notes", "list_note_files")
    log_file: str = "observability/events.jsonl"
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    @classmethod
    def from_env(cls) -> "Settings":
        mode = (os.getenv("API_MODE", "responses") or "responses").strip().lower()
        if mode not in {"responses", "chat"}:
            mode = "responses"

        timeout_raw = os.getenv("REQUEST_TIMEOUT", "45")
        try:
            timeout = max(5, int(timeout_raw))
        except Exception:
            timeout = 45

        retries_raw = os.getenv("REQUEST_MAX_RETRIES", "2")
        backoff_raw = os.getenv("REQUEST_RETRY_BACKOFF_MS", "500")
        restricted_limit_raw = os.getenv("RESTRICTED_QUERY_LIMIT_PER_MINUTE", "3")
        try:
            retries = max(0, int(retries_raw))
        except Exception:
            retries = 2
        try:
            backoff_ms = max(100, int(backoff_raw))
        except Exception:
            backoff_ms = 500
        try:
            restricted_limit = max(1, int(restricted_limit_raw))
        except Exception:
            restricted_limit = 3

        tools_raw = (os.getenv("ALLOWED_TOOLS") or "search_notes,list_note_files").strip()
        if tools_raw:
            allowed_tools = tuple([x.strip() for x in tools_raw.split(",") if x.strip()])
        else:
            allowed_tools = ("search_notes", "list_note_files")

        log_file = (os.getenv("LOG_FILE") or "observability/events.jsonl").strip()
        reranker_enabled = (os.getenv("RERANKER_ENABLED", "false") or "false").strip().lower() == "true"
        reranker_model = (os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-base").strip()

        return cls(
            model_name=os.getenv("MODEL", "gpt-5.3-codex"),
            api_mode=mode,
            request_timeout=timeout,
            request_max_retries=retries,
            request_retry_backoff_ms=backoff_ms,
            base_url=(os.getenv("BASE_URL") or "").rstrip("/"),
            api_key=os.getenv("API_KEY") or "",
            access_code_hash=os.getenv("SECURITY_KEY_HASH") or "",
            restricted_query_limit_per_minute=restricted_limit,
            allowed_tools=allowed_tools,
            log_file=log_file,
            reranker_enabled=reranker_enabled,
            reranker_model=reranker_model,
        )

    def validate_startup(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("BASE_URL")
        if not self.api_key:
            missing.append("API_KEY")
        if missing:
            raise RuntimeError(f"缺少必要环境变量: {', '.join(missing)}")

