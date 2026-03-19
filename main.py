import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from cli.fullscreen_terminal import AlternateScreenSession, safe_print, supports_fullscreen
from config.settings import Settings
from core.observability.logger import log_event
from core.observability.metrics import runtime_metrics
from core.security.frequency_guard import SlidingWindowFrequencyGuard
from infra.llm.openai_compatible import OpenAICompatibleClient
from rag_engine import RAGEngine
from workflow.graph import build_app
from workflow.nodes.agent_node import create_agent_node

CLI_VERSION = "0.3.0"

EXIT_OK = 0
EXIT_GENERIC_ERROR = 1
EXIT_INTERRUPTED = 130


@dataclass
class CliFlags:
    quiet: bool = False
    verbose: bool = False
    color_mode: str = "auto"


FLAGS = CliFlags()


COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"


def color_enabled() -> bool:
    if FLAGS.color_mode == "always":
        return True
    if FLAGS.color_mode == "never":
        return False
    stream = getattr(sys, "stdout", None)
    return bool(stream and hasattr(stream, "isatty") and stream.isatty())


def paint(text: str, *styles: str) -> str:
    if not styles or not color_enabled():
        return text
    return "".join(styles) + text + COLOR_RESET


def ui_title(text: str) -> str:
    return paint(text, COLOR_BOLD, COLOR_CYAN)


def ui_section(text: str) -> str:
    return paint(text, COLOR_BOLD, COLOR_BLUE)


def ui_user_prompt() -> str:
    return paint("你", COLOR_BOLD, COLOR_GREEN) + " › "


def ui_assistant_prefix() -> str:
    return paint("助手", COLOR_BOLD, COLOR_CYAN) + " ›"


def ui_warn(text: str) -> str:
    return paint(text, COLOR_YELLOW)


def ui_error(text: str) -> str:
    return paint(text, COLOR_RED)


def ui_hint(text: str) -> str:
    return paint(text, COLOR_DIM)


def ui_label(text: str) -> str:
    return paint(text, COLOR_BOLD)


def _terminal_width(fallback: int = 88) -> int:
    try:
        return max(40, shutil.get_terminal_size((fallback, 20)).columns)
    except Exception:
        return fallback


def ui_divider(char: str = "─", width: Optional[int] = None) -> str:
    line_width = width if width is not None else _terminal_width()
    line_width = max(8, int(line_width))
    return ui_hint(char * line_width)


def ui_badge(text: str) -> str:
    return paint(f"[{text}]", COLOR_BOLD, COLOR_BLUE)


def ui_kv(key: str, value: str) -> str:
    return f"{ui_label(key)} {value}"


def _shorten_middle(text: str, max_len: int = 68, ellipsis: str = "...") -> str:
    if max_len <= len(ellipsis) + 2:
        return text[:max_len]
    if len(text) <= max_len:
        return text

    remain = max_len - len(ellipsis)
    left = remain // 2
    right = remain - left
    return f"{text[:left]}{ellipsis}{text[-right:]}"


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def out(message: str = "", force: bool = False) -> None:
    if force or not FLAGS.quiet:
        print(message)


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def json_out(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run_silenced(func, *args, **kwargs):
    if FLAGS.verbose:
        return func(*args, **kwargs)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        return func(*args, **kwargs)


def build_runtime() -> Tuple[Settings, RAGEngine, Any]:
    load_dotenv()
    settings = Settings.from_env()
    llm_client = OpenAICompatibleClient(settings)
    restricted_query_limiter = SlidingWindowFrequencyGuard(
        limit=settings.restricted_query_limit_per_minute,
        window_seconds=60,
    )
    rag = RAGEngine()
    agent_node = create_agent_node(settings, llm_client, rag, restricted_query_limiter)
    app = build_app(agent_node)
    return settings, rag, app


def safe_build_runtime():
    try:
        runtime = run_silenced(build_runtime)
        return runtime, None
    except Exception as e:
        return None, e


def get_system_message() -> SystemMessage:
    return SystemMessage(
        content="""
你是一个智能个人助理。
1. 你的首要任务是帮助用户管理和检索【本地笔记】。
2. 你会收到系统注入的本地检索上下文，请优先基于该上下文回答。
3. 若本地检索无结果，请直接说明“未找到相关本地笔记”。
4. 回答要简洁明了，引用笔记内容时请说明。
"""
    )


def gather_config(settings: Settings, rag: RAGEngine) -> Dict[str, Any]:
    return {
        "model": settings.model_name,
        "base_url": settings.base_url,
        "api_mode": settings.api_mode,
        "reranker_enabled": settings.reranker_enabled,
        "image_vlm_enabled": rag.enable_image_vlm,
        "image_ocr_enabled": rag.enable_image_ocr,
        "allowed_tools": settings.allowed_tools or [],
    }


def gather_metrics() -> Dict[str, Any]:
    m = runtime_metrics.snapshot()
    return {
        "requests_total": m.get("requests_total", 0),
        "retrieval_hit_total": m.get("retrieval_hit_total", 0),
        "retrieval_miss_total": m.get("retrieval_miss_total", 0),
        "error_total": m.get("error_total", 0),
        "avg_latency_ms": m.get("avg_latency_ms", 0),
        "retrieval_hit_rate": m.get("retrieval_hit_rate", 0),
        "error_rate": m.get("error_rate", 0),
    }


def gather_files(rag: RAGEngine) -> List[str]:
    return rag.list_note_files()


def format_help_card() -> str:
    lines = [
        "",
        ui_section("可用命令"),
        ui_hint("  / 基础"),
        "  help / h / ?      查看帮助",
        "  q / quit / exit   退出",
        ui_hint("  / 信息"),
        "  files             列出本地笔记文件",
        "  config            查看当前关键配置",
        "  metrics           查看会话指标",
        "  update            强制重建索引",
        "",
        ui_section("提问示例"),
        "  帮我总结我写过的 LangGraph 笔记",
        "  列出我最近的项目 TODO",
        "  看一下这张图片里的内容，不要ocr",
        "",
    ]
    return "\n".join(lines)


def print_help_card() -> None:
    out(format_help_card())


def format_config_text(config_data: Dict[str, Any]) -> str:
    allowed_tools = config_data.get("allowed_tools", [])
    lines = [
        "",
        ui_section("当前关键配置"),
        f"- {ui_label('model')} {config_data['model']}",
        f"- {ui_label('base_url')} {config_data['base_url']}",
        f"- {ui_label('api_mode')} {config_data['api_mode']}",
        f"- {ui_label('reranker_enabled')} {config_data['reranker_enabled']}",
        f"- {ui_label('image_vlm_enabled')} {config_data['image_vlm_enabled']}",
        f"- {ui_label('image_ocr_enabled')} {config_data['image_ocr_enabled']}",
        f"- {ui_label('allowed_tools')} {', '.join(allowed_tools) if allowed_tools else '(none)'}",
        "",
    ]
    return "\n".join(lines)


def print_config_text(config_data: Dict[str, Any]) -> None:
    out(format_config_text(config_data))


def format_metrics_text(metrics_data: Dict[str, Any]) -> str:
    lines = [
        "",
        ui_section("会话指标快照（进程内）"),
        f"- {ui_label('requests_total')} {metrics_data['requests_total']}",
        f"- {ui_label('retrieval_hit_total')} {metrics_data['retrieval_hit_total']}",
        f"- {ui_label('retrieval_miss_total')} {metrics_data['retrieval_miss_total']}",
        f"- {ui_label('error_total')} {metrics_data['error_total']}",
        f"- {ui_label('avg_latency_ms')} {metrics_data['avg_latency_ms']}",
        f"- {ui_label('retrieval_hit_rate')} {metrics_data['retrieval_hit_rate']}",
        f"- {ui_label('error_rate')} {metrics_data['error_rate']}",
        "",
    ]
    return "\n".join(lines)


def print_metrics_text(metrics_data: Dict[str, Any]) -> None:
    out(format_metrics_text(metrics_data))


def format_files_text(files: List[str]) -> str:
    if files:
        lines = ["", ui_section("当前可见本地笔记文件：")]
        lines.extend([f"- {f}" for f in files])
        lines.append("")
        return "\n".join(lines)
    return "\n" + ui_warn("当前 data/ 目录下未发现可用笔记文件（.md/.txt/.图片）。") + "\n"


def print_files_text(files: List[str]) -> None:
    out(format_files_text(files))


def ensure_startup(settings: Settings, rag: RAGEngine) -> bool:
    try:
        settings.validate_startup()
    except Exception as e:
        eprint(ui_error(f"启动配置校验失败: {e}"))
        return False

    try:
        if not run_silenced(rag.load_index):
            run_silenced(rag.build_index)
    except Exception as e:
        eprint(ui_error(f"索引初始化失败: {e}"))
        return False

    return True


def run_agent_once(app, chat_history, stream_sink: Optional[Callable[[str], None]] = None):
    def _run():
        final_response = None
        event = None
        state = {"messages": chat_history}
        if stream_sink is not None:
            state["stream_sink"] = stream_sink
        for event in app.stream(state, stream_mode="values"):
            final_response = event["messages"][-1]
        if event is None or final_response is None:
            raise RuntimeError("未获取到模型响应")
        return final_response, event["messages"]

    return run_silenced(_run)


def _build_status_lines(settings: Settings, no_banner: bool = False) -> List[str]:
    show_banner = not no_banner and not FLAGS.quiet
    if not show_banner:
        return []

    width = _terminal_width()
    endpoint = _shorten_middle(str(settings.base_url), max_len=max(36, width - 14))

    lines: List[str] = [
        ui_divider(width=width),
        ui_title("Agent CLI"),
        ui_hint("本地智能笔记助理"),
        f"{ui_badge(f'model {settings.model_name}')} {ui_badge(f'mode {settings.api_mode}')}",
        ui_kv("endpoint", endpoint),
        ui_hint("命令: help 查看帮助 · files/config/metrics/update · quit 退出"),
        ui_divider(width=width),
    ]
    return lines


def stream_reindex_progress(rag: RAGEngine, on_line) -> None:
    class _LiveWriter:
        def __init__(self, callback):
            self._callback = callback
            self._buffer = ""

        def write(self, data):
            if not data:
                return 0
            self._buffer += str(data)
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line:
                    self._callback(line)
            return len(data)

        def flush(self):
            if self._buffer:
                line = self._buffer.rstrip("\r")
                if line:
                    self._callback(line)
                self._buffer = ""

    writer = _LiveWriter(on_line)
    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
        rag.build_index()
    writer.flush()


def _is_interactive_command(user_input: str) -> bool:
    return user_input.lower() in ["q", "quit", "exit", "help", "h", "?", "config", "metrics", "files", "update"]


def _run_chat_command(
    user_input: str,
    settings: Settings,
    rag: RAGEngine,
    app,
    chat_history,
    on_progress_line: Optional[Callable[[str], None]] = None,
    on_assistant_chunk: Optional[Callable[[str], None]] = None,
):
    lowered = user_input.lower()
    if lowered in ["q", "quit", "exit"]:
        return {"kind": "exit", "code": EXIT_OK, "message": ui_warn("已退出。"), "chat_history": chat_history}

    if lowered in ["help", "h", "?"]:
        return {"kind": "text", "text": format_help_card(), "chat_history": chat_history}

    if lowered == "config":
        return {"kind": "text", "text": format_config_text(gather_config(settings, rag)), "chat_history": chat_history}

    if lowered == "metrics":
        return {"kind": "text", "text": format_metrics_text(gather_metrics()), "chat_history": chat_history}

    if lowered == "files":
        return {"kind": "text", "text": format_files_text(gather_files(rag)), "chat_history": chat_history}

    if lowered == "update":
        if on_progress_line is not None:
            stream_reindex_progress(rag, on_progress_line)
            run_silenced(log_event, "manual_reindex", "console", result="triggered")
            return {"kind": "text", "text": ui_hint("索引重建完成。"), "chat_history": chat_history}

        lines: List[str] = []
        stream_reindex_progress(rag, lines.append)
        run_silenced(log_event, "manual_reindex", "console", result="triggered")
        text = "\n".join(lines) if lines else ui_hint("索引重建完成。")
        return {"kind": "text", "text": text, "chat_history": chat_history}

    next_history = list(chat_history)
    next_history.append(HumanMessage(content=user_input))
    final_response, new_history = run_agent_once(app, next_history, stream_sink=on_assistant_chunk)
    return {
        "kind": "assistant",
        "answer": final_response.content,
        "chat_history": new_history,
    }


def run_chat_interactive_plain(settings: Settings, rag: RAGEngine, app, no_banner: bool = False) -> int:
    if not ensure_startup(settings, rag):
        return EXIT_GENERIC_ERROR

    for line in _build_status_lines(settings, no_banner=no_banner):
        out(line)

    chat_history = [get_system_message()]

    while True:
        try:
            user_input = input("\n" + ui_user_prompt()).strip()
            if not user_input:
                continue

            is_plain_question = not _is_interactive_command(user_input)
            if not FLAGS.quiet and is_plain_question:
                out("\n" + ui_hint("助手思考中..."), force=True)

            terminal_stream = getattr(sys, "__stdout__", sys.stdout)
            streamed_any = False
            streamed_text_parts: List[str] = []

            def _on_assistant_chunk(chunk: str) -> None:
                nonlocal streamed_any
                if not is_plain_question:
                    return
                if not chunk:
                    return
                streamed_text_parts.append(chunk)
                if not streamed_any:
                    terminal_stream.write("\n" + ui_assistant_prefix() + "\n")
                    streamed_any = True
                terminal_stream.write(chunk)
                if hasattr(terminal_stream, "flush"):
                    terminal_stream.flush()

            result = _run_chat_command(
                user_input,
                settings,
                rag,
                app,
                chat_history,
                on_progress_line=lambda line: safe_print(line, stream=getattr(sys, "__stdout__", sys.stdout)),
                on_assistant_chunk=_on_assistant_chunk if is_plain_question else None,
            )
            chat_history = result["chat_history"]

            if result["kind"] == "exit":
                out(result["message"], force=True)
                return result["code"]

            if result["kind"] == "text":
                out(result["text"])
                continue

            if streamed_any:
                streamed_text = "".join(streamed_text_parts)
                if streamed_text != str(result["answer"]):
                    terminal_stream.write("\r\n")
                    terminal_stream.write(ui_warn("[提示] 流式与最终结果不一致，以下为最终答案：") + "\n")
                    terminal_stream.write(str(result["answer"]))
                out("", force=True)
            else:
                out("\n" + ui_assistant_prefix(), force=True)
                out(result["answer"], force=True)

        except KeyboardInterrupt:
            out("\n" + ui_warn("已中断。"), force=True)
            return EXIT_INTERRUPTED
        except EOFError:
            out("\n" + ui_warn("已退出。"), force=True)
            return EXIT_OK
        except Exception as e:
            eprint(ui_error(f"发生错误: {type(e).__name__}: {e}"))
            if FLAGS.verbose:
                traceback.print_exc(file=sys.stderr)
            return EXIT_GENERIC_ERROR


def run_chat_interactive_fullscreen(settings: Settings, rag: RAGEngine, app, no_banner: bool = False) -> int:
    if not ensure_startup(settings, rag):
        return EXIT_GENERIC_ERROR

    screen_lines: List[str] = []
    screen_lines.extend(_build_status_lines(settings, no_banner=no_banner))

    chat_history = [get_system_message()]

    def redraw() -> None:
        terminal_stream = getattr(sys, "__stdout__", sys.stdout)
        safe_print("\x1b[2J\x1b[H", stream=terminal_stream)
        if screen_lines:
            safe_print("\n".join(screen_lines), stream=terminal_stream)
        safe_print("", stream=terminal_stream)

    def redraw_with_pending(pending_assistant: str) -> None:
        terminal_stream = getattr(sys, "__stdout__", sys.stdout)
        safe_print("\x1b[2J\x1b[H", stream=terminal_stream)
        lines = list(screen_lines)
        if pending_assistant:
            lines.append(ui_assistant_prefix())
            lines.append(pending_assistant)
        if lines:
            safe_print("\n".join(lines), stream=terminal_stream)
        safe_print("", stream=terminal_stream)

    try:
        with AlternateScreenSession():
            redraw()
            while True:
                try:
                    user_input = input(ui_user_prompt()).strip()
                    if not user_input:
                        continue

                    screen_lines.append(f"{ui_user_prompt()}{user_input}")
                    is_plain_question = not _is_interactive_command(user_input)
                    if not FLAGS.quiet and is_plain_question:
                        screen_lines.append(ui_hint("助手思考中..."))
                    redraw()

                    pending_assistant = ""
                    last_redraw_at = 0.0

                    def _on_assistant_chunk(chunk: str) -> None:
                        nonlocal pending_assistant, last_redraw_at
                        if not is_plain_question:
                            return
                        if not chunk:
                            return
                        pending_assistant += chunk
                        now = time.perf_counter()
                        if now - last_redraw_at >= 0.05:
                            redraw_with_pending(pending_assistant)
                            last_redraw_at = now

                    result = _run_chat_command(
                        user_input,
                        settings,
                        rag,
                        app,
                        chat_history,
                        on_progress_line=lambda line: (screen_lines.append(line), redraw()),
                        on_assistant_chunk=_on_assistant_chunk if is_plain_question else None,
                    )
                    chat_history = result["chat_history"]

                    if result["kind"] == "exit":
                        return result["code"]

                    if result["kind"] == "text":
                        screen_lines.append(result["text"])
                    else:
                        if pending_assistant:
                            screen_lines.append(ui_assistant_prefix())
                            screen_lines.append(result["answer"])
                        else:
                            screen_lines.append(ui_assistant_prefix())
                            screen_lines.append(result["answer"])
                    redraw()

                except EOFError:
                    return EXIT_OK
    except KeyboardInterrupt:
        safe_print(ui_warn("已中断。"), stream=sys.stdout)
        return EXIT_INTERRUPTED
    except Exception as e:
        eprint(ui_error(f"发生错误: {type(e).__name__}: {e}"))
        if FLAGS.verbose:
            traceback.print_exc(file=sys.stderr)
        return EXIT_GENERIC_ERROR

    return EXIT_OK


def resolve_fullscreen_mode(requested_mode: str):
    mode = (requested_mode or "auto").lower()
    support = supports_fullscreen()

    if mode == "off":
        return False, ""

    if mode == "on":
        if support.supported:
            return True, ""
        return False, f"全屏模式不可用，已降级为普通交互：{support.reason}"

    if support.supported:
        return True, ""
    return False, ""


def run_chat_entry(settings: Settings, rag: RAGEngine, app, no_banner: bool, fullscreen_mode: str) -> int:
    use_fullscreen, fallback_msg = resolve_fullscreen_mode(fullscreen_mode)
    if fallback_msg:
        safe_print(ui_warn(fallback_msg), stream=sys.stderr)

    if use_fullscreen:
        return run_chat_interactive_fullscreen(settings, rag, app, no_banner=no_banner)
    return run_chat_interactive_plain(settings, rag, app, no_banner=no_banner)


def run_chat_interactive(settings: Settings, rag: RAGEngine, app, no_banner: bool = False) -> int:
    return run_chat_interactive_plain(settings, rag, app, no_banner=no_banner)


def run_chat_once(settings: Settings, rag: RAGEngine, app, message: str, output: str) -> int:
    if not ensure_startup(settings, rag):
        return EXIT_GENERIC_ERROR

    try:
        chat_history = [get_system_message(), HumanMessage(content=message)]
        final_response, _ = run_agent_once(app, chat_history)
        if output == "json":
            json_out({"answer": final_response.content})
        else:
            out(ui_assistant_prefix(), force=True)
            out(final_response.content, force=True)
        return EXIT_OK
    except Exception as e:
        eprint(ui_error(f"发生错误: {type(e).__name__}: {e}"))
        if FLAGS.verbose:
            traceback.print_exc(file=sys.stderr)
        return EXIT_GENERIC_ERROR


def run_doctor(settings: Optional[Settings], rag: Optional[RAGEngine], output: str, runtime_err: Exception = None) -> int:
    if runtime_err is not None or settings is None or rag is None:
        result = {
            "config_ok": False,
            "config_error": str(runtime_err) if runtime_err else "runtime unavailable",
            "index_loaded": False,
            "model": None,
            "api_mode": None,
            "runtime_bootstrap_ok": False,
        }
        if output == "json":
            json_out(result)
        else:
            out(ui_section("doctor 检查结果"), force=True)
            out(f"- {ui_label('config_ok')} False", force=True)
            out(f"- {ui_label('config_error')} {result['config_error']}", force=True)
            out(f"- {ui_label('index_loaded')} False", force=True)
            out(f"- {ui_label('runtime_bootstrap_ok')} False", force=True)
            out(force=True)
        return EXIT_GENERIC_ERROR

    config_ok = True
    config_error = ""
    try:
        settings.validate_startup()
    except Exception as e:
        config_ok = False
        config_error = str(e)

    index_loaded = False
    if config_ok:
        try:
            index_loaded = bool(run_silenced(rag.load_index))
        except Exception:
            index_loaded = False

    result = {
        "config_ok": config_ok,
        "config_error": config_error,
        "index_loaded": index_loaded,
        "model": settings.model_name,
        "api_mode": settings.api_mode,
        "runtime_bootstrap_ok": True,
    }

    if output == "json":
        json_out(result)
    else:
        out(ui_section("doctor 检查结果"), force=True)
        out(f"- {ui_label('config_ok')} {result['config_ok']}", force=True)
        if config_error:
            out(f"- {ui_label('config_error')} {config_error}", force=True)
        out(f"- {ui_label('index_loaded')} {result['index_loaded']}", force=True)
        out(f"- {ui_label('model')} {result['model']}", force=True)
        out(f"- {ui_label('api_mode')} {result['api_mode']}", force=True)
        out(force=True)

    return EXIT_OK if config_ok else EXIT_GENERIC_ERROR


def resolve_chat_message(message_arg: str, use_stdin: bool) -> str:
    if use_stdin:
        return sys.stdin.read().strip()
    return (message_arg or "").strip()


def run_init(force: bool, dry_run: bool, output: str) -> int:
    example_path = os.path.join(os.getcwd(), ".env.example")
    env_path = os.path.join(os.getcwd(), ".env")

    if not os.path.exists(example_path):
        msg = f"未找到模板文件: {example_path}"
        if output == "json":
            json_out({"ok": False, "error": msg})
        else:
            eprint(msg)
        return EXIT_GENERIC_ERROR

    if os.path.exists(env_path) and not force:
        msg = f".env 已存在: {env_path}（使用 --force 可覆盖）"
        if output == "json":
            json_out({"ok": False, "error": msg, "env_path": env_path})
        else:
            eprint(msg)
        return EXIT_GENERIC_ERROR

    if dry_run:
        result = {
            "ok": True,
            "dry_run": True,
            "action": "copy",
            "from": example_path,
            "to": env_path,
            "overwrite": os.path.exists(env_path),
        }
        if output == "json":
            json_out(result)
        else:
            out(ui_hint("dry-run: 将执行 .env 初始化"), force=True)
            out(f"- {ui_label('from')} {example_path}", force=True)
            out(f"- {ui_label('to')} {env_path}", force=True)
        return EXIT_OK

    shutil.copyfile(example_path, env_path)
    result = {
        "ok": True,
        "dry_run": False,
        "action": "copied",
        "from": example_path,
        "to": env_path,
    }
    if output == "json":
        json_out(result)
    else:
        out(ui_section(f"已生成 .env: {env_path}"), force=True)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="个人智能笔记助理 CLI",
        epilog=(
            "示例:\n"
            "  agent chat -i\n"
            "  agent chat -m \"帮我总结本地笔记\"\n"
            "  echo \"帮我看一下最近的会议记录\" | agent chat --stdin\n"
            "  agent ask \"列出我有哪些项目TODO\"\n"
            "  agent init --dry-run\n"
            "  agent files --output json\n"
            "  agent doctor"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"agent {CLI_VERSION}")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quiet", action="store_true", help="精简输出（保留必要结果）")
    mode_group.add_argument("--verbose", action="store_true", help="详细输出（含底层日志）")
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="颜色输出模式（auto=自动，always=总是，never=关闭）",
    )

    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="聊天模式（支持交互/单次调用）")
    chat_parser.add_argument("-m", "--message", help="单次提问内容")
    chat_parser.add_argument("--stdin", action="store_true", help="从标准输入读取提问内容")
    chat_parser.add_argument("-i", "--interactive", action="store_true", help="强制进入交互模式")
    chat_parser.add_argument("--no-banner", action="store_true", help="交互模式下不显示启动横幅")
    chat_parser.add_argument("--output", choices=["text", "json"], default="text", help="单次调用输出格式")
    fullscreen_group = chat_parser.add_mutually_exclusive_group()
    fullscreen_group.add_argument(
        "--fullscreen-mode",
        dest="fullscreen_mode",
        choices=["auto", "on", "off"],
        help="交互显示模式（auto=自动探测，on=尽量启用全屏，off=关闭全屏）",
    )
    fullscreen_group.add_argument(
        "--fullscreen",
        dest="fullscreen_mode",
        action="store_const",
        const="on",
        help="兼容参数：等同 --fullscreen-mode on",
    )
    fullscreen_group.add_argument(
        "--no-fullscreen",
        dest="fullscreen_mode",
        action="store_const",
        const="off",
        help="兼容参数：等同 --fullscreen-mode off",
    )
    chat_parser.set_defaults(fullscreen_mode="auto")

    ask_parser = subparsers.add_parser("ask", help="单次提问（快捷命令）")
    ask_parser.add_argument("message", help="提问内容")
    ask_parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    files_parser = subparsers.add_parser("files", aliases=["ls"], help="列出当前可见本地笔记文件")
    files_parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    config_parser = subparsers.add_parser("config", aliases=["cfg"], help="查看当前关键配置")
    config_parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    metrics_parser = subparsers.add_parser("metrics", aliases=["stat"], help="查看会话指标快照")
    metrics_parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    reindex_parser = subparsers.add_parser(
        "index-rebuild", aliases=["reindex"], help="强制重建本地索引"
    )
    reindex_parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    init_parser = subparsers.add_parser("init", help="初始化 .env（由 .env.example 生成）")
    init_parser.add_argument("--force", action="store_true", help="覆盖已有 .env")
    init_parser.add_argument("--dry-run", action="store_true", help="只显示将执行的动作，不落盘")
    init_parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    doctor_parser = subparsers.add_parser("doctor", help="检查配置和索引健康状态")
    doctor_parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    return parser


def main() -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()

    FLAGS.quiet = bool(getattr(args, "quiet", False))
    FLAGS.verbose = bool(getattr(args, "verbose", False))
    FLAGS.color_mode = str(getattr(args, "color", "auto"))

    if args.command == "init":
        return run_init(args.force, args.dry_run, args.output)

    runtime, runtime_err = safe_build_runtime()

    if runtime is None:
        if args.command == "doctor":
            return run_doctor(None, None, args.output, runtime_err=runtime_err)
        eprint(ui_error(f"启动失败: {type(runtime_err).__name__}: {runtime_err}"))
        return EXIT_GENERIC_ERROR

    settings, rag, app = runtime

    if args.command is None:
        return run_chat_entry(settings, rag, app, no_banner=False, fullscreen_mode="auto")

    if args.command == "chat":
        if args.interactive or (not args.message and not args.stdin):
            return run_chat_entry(
                settings,
                rag,
                app,
                no_banner=args.no_banner,
                fullscreen_mode=getattr(args, "fullscreen_mode", "auto"),
            )

        if args.message and args.stdin:
            eprint(ui_error("chat 不能同时使用 --message 和 --stdin"))
            return EXIT_GENERIC_ERROR

        message = resolve_chat_message(args.message, args.stdin)
        if not message:
            eprint(ui_error("提问内容为空，请用 --message 传参或通过 --stdin 输入"))
            return EXIT_GENERIC_ERROR
        return run_chat_once(settings, rag, app, message, args.output)

    if args.command == "ask":
        return run_chat_once(settings, rag, app, args.message.strip(), args.output)

    if args.command == "doctor":
        return run_doctor(settings, rag, args.output)

    if not ensure_startup(settings, rag):
        return EXIT_GENERIC_ERROR

    if args.command in ["files", "ls"]:
        files = gather_files(rag)
        if args.output == "json":
            json_out({"files": files})
        else:
            print_files_text(files)
        return EXIT_OK

    if args.command in ["config", "cfg"]:
        config_data = gather_config(settings, rag)
        if args.output == "json":
            json_out(config_data)
        else:
            print_config_text(config_data)
        return EXIT_OK

    if args.command in ["metrics", "stat"]:
        metrics_data = gather_metrics()
        if args.output == "json":
            json_out(metrics_data)
        else:
            print_metrics_text(metrics_data)
        return EXIT_OK

    if args.command in ["index-rebuild", "reindex"]:
        if args.output == "json":
            run_silenced(rag.build_index)
            run_silenced(log_event, "manual_reindex", "console", result="triggered")
            json_out({"ok": True, "action": "index-rebuild"})
        else:
            has_progress = False

            def _on_reindex_line(line: str) -> None:
                nonlocal has_progress
                has_progress = True
                safe_print(line, stream=getattr(sys, "__stdout__", sys.stdout))

            stream_reindex_progress(rag, _on_reindex_line)
            if not has_progress:
                out(ui_hint("索引重建完成。"), force=True)
            run_silenced(log_event, "manual_reindex", "console", result="triggered")
        return EXIT_OK

    parser.print_help()
    return EXIT_GENERIC_ERROR


if __name__ == "__main__":
    sys.exit(main())
