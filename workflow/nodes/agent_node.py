import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import Settings
from core.domain.policies import (
    contains_restricted_keywords,
    should_force_vlm_only,
    should_list_notes_directly,
    should_use_retrieval,
)
from core.observability.logger import log_event
from core.observability.metrics import runtime_metrics
from core.observability.telemetry import traced_span
from core.security.audit import audit_security_event
from core.security.auth import verify_access_code as verify_access_code_hash
from core.security.input_guard import check_prompt_injection
from core.security.frequency_guard import SlidingWindowFrequencyGuard
from core.security.tool_guard import ensure_tool_allowed
from workflow.state import AgentState

if TYPE_CHECKING:
    from rag_engine import RAGEngine


@dataclass
class RestrictedQueryTicketRequired(Exception):
    ticket_id: str
    message: str = "检测到受限查询，请输入访问口令。"


class RestrictedQueryAborted(Exception):
    pass


def _verify_access_code(settings: Settings, user_input: str) -> bool:
    expected_hash = settings.access_code_hash
    if not expected_hash:
        print("[Warn] 访问口令哈希未配置")
        return False
    return verify_access_code_hash(user_input, expected_hash)


def _search_notes(
    rag: "RAGEngine",
    query: str,
    vlm_only: bool,
    settings: Settings,
    restricted_query_limiter: SlidingWindowFrequencyGuard,
    access_code: str | None = None,
    ticket_id: str | None = None,
    abort_ticket: bool = False,
) -> tuple[str, str | None]:
    print(f"\n[Tool] 正在检索本地笔记: {query}")

    is_password_query = contains_restricted_keywords(query)
    if is_password_query:
        if abort_ticket:
            audit_security_event(
                action="restricted_query",
                result="aborted",
                reason="user_aborted",
                ticket_id=ticket_id or "",
            )
            raise RestrictedQueryAborted()

        if not restricted_query_limiter.allow():
            audit_security_event(
                action="restricted_query",
                result="blocked",
                reason="frequency_guarded",
                query_preview="***REDACTED***",
            )
            return "", (
                "❌ 受限查询过于频繁，请稍后再试。"
                f"（频控阈值：每分钟 {settings.restricted_query_limit_per_minute} 次）"
            )

        if access_code is None:
            new_ticket_id = ticket_id or uuid.uuid4().hex[:12]
            audit_security_event(
                action="restricted_query",
                result="ticket_required",
                reason="awaiting_access_code",
                ticket_id=new_ticket_id,
            )
            raise RestrictedQueryTicketRequired(ticket_id=new_ticket_id)

        if not _verify_access_code(settings, access_code):
            audit_security_event(
                action="restricted_query",
                result="blocked",
                reason="auth_failed",
                ticket_id=ticket_id or "",
            )
            return "", "❌ 没有获取密码的权限。访问口令核验失败。"

        audit_security_event(
            action="restricted_query",
            result="allowed",
            reason="auth_passed",
            ticket_id=ticket_id or "",
        )
        print("[OK] 访问口令核验通过")

    ensure_tool_allowed(settings, "search_notes")
    context = rag.search(query, vlm_only=vlm_only)
    if not context:
        return "", None
    return context, None


def create_agent_node(
    settings: Settings,
    llm_client,
    rag: "RAGEngine",
    restricted_query_limiter: SlidingWindowFrequencyGuard,
) -> Callable[[AgentState], dict]:
    def agent_node(state: AgentState):
        trace_id = uuid.uuid4().hex[:10]
        started = time.perf_counter()
        messages = list(state["messages"])
        access_code = state.get("access_code")
        ticket_id = state.get("ticket_id")
        abort_ticket = bool(state.get("abort_ticket", False))

        last_user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        inj = check_prompt_injection(last_user_query)
        if inj.blocked:
            runtime_metrics.mark_error()
            audit_security_event(
                action="prompt_injection_check",
                result="blocked",
                reason=inj.reason,
                query_preview=last_user_query[:80],
            )
            return {"messages": [AIMessage(content="⚠️ 你的输入触发了防护策略（疑似提示词注入）。请改写后重试。")]}

        with traced_span("agent.request", trace_id=trace_id, query_len=len(last_user_query or "")) as span:
            log_event(
                "request_received",
                trace_id,
                api_mode=settings.api_mode,
                model=settings.model_name,
                query_preview=(
                    "***REDACTED***"
                    if contains_restricted_keywords(last_user_query)
                    else (last_user_query[:120] if last_user_query else "")
                ),
            )
            runtime_metrics.mark_request()

            force_vlm_only = should_force_vlm_only(last_user_query)

            if should_list_notes_directly(last_user_query):
                ensure_tool_allowed(settings, "list_note_files")
                files = rag.list_note_files()
                log_event("list_notes", trace_id, file_count=len(files))
                if files:
                    file_lines = "\n".join([f"- {f}" for f in files])
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    "我当前可访问到以下本地笔记文件：\n"
                                    f"{file_lines}\n\n"
                                    "你可以继续说：‘找一下某篇的重点’或‘总结这篇内容’。"
                                )
                            )
                        ]
                    }
                return {"messages": [AIMessage(content="当前 data/ 目录下未发现可用笔记文件（.md/.txt）。")]}

            if last_user_query and (
            should_use_retrieval(last_user_query)
            or contains_restricted_keywords(last_user_query)
        ):
                try:
                    local_context, blocked_message = _search_notes(
                        rag,
                        last_user_query,
                        force_vlm_only,
                        settings,
                        restricted_query_limiter,
                        access_code=access_code,
                        ticket_id=ticket_id,
                        abort_ticket=abort_ticket,
                    )
                except RestrictedQueryAborted:
                    return {
                        "messages": [AIMessage(content="查询已中止。")],
                        "security_status": "aborted",
                    }
                except RestrictedQueryTicketRequired as ticket:
                    return {
                        "messages": [AIMessage(content=ticket.message)],
                        "ticket_id": ticket.ticket_id,
                        "ticket_required": True,
                        "security_status": "awaiting_access_code",
                    }
                if blocked_message:
                    payload = {"messages": [AIMessage(content=blocked_message)]}
                    if access_code is not None and ticket_id:
                        payload["ticket_id"] = ticket_id
                        payload["security_status"] = "verification_failed"
                    return payload

                runtime_metrics.mark_retrieval(bool(local_context))
                log_event(
                    "retrieval_done",
                    trace_id,
                    hit=bool(local_context),
                    context_len=len(local_context),
                    vlm_only=force_vlm_only,
                )
                if local_context:
                    policy = (
                        "你必须优先基于【本地检索结果】回答，"
                        "不要说自己无法访问本地笔记；若检索结果中有来源，请明确引用。"
                    )
                    if force_vlm_only:
                        policy += " 用户明确要求不使用OCR，请仅依据【图片语义理解】与图片向量信息回答，不要引用OCR字段。"
                    messages.append(SystemMessage(content=policy))
                    messages.append(
                        HumanMessage(
                            content=(
                                "【本地检索结果】\n"
                                f"找到以下相关笔记片段：\n{local_context}\n\n"
                                f"【用户问题】{last_user_query}\n"
                                "请根据以上检索结果直接回答。"
                            )
                        )
                    )
                else:
                    messages.append(
                        SystemMessage(
                            content=(
                                "本轮本地笔记检索未命中，请明确告知用户未找到相关笔记，"
                                "不要假装已读取到本地内容。"
                            )
                        )
                    )
            elif last_user_query:
                log_event("retrieval_skipped", trace_id, reason="smalltalk_or_non_retrieval_intent")

            try:
                reply = llm_client.call_model(messages)
            except Exception as e:
                runtime_metrics.mark_error()
                log_event("response_failed", trace_id, error=type(e).__name__, detail=str(e)[:200])
                raise

            cost_ms = int((time.perf_counter() - started) * 1000)
            runtime_metrics.mark_latency(cost_ms)
            log_event(
                "response_generated",
                trace_id,
                latency_ms=cost_ms,
                output_len=len(reply or ""),
                metrics=runtime_metrics.snapshot(),
            )
            if span is not None:
                span.set_attribute("agent.latency_ms", cost_ms)
                span.set_attribute("agent.retrieval_hits", runtime_metrics.retrieval_hit_total)
            return {"messages": [AIMessage(content=reply)]}

    return agent_node


