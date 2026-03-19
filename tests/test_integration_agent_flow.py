import hashlib
import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings
from core.security.frequency_guard import SlidingWindowFrequencyGuard
from workflow.nodes.agent_node import create_agent_node


class _FakeLLM:
    def __init__(self):
        self.last_messages = None
        self.last_on_delta = None

    def call_model(self, messages, on_delta=None):
        self.last_messages = messages
        self.last_on_delta = on_delta
        if callable(on_delta):
            on_delta("ok")
        return "ok"


class _FakeRAG:
    def __init__(self, context=""):
        self.context = context
        self.last_vlm_only = None

    def search(self, query, vlm_only=False):
        self.last_vlm_only = vlm_only
        return self.context

    def list_note_files(self):
        return ["a.md"]


class TestAgentIntegrationFlow(unittest.TestCase):
    def _build_settings(self, access_code_hash: str = "") -> Settings:
        return Settings(
            model_name="test",
            api_mode="chat",
            request_timeout=10,
            request_max_retries=0,
            request_retry_backoff_ms=100,
            base_url="http://localhost",
            api_key="k",
            access_code_hash=access_code_hash,
            restricted_query_limit_per_minute=3,
            allowed_tools=("search_notes", "list_note_files"),
            log_file="observability/test_events.jsonl",
        )

    def test_retrieval_path_injects_context(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nhello")
        agent = create_agent_node(self._build_settings(), llm, rag, SlidingWindowFrequencyGuard(3, 60))

        state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="帮我检索Agent总结")]}
        out = agent(state)
        self.assertIn("messages", out)
        self.assertEqual(out["messages"][0].content, "ok")
        self.assertIsNotNone(llm.last_messages)
        merged = "\n".join([str(m.content) for m in llm.last_messages])
        self.assertIn("本地检索结果", merged)

    def test_restricted_query_requires_ticket_first(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nsecret")
        security_hash = hashlib.sha256("secret".encode("utf-8")).hexdigest()
        agent = create_agent_node(self._build_settings(access_code_hash=security_hash), llm, rag, SlidingWindowFrequencyGuard(3, 60))

        state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="帮我找密码")]}
        out = agent(state)
        self.assertTrue(out.get("ticket_required"))
        self.assertEqual(out.get("security_status"), "awaiting_access_code")
        self.assertIn("ticket_id", out)
        self.assertIsNone(llm.last_messages)

    def test_restricted_query_wrong_key_returns_failed_status(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nsecret")
        security_hash = hashlib.sha256("secret".encode("utf-8")).hexdigest()
        agent = create_agent_node(self._build_settings(access_code_hash=security_hash), llm, rag, SlidingWindowFrequencyGuard(3, 60))

        state = {
            "messages": [SystemMessage(content="sys"), HumanMessage(content="帮我找密码")],
            "access_code": "wrong",
            "ticket_id": "abc123",
        }
        out = agent(state)
        self.assertEqual(out.get("security_status"), "verification_failed")
        self.assertIn("访问口令核验失败", out["messages"][0].content)

    def test_restricted_query_correct_key_allows_retrieval(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nsecret")
        security_hash = hashlib.sha256("secret".encode("utf-8")).hexdigest()
        agent = create_agent_node(self._build_settings(access_code_hash=security_hash), llm, rag, SlidingWindowFrequencyGuard(3, 60))

        state = {
            "messages": [SystemMessage(content="sys"), HumanMessage(content="帮我找密码")],
            "access_code": "secret",
            "ticket_id": "abc123",
        }
        out = agent(state)
        self.assertEqual(out["messages"][0].content, "ok")
        self.assertIsNotNone(llm.last_messages)

    def test_restricted_query_aborted(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nsecret")
        security_hash = hashlib.sha256("secret".encode("utf-8")).hexdigest()
        agent = create_agent_node(self._build_settings(access_code_hash=security_hash), llm, rag, SlidingWindowFrequencyGuard(3, 60))

        state = {
            "messages": [SystemMessage(content="sys"), HumanMessage(content="帮我找密码")],
            "ticket_id": "abc123",
            "abort_ticket": True,
        }
        out = agent(state)
        self.assertEqual(out.get("security_status"), "aborted")
        self.assertIn("查询已中止", out["messages"][0].content)

    def test_restricted_query_frequency_guarded(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nsecret")
        security_hash = hashlib.sha256("secret".encode("utf-8")).hexdigest()
        limiter = SlidingWindowFrequencyGuard(limit=1, window_seconds=60)
        agent = create_agent_node(self._build_settings(access_code_hash=security_hash), llm, rag, limiter)

        first = agent({"messages": [SystemMessage(content="sys"), HumanMessage(content="帮我找密码")]})
        self.assertTrue(first.get("ticket_required"))
        self.assertEqual(first.get("security_status"), "awaiting_access_code")

        second = agent({"messages": [SystemMessage(content="sys"), HumanMessage(content="帮我找密码")]})
        self.assertFalse(second.get("ticket_required", False))
        self.assertNotIn("ticket_id", second)
        self.assertIn("受限查询过于频繁", second["messages"][0].content)

    def test_stream_sink_passed_to_llm(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nhello")
        agent = create_agent_node(self._build_settings(), llm, rag, SlidingWindowFrequencyGuard(3, 60))

        chunks = []
        state = {
            "messages": [SystemMessage(content="sys"), HumanMessage(content="帮我检索Agent总结")],
            "stream_sink": chunks.append,
        }
        out = agent(state)

        self.assertEqual(out["messages"][0].content, "ok")
        self.assertTrue(callable(llm.last_on_delta))
        self.assertEqual(chunks, ["ok"])



