import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings
from core.security.rate_limit import SlidingWindowRateLimiter
from workflow.nodes.agent_node import create_agent_node


class _FakeLLM:
    def __init__(self):
        self.last_messages = None

    def call_model(self, messages):
        self.last_messages = messages
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
    def _build_settings(self) -> Settings:
        return Settings(
            model_name="test",
            api_mode="chat",
            request_timeout=10,
            request_max_retries=0,
            request_retry_backoff_ms=100,
            base_url="http://localhost",
            api_key="k",
            security_key_hash="",
            sensitive_query_limit_per_minute=3,
            allowed_tools=("search_notes", "list_note_files"),
            log_file="observability/test_events.jsonl",
        )

    def test_retrieval_path_injects_context(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[文本来源]\nhello")
        agent = create_agent_node(self._build_settings(), llm, rag, SlidingWindowRateLimiter(3, 60))

        state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="帮我检索Agent总结")]}
        out = agent(state)
        self.assertIn("messages", out)
        self.assertEqual(out["messages"][0].content, "ok")
        self.assertIsNotNone(llm.last_messages)
        merged = "\n".join([str(m.content) for m in llm.last_messages])
        self.assertIn("本地检索结果", merged)

    def test_no_ocr_marker_sets_vlm_only(self):
        llm = _FakeLLM()
        rag = _FakeRAG(context="[图片文本来源]\nimg")
        agent = create_agent_node(self._build_settings(), llm, rag, SlidingWindowRateLimiter(3, 60))

        state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="不要ocr，检索这张图片")]} 
        agent(state)
        self.assertTrue(rag.last_vlm_only)


if __name__ == "__main__":
    unittest.main()
